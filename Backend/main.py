
import io
import json
import re
import whisper
import shutil
import base64
import pandas as pd
import logging
from xhtml2pdf import pisa
import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
from sql_agent import SqlAgent
from connection_pooling import initialize_connection_pool, DatabaseCursor
import traceback
from contextlib import asynccontextmanager
import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import openai

# Load env variables
load_dotenv()
LOGS_DIR = os.getenv("LOGS_DIR")

os.makedirs(LOGS_DIR, exist_ok=True)

# Set up logging with proper formatting
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR + "/server.log"),
    ]
)
logger = logging.getLogger(__name__)

# Custom streaming callback handler
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

# Configuration class for application settings
class Settings:
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MAX_EXCEL_ROWS: int = int(os.getenv("MAX_EXCEL_ROWS", "10000"))
    MAX_SAMPLE_ROWS: int = int(os.getenv("MAX_SAMPLE_ROWS", "5"))
    MAX_SAMPLE_COLUMNS: int = int(os.getenv("MAX_SAMPLE_COLUMNS", "5"))
    DDL_PATH: str = os.getenv("DDL_PATH", "Data/ddls.json")
    TEMPLATE_DIR: str = os.getenv("TEMPLATE_DIR", "templates")
    EXAMPLES_PATH: str = os.getenv("EXAMPLES_PATH", "Data/examples.json")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))


@lru_cache()
def get_settings():
    return Settings()


# Create temp directory if it doesn't exist
os.makedirs(get_settings().TEMP_DIR, exist_ok=True)


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize resources
    logger.info("Initializing application resources...")
    initialize_connection_pool()
    # Yield control to the application
    yield
    # Shutdown: clean up resources
    logger.info("Shutting down application resources...")
    from connection_pooling import pool
    if pool:
        pool.close_all()
    # Clean up temp directory
    for file in os.listdir(get_settings().TEMP_DIR):
        try:
            os.remove(os.path.join(get_settings().TEMP_DIR, file))
        except Exception as e:
            logger.warning(f"Failed to remove temp file {file}: {e}")


# Initialize FastAPI with lifespan
app = FastAPI(
    title="AI SQL Chatbot API",
    description="API for AI SQL chatbot assistant",
    version="1.0.0",
    debug=False,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Has-Excel"],
)


# ============ Pydantic Models ============
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")


class QueryResult(BaseModel):
    response: str = Field(..., description="Response from the chatbot")
    has_excel: bool = Field(default=False, description="Indicates if Excel data is available")
    excel_data: Optional[bytes] = Field(default=None, description="Excel data in bytes format")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")


# ============ Dependencies ============
@lru_cache()
def get_whisper_model():
    """Return the whisper model as a singleton."""
    settings = get_settings()
    return whisper.load_model(settings.WHISPER_MODEL)


class ChatBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Initialize OpenAI client
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize LangChain ChatOpenAI
        self.llm_client = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize streaming client for streaming responses
        self.streaming_client = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            openai_api_key=settings.OPENAI_API_KEY,
            streaming=True
        )
        
        # Initialize SQL Agent with OpenAI client
        self.llm = SqlAgent(client=self.llm_client, model=settings.OPENAI_MODEL)
        self.load_ddl()
        self.last_query_excel_data = None
        self.last_query_chart_data = None

    def load_ddl(self):
        """Load all DDL schema definitions."""
        try:
            with open(self.settings.DDL_PATH, 'r', encoding='utf-8') as ddls_file:
                ddls = json.load(ddls_file).get("ddls", [])
            self.llm.ddl_list = ddls
            logger.info(f"Loaded {len(ddls)} DDL schemas successfully")
        except Exception as e:
            logger.error(f"Failed to load DDL schemas: {str(e)}")
            self.llm.ddl_list = []

    def execute_query(self, query: str) -> tuple:
        """Execute SQL query and return results and column names."""
        result = self.llm.execute_sql(query)
        if not result:
            return None, None

        with DatabaseCursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]            
            # Convert to DataFrame with size limit
            df = pd.DataFrame(result, columns=columns)
            if len(df) > self.settings.MAX_EXCEL_ROWS:
                logger.warning(f"Query result exceeded max rows ({len(df)}), truncating")
                df = df.head(self.settings.MAX_EXCEL_ROWS)

            sample_result = df
            if len(result) > self.settings.MAX_SAMPLE_ROWS:
                # Reduce number of rows for display
                sample_result = sample_result.head(self.settings.MAX_SAMPLE_ROWS)
            
            if len(columns) > self.settings.MAX_SAMPLE_COLUMNS:
                # Reduce number of columns for display
                sample_result = sample_result.iloc[:, :self.settings.MAX_SAMPLE_COLUMNS]

            self.last_query_chart_data = sample_result
            if len(df) > len(sample_result):
                self.last_query_excel_data = df
            return sample_result
    
    def get_excel_download_data(self) -> Optional[bytes]:
        """Return Excel data as bytes."""
        if self.last_query_excel_data is not None and not self.last_query_excel_data.empty:
            try:
                # Create a BytesIO object for the Excel file
                buffer = io.BytesIO()
                
                # Write dataframe to Excel
                self.last_query_excel_data.to_excel(buffer, index=False)
                
                # Clear the dataframe to free memory
                excel_data = buffer.getvalue()
                buffer.close()
                self.last_query_excel_data = None

                return excel_data
            except Exception as e:
                logger.error(f"Error generating Excel file: {e}")
                return None
        return None

    def get_resphrase_sql_answer_prompt(self, question: str, result: list) -> str:
        """Create prompt for SQL answer rephrasing."""
        return f"""
            **You are an SQL Assistant.** Your task is to generate a well-structured and relevant response in **English** based strictly on the provided SQL result.  
            
            ---

            ### **Inputs:** 

            User Question: {question}
            SQL Result:
            {result}

            ---
            
            ### **Instructions (Read and Follow Carefully):**  
            1. **Strictly base your response on the SQL result and in english.** 
            2. **Strictly Do NOT** generate any assumptions or explanations for the user!! Your job is just to provide SQL result and NOTHING ELSE.  
            3. You can provide a clear, Concise and direct response OR You may use a table format in your response if data needs a tabular format to be shown.
            4. Never try modifying SQL result data neither by adding or removing rows.
            5. Keep in mind that SQL result passed to you is a sample of the actual data. You should not assume or tell the user anything beyond what is provided in.
            ---
            
            Begin! 
            """
            
    def get_general_question_answer_prompt(self) -> str:
        """Create prompt for general question answering."""
        return"""
            You're a General purpose Assistant named DeepInsight, Given the user question generate a **direct** and **concise** response Strictly in English.

            begin!
            """

    def get_generate_chart_prompt(self, question: str, result: Dict[str, Any]) -> str:
        """Create prompt for chart generation."""
        return f"""
            **You are a Chart Recommendation Assistant.** Your task is to analyze the SQL query result and suggest the most appropriate chart type and configuration for visualization.
            
            ---
            
            ### **Inputs:**
            
            User Question: {question}
            SQL Result:
            {result}
            
            ---
            
            ### **Instructions:**
            1. Analyze the data and determine the most appropriate chart type (bar, line, pie, scatter, etc.)
            2. Identify which columns should be used for x-axis and y-axis (columns names should be copied exactly from the result)
            3. Suggest appropriate titles, labels, and other configuration options
            4. Return your recommendation in JSON format
            
            ---
            
            ### **Response Format:**
            {{
                "chart_type": "bar|line|pie|scatter",
                "options": {{
                    "x_column": "column_name",
                    "y_column": "column_name",
                    "title": "Suggested chart title in English",
                    "xlabel": "X-axis label in English",
                    "ylabel": "Y-axis label in English"
                }}
            }}
            
            Begin!
        """
        
    def is_sql_question(self, question: str) -> bool:
        """Determine if the question requires SQL processing."""
        sql_keywords = ["BASE", "CLIENT", "CUSTOMER", "ORDER", "PRODUCT", "CATEGORIE", "QUANTITY", "EXPEDITION", "DATABASE"]
        question_words = question.lower().split()

        for word in question_words:
            if any(fuzz.ratio(word, keyword.lower()) > 70 for keyword in sql_keywords): 
                return True
        return False

         
    
    def verify_sql_query(self, sql_query: str) -> str:
        """Verify SQL query for security."""
        restricted_commands = ["UPDATE", "DELETE", "INSERT", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT"]
        if re.search(r"\b(" + "|".join(restricted_commands) + r")\b", sql_query, re.IGNORECASE):
            raise ValueError("You don't have permission to execute this query.")
        return sql_query
    
    def query_to_sql(self, question: str) -> tuple:
        """Generate and execute SQL query from question."""
        try:
            query = self.llm.generate_sql(question)
            logger.info(f"Generated SQL Query: {query}")
            verified_query = self.verify_sql_query(query)
            result = self.execute_query(verified_query)
            return result
        
        except Exception as e:
            logger.error(f"SQL query execution error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError("An error occurred during SQL query execution.")
        
    async def generate_chart(self, question: str, result: pd.DataFrame) -> str:
        """Generate a chart for corresponding response."""
        try:
            # Convert DataFrame to dict for the prompt
            chart_data = {
                "data": result.values.tolist(),
                "columns": result.columns.tolist()
            }
            
            prompt = self.get_generate_chart_prompt(question, chart_data)
            
            # Get chart configuration from OpenAI
            messages = [HumanMessage(content=prompt)]
            chart_config_response = await self.llm_client.ainvoke(messages)
            
            chart_config = chart_config_response.content
            
            # Parse the chart configuration
            try:
                chart_type, chart_options = self._parse_chart_config(chart_config)
            except Exception as e:
                logger.error(f"Error parsing chart configuration: {str(e)}")
                return "Error parsing chart configuration"
            
            # Generate the chart
            chart_image = self._create_chart(chart_data, chart_type, chart_options)
            
            # Clear chart data
            self.last_query_chart_data = None
            
            return chart_image
            
        except Exception as e:
            logger.error(f"Chart generation error: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error generating chart: {str(e)}"
    
    def plot_pie_chart(self, title: str, result: list) -> str:
        """Generate a pie chart from data."""
        try:
            categories = [row[0] for row in result]
            quantities = [row[1] for row in result]

            plt.figure(figsize=(8, 8))
            wedges, texts, autotexts = plt.pie(quantities, labels=categories, autopct='%1.1f%%', startangle=140)
            plt.title(title)
            plt.legend(wedges, categories, title="Default Categories", loc="best")

            # Save chart to bytes
            image_stream = io.BytesIO()
            plt.savefig(image_stream, format='png', dpi=100)
            plt.close()
            image_stream.seek(0)

            image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')
            image_stream.close()

            return image_base64
        except Exception as e:
            logger.error(f"Error generating pie chart: {str(e)}")
            return ""
    
    def plot_bar_chart(self, title: str, x: int, y: int, x_label: str, y_label: str, result: list) -> str:
        """Generate a bar chart from data."""
        try:
            x_data = [row[x] for row in result]
            y_data = [row[y] for row in result]
            
            plt.figure(figsize=(8, 8))
            plt.bar(x_data, y_data)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            
            # Rotate x labels if there are many categories
            if len(x_data) > 5:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save chart to bytes
            image_stream = io.BytesIO()
            plt.savefig(image_stream, format='png', dpi=100)
            plt.close()
            image_stream.seek(0)
            image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')
            image_stream.close()
            
            return image_base64
        except Exception as e:
            logger.error(f"Error generating bar chart: {str(e)}")
            return ""
    
    async def generate_pdf(self, question: str) -> Dict[str, Any]:
        """Generate PDF report from order ID."""
        try:
            # Extract order ID 
            match = re.search(r"order\s+(\S+)", question, re.IGNORECASE)
            if not match:
                raise ValueError("Invalid format. Use 'order [ID]'")
            
            order_id = match.group(1).strip()
            logger.info(f"Generating PDF for order ID: {order_id}")
            
            results = []
            query_columns_list = []
            
            # List of queries to execute
            queries = [
                f"SELECT * FROM ORDERS WHERE id = {order_id}",
            ]
            
            # Execute all queries
            for query in queries:
                try:
                    with DatabaseCursor() as cursor:
                        cursor.execute(query)
                        columns = [desc[0] for desc in cursor.description]
                        result_data = cursor.fetchall()
                        
                        if not result_data and not results: 
                            raise ValueError(f"No data found for order {order_id}")
                        
                        # Append results and columns
                        results.append(result_data)
                        query_columns_list.append(columns)
                except Exception as e:
                    logger.warning(f"Error executing query {query}: {str(e)}")
                    # Add empty result to maintain order
                    results.append([])
                    query_columns_list.append([])
            
            
            # Create HTML template for the PDF
            html_content = self._render_template("base_report.html", {
                "order_id": order_id,
                "query_columns_list": query_columns_list,
                "results": results,
                "current_date": datetime.datetime.now().strftime('%d/%m/%Y at %H:%M')
            })
                
            # Generate PDF from HTML
            pdf = await self._html_to_pdf(html_content)
            
            return {
                "pdf": pdf,
                "filename": f"order_{order_id}.pdf",
                "command_id": order_id
            }
        except ValueError as e:
            logger.warning(f"PDF generation validation error: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"PDF generation error: {str(e)}"}
        
    def _render_template(self, template_file: str, context: Dict[str, Any]) -> str:
        """Render HTML template with given context."""
        try:
            from jinja2 import Environment, FileSystemLoader
            
            # Set up Jinja2 environment
            env = Environment(loader=FileSystemLoader(self.settings.TEMPLATE_DIR))
            template = env.get_template(template_file)
            
            # Render template with context
            return template.render(**context)
        except Exception as e:
            logger.error(f"Template rendering error: {str(e)}")
            raise
                    
    async def _html_to_pdf(self, html_content: str) -> bytes:
        """Convert HTML content to PDF bytes."""
        # Create BytesIO object
        result = io.BytesIO()
        
        try:
            # Convert HTML to PDF
            pdf_status = pisa.CreatePDF(
                src=html_content,
                dest=result
            )
            
            # Check if PDF generation was successful
            if pdf_status.err:
                raise Exception("PDF generation failed: " + str(pdf_status.err))
            
            # Get PDF bytes
            pdf = result.getvalue()
            return pdf
        except Exception as e:
            logger.error(f"HTML to PDF conversion error: {str(e)}")
            raise
        finally:
            result.close()

    def _parse_chart_config(self, config_text: str) -> tuple:
        """Parse chart configuration from LLM response."""
        # Extract JSON configuration if present
        json_pattern = r'{[\s\S]*}'
        json_match = re.search(json_pattern, config_text)
        
        if json_match:
            try:
                config = json.loads(json_match.group(0))
                chart_type = config.get('chart_type', 'bar')
                chart_options = config.get('options', {})
                return chart_type, chart_options
            except json.JSONDecodeError:
                pass
        
        # Fallback parsing
        chart_type = 'bar'  # Default
        chart_options = {}
        
        if "pie" in config_text.lower():
            chart_type = "pie"
        elif "line" in config_text.lower():
            chart_type = "line"
        elif "scatter" in config_text.lower():
            chart_type = "scatter"
            
        # Extract column names
        x_column_match = re.search(r'x[\s_-]*column["\s:]*([^",\n]+)', config_text, re.IGNORECASE)
        if x_column_match:
            chart_options["x_column"] = x_column_match.group(1).strip()
            
        y_column_match = re.search(r'y[\s_-]*column["\s:]*([^",\n]+)', config_text, re.IGNORECASE)
        if y_column_match:
            chart_options["y_column"] = y_column_match.group(1).strip()
            
        title_match = re.search(r'title["\s:]*([^",\n]+)', config_text, re.IGNORECASE)
        if title_match:
            chart_options["title"] = title_match.group(1).strip()
            
        return chart_type, chart_options

    def _create_chart(self, data: Dict[str, Any], chart_type: str, options: Dict[str, Any]) -> str:
        """Create chart based on data and options."""
        try:
            # Convert data to pandas DataFrame
            if isinstance(data['data'], list) and len(data['data']) > 0:
                df = pd.DataFrame(data['data'], columns=data['columns'])
            else:
                df = pd.DataFrame(data['data'])
            
            # Default column selection logic
            x_column = options.get('x_column', df.columns[0] if len(df.columns) > 0 else None)
            y_column = options.get('y_column', df.columns[1] if len(df.columns) > 1 else None)
            
            # Validate columns exist in dataframe
            if x_column not in df.columns or y_column not in df.columns:
                logger.warning(f"Column not found in dataframe. Using defaults.")
                x_column = df.columns[0] if len(df.columns) > 0 else None
                y_column = df.columns[1] if len(df.columns) > 1 else None
                
            if x_column is None or y_column is None:
                raise ValueError("Cannot create chart: missing x or y column data")
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create chart based on type
            if chart_type == 'bar':
                df.plot(kind='bar', x=x_column, y=y_column, ax=ax)
            elif chart_type == 'line':
                df.plot(kind='line', x=x_column, y=y_column, ax=ax)
            elif chart_type == 'pie':
                df.plot(kind='pie', y=y_column, ax=ax)
            elif chart_type == 'scatter':
                df.plot(kind='scatter', x=x_column, y=y_column, ax=ax)
            else:
                # Default to bar chart
                df.plot(kind='bar', x=x_column, y=y_column, ax=ax)
            
            # Apply options
            title = options.get('title', f'Chart for: {y_column} by {x_column}')
            ax.set_title(title)
            ax.set_xlabel(options.get('xlabel', x_column))
            ax.set_ylabel(options.get('ylabel', y_column))
            
            # Save to BytesIO buffer
            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            
            # Encode as base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            buffer.close()
            
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Chart creation error: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    async def process_sql_question(self, question: str, result: list):
        """Process SQL question and stream response."""
        try:
            prompt = self.get_resphrase_sql_answer_prompt(question, result)
            
            # Create streaming callback
            callback = StreamingCallbackHandler()
            
            # Stream response from OpenAI using LangChain
            messages = [HumanMessage(content=prompt)]
            
            # Use direct OpenAI client for streaming
            response = await openai.ChatCompletion.acreate(
                model=self.settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.settings.OPENAI_TEMPERATURE,
                max_tokens=self.settings.OPENAI_MAX_TOKENS,
                stream=True,
                api_key=self.settings.OPENAI_API_KEY
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error processing SQL question: {e}")
            logger.error(traceback.format_exc())
            yield f"Error: {str(e)}"

    async def process_general_question(self, history):
        """Process general question and stream response."""
        try:
            prompt = self.get_general_question_answer_prompt()
            
            # Convert history to OpenAI format
            messages = [{"role": "system", "content": prompt}]
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
            
            # Stream response from OpenAI
            response = await openai.ChatCompletion.acreate(
                model=self.settings.OPENAI_MODEL,
                messages=messages,
                temperature=self.settings.OPENAI_TEMPERATURE,
                max_tokens=self.settings.OPENAI_MAX_TOKENS,
                stream=True,
                api_key=self.settings.OPENAI_API_KEY
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error processing general question: {e}")
            logger.error(traceback.format_exc())
            yield f"Error: {str(e)}"

    def clear_messages(self):
        """Clear message history."""
        self.llm.clear_messages()


chatbot = ChatBot(get_settings())

# ============ API Endpoints ============
@app.post("/api/query")
async def process_query(messages: List[Message]):
    """
    Process a user query and stream the response.
    """
    question = messages[-1].content
    if chatbot.is_sql_question(question):
       result = chatbot.query_to_sql(question)
       response = chatbot.process_sql_question(question, result)
    else:
        response = chatbot.process_general_question(messages)
    
    # Set headers based on whether Excel data is included
    headers = {"X-Has-Excel": "true" if chatbot.last_query_excel_data is not None else "false"}
    
    # Return the response as streaming
    return StreamingResponse(
        response,
        media_type="text/plain",
        headers=headers
    )

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"}

@app.get("/api/excel")
async def get_excel():
    """
    Get excel data.
    """
    buffer = chatbot.get_excel_download_data()
    if buffer:
        return Response(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=data.xlsx"}
        )
    else:
        return Response(content="No data available", status_code=404)


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), whisper_model: Any = Depends(get_whisper_model)):
    """Transcribes the uploaded WAV file using Whisper."""
    
    # Save uploaded file temporarily
    temp_path = Path(f"temp_{file.filename}")
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and transcribe audio
    audio = whisper.load_audio(str(temp_path))
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisper_model, mel, options)

    # Remove temporary file
    temp_path.unlink()

    return JSONResponse({"transcription": result.text}, status_code=200)
@app.get("/generate_chart/")
async def generate_chart(question: str):
    """Generate a chart based on the last query results."""
    try:
        # Get the last query data
        data = chatbot.last_query_chart_data
        
        if data is None or data.empty:
            return JSONResponse(
                status_code=400,
                content={"error": "Insufficient Data to generate un graphique."}
            )
                
        # Generate chart
        chart_image = await chatbot.generate_chart(question, data)
        
        if chart_image.startswith("Error"):
            return JSONResponse(
                status_code=400,
                content={"error": chart_image}
            )
                # Return the base64-encoded chart image
        return JSONResponse(
            status_code=200,
            content={"chart": chart_image}
        )
        
    except IndexError:
        return JSONResponse(
            status_code=400,
            content={"error": "No query has been executed."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error generating graphic: {str(e)}"},
        )
   
@app.get("/generate_pdf/")
async def generate_pdf(question: str):
    try:        
        # Call the generate_pdf method
        result = await chatbot.generate_pdf(question)
        
        if "error" in result:
            return Response(status_code=404, content="An error occured during generating PDF, Verify that order id exists in the database.")
        
        # Get PDF bytes from result
        pdf_bytes = result["pdf"]
        filename = result["filename"]
        
        # Return the PDF as a downloadable file
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/retrain_examples")
async def retrain_model(file: UploadFile = File(...)):
    try:
        temp_path = Path(f"temp_{file.filename}")
        # Save the uploaded JSON file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Calculate and save embeddings
        chatbot.llm.calculate_embeddings(str(temp_path))

        # Define the target path in the Data directory
        target_path = chatbot.settings.EXAMPLES_PATH

        # Move and rename the uploaded file to the Data folder as examples.json
        shutil.move(str(temp_path), str(target_path))

        return Response("Model retrained successfully, examples saved.", status_code=200)

    except Exception as e:
        temp_path.unlink()
        return Response("Couldn't retrain model on new examples verify your file is in the correct format", status_code=404)

    
@app.get("/clear_messages/")
async def clear_messages():
    chatbot.clear_messages()
    return Response(status_code=200)
