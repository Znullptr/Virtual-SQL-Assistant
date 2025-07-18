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
from ollama import Client

# Load env variables
load_dotenv()
LOGS_DIR = os.environ.get("LOGS_DIR")

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

# Configuration class for application settings
class Settings:
    WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "base")
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "mistral")
    MAX_EXCEL_ROWS: int = int(os.environ.get("MAX_EXCEL_ROWS", "10000"))
    MAX_SAMPLE_ROWS: int = int(os.environ.get("MAX_SAMPLE_ROWS", "8"))
    MAX_SAMPLE_COLUMNS: int = int(os.environ.get("MAX_SAMPLE_COLUMNS", "6"))
    DDL_PATH: str = os.environ.get("DDL_PATH", "Data/ddls.json")
    TEMPLATE_DIR: str = os.environ.get("TEMPLATE_DIR", "templates")
    LLM_HOST: str = "http://"+ os.environ.get("OLLAMA_HOST", "127.0.0.1" ) + ":" + os.environ.get("OLLAMA_PORT", "11434")
    EXAMPLES_PATH: str = os.environ.get("EXAMPLES_PATH", "Data/examples.json")
    TEMP_DIR: str = os.environ.get("TEMP_DIR", "temp")


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
        self.client = Client(host=settings.LLM_HOST)
        self.llm = SqlAgent(client=self.client, model=settings.LLM_MODEL)
        self.load_ddl()
        self.last_query_excel_data = None

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
            sample_result = result
            
            if len(result) >= 5:
                # Convert to DataFrame with size limit
                df = pd.DataFrame(result, columns=columns)
                if len(df) > self.settings.MAX_EXCEL_ROWS:
                    logger.warning(f"Query result exceeded max rows ({len(df)}), truncating")
                    df = df.head(self.settings.MAX_EXCEL_ROWS)
                self.last_query_excel_data = df

            if len(result) >= self.settings.MAX_SAMPLE_ROWS:
                # Reduce number of rows for display
                sample_result = result[:self.settings.MAX_SAMPLE_ROWS]
            
            if len(columns) >= self.settings.MAX_SAMPLE_COLUMNS:
                # Reduce number of columns for display
                columns = columns[:self.settings.MAX_SAMPLE_COLUMNS]
                sample_result = [row[:self.settings.MAX_SAMPLE_COLUMNS] for row in sample_result]
            
            return sample_result, columns
    
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

    def get_resphrase_sql_answer_prompt(self, question: str, column_names: list, result: list) -> str:
        """Create prompt for SQL answer rephrasing."""
        return f"""
            **You are an SQL Assistant.** Your task is to generate a well-structured and relevant response in **English** based strictly on the provided SQL result.  
            
            ---

            ### **Inputs:** 

            User Question: {question}
            Column Names: {column_names}
            SQL Result:
            {result}

            ---
            
            ### **Instructions (Read and Follow Carefully):**  
            1. **Strictly base your response on the SQL result.** Do NOT generate assumptions, explanations, or additional insights beyond what is provided.  
            2. **Always respond in English.**  
            3. You can provide a clear, Concise and direct response OR You may use a table format in your response but it's NOT ALWAYS Needed.
            4. Never try modifying SQL result data neither by adding or removing rows.
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
            raise ValueError("Vous n'avez pas l'autorisation d'exécuter une telle action sur la base de données sélectionnée.")
        return sql_query
    
    def query_to_sql(self, question: str) -> tuple:
        """Generate and execute SQL query from question."""
        try:
            query = self.llm.generate_sql(question)
            logger.info(f"Generated SQL Query: {query}")
            verified_query = self.verify_sql_query(query)
            print(verified_query)
            result, columns = self.execute_query(verified_query)
            return result, columns
        
        except Exception as e:
            logger.error(f"SQL query execution error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError("Une erreur s'est produite pendant l'exécution de la requête")
        
    async def generate_chart(self, question: str, result: pd.DataFrame) -> str:
        """Generate a chart for corresponding response."""
        try:
            # Convert DataFrame to dict for the prompt
            chart_data = {
                "data": result.values.tolist(),
                "columns": result.columns.tolist()
            }
            
            prompt = self.get_generate_chart_prompt(question, chart_data)
            
            # Get chart configuration from LLM
            chart_config_response = self.client.chat(
                model=self.settings.LLM_MODEL, 
                messages=[{"role": "user", "content": prompt}],
            )
            
            chart_config = chart_config_response['message']['content']
            
            # Parse the chart configuration
            try:
                chart_type, chart_options = self._parse_chart_config(chart_config)
            except Exception as e:
                logger.error(f"Error parsing chart configuration: {str(e)}")
                return "Erreur lors de l'analyse de la configuration du graphique."
            
            # Generate the chart
            chart_image = self._create_chart(chart_data, chart_type, chart_options)
            
            return chart_image
            
        except Exception as e:
            logger.error(f"Chart generation error: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Erreur lors de la génération du graphique: {str(e)}"
    
    def plot_pie_chart(self, title: str, result: list) -> str:
        """Generate a pie chart from data."""
        try:
            categories = [row[0] for row in result]
            quantities = [row[1] for row in result]

            plt.figure(figsize=(8, 8))
            wedges, texts, autotexts = plt.pie(quantities, labels=categories, autopct='%1.1f%%', startangle=140)
            plt.title(title)
            plt.legend(wedges, categories, title="Catégories Défaut", loc="best")

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
        """Generate PDF report from command ID."""
        try:
            # Extract command ID 
            match = re.search(r"commande\s+(\S+)", question, re.IGNORECASE)
            if not match:
                raise ValueError("Format de commande invalide. Utilisez 'commande [ID]'")
            
            command_id = match.group(1).strip()
            logger.info(f"Generating PDF for command ID: {command_id}")
            
            results = []
            query_columns_list = []
            
            # List of queries to execute
            queries = [
                f"SELECT [Référence client], Season, PARAMPAGE as Client,Lavage,Modèle,[Désignation Tissu],[Désignation Façonnier] as Façonnier,[Schedulded Export Date] as [Date Export Programmée],[Ligne Soldée] as Soldée from mes_info_de_base_wic where [Référence client] LIKE '{command_id}%'",
                f"exec PS_TP_Prod @ref='{command_id}'",
                f"exec PS_Entrée_Par_BL @ref='{command_id}'",
                f"select ref, size,inseam, count(*) as Qté from A009_Magasin_brut  where ref  like '{command_id}%' and reprise=0 group by ref, size,inseam order by ref, size,inseam",
                f"exec PS_Consommation_Tissu @ref='{command_id}'",
                f"SELECT [Réference Client],b.Jalon, CASE WHEN CONVERT(INT,SUM([Qte entrée])-SUM([Qte Sortie])) <0 THEN 0 ELSE CONVERT(INT,SUM([Qte entrée])-SUM([Qte Sortie])) END AS Qté_Encours FROM [WIC$En-tête Transfert Atelier] a LEFT JOIN Jalons_Encours b ON b.ID=a.Avancement WHERE [Réference Client] LIKE '{command_id}%' group by [Réference Client],b.Jalon;",
                f"SELECT ref as Ref,size as Taille,inseam as Ej, Défaults,Catégorie_default,[type ligne vente],sum(qté) as Qté_2eme from A900_Deuxieme_choix  where ref like '{command_id}%' group by ref,size,inseam, Défaults,Catégorie_default,[type ligne vente]",
                f"exec PS_Certification @ref='{command_id}'",
                f"exec [PS_Transfert Entrée-Sortie] @ref='{command_id}'",
                f"exec PS_Recette @ref='{command_id}'",
                f"exec PS_Controle_Qualité @ref='{command_id}'",
                f"exec PS_Mesure_Dimensionnelle @ref='{command_id}'",
                f"exec PS_Emballage @ref='{command_id}'",
                f"exec [PS_QD_VS_QEMB] @ref='{command_id}'",
                f"exec PS_Expédition @ref='{command_id}'"
            ]
            
            # Execute all queries
            for query in queries:
                try:
                    with DatabaseCursor() as cursor:
                        cursor.execute(query)
                        columns = [desc[0] for desc in cursor.description]
                        result_data = cursor.fetchall()
                        
                        if not result_data and not results: 
                            raise ValueError(f"Pas de données trouvées pour la commande {command_id}")
                        
                        # Append results and columns
                        results.append(result_data)
                        query_columns_list.append(columns)
                except Exception as e:
                    logger.warning(f"Error executing query {query}: {str(e)}")
                    # Add empty result to maintain order
                    results.append([])
                    query_columns_list.append([])

            # Calculate statistics
            Qte_deuxiéme_choix, Qte_entrante, Qte_embalee = 0, 0, 0
            
            # Calculate second choice quantity
            if len(results) > 6 and results[6]:
                for row in results[6]:
                    Qte_deuxiéme_choix += row[6] if len(row) > 6 else 0
            
            # Calculate incoming quantity
            if len(results) > 2 and results[2]:
                for row in results[2]:
                    Qte_entrante += row[2] if len(row) > 2 else 0
            
            # Calculate packaged quantity
            if len(results) > 12 and results[12]:
                for row in results[12]:
                    Qte_embalee += row[3] if len(row) > 3 else 0
            
            # Generate charts for the report
            deuxieme_choix_per_default_category = f"SELECT Catégorie_default,sum(qté) as Qté_2eme from A900_Deuxieme_choix where ref like '{command_id}%' group by Catégorie_default;"
            deuxieme_choix_per_default_category_result = self.llm.execute_sql(deuxieme_choix_per_default_category)
            pie_chart_categorie_default = self.plot_pie_chart("Répartition du Deuxième Choix par Catégorie Défaut", deuxieme_choix_per_default_category_result)
            
            deuxieme_choix_per_default = f"SELECT Défaults,sum(qté) as Qté_2eme from A900_Deuxieme_choix where ref like '{command_id}%' group by Défaults;"
            deuxieme_choix_per_default_result = self.llm.execute_sql(deuxieme_choix_per_default)
            pie_chart_default = self.plot_pie_chart("Répartition du Deuxième Choix par Défaut", deuxieme_choix_per_default_result)
            
            # Create encours chart if data exists
            bar_chart_encours = ""
            if len(results) > 5 and results[5]:
                bar_chart_encours = self.plot_bar_chart("Répartition des commandes en cours par jalon", 1, 2, "Jalon", "Qté Encours", results[5])        
            
            # Create HTML template for the PDF
            html_content = self._render_template("base_report.html", {
                "command_id": command_id,
                "query_columns_list": query_columns_list,
                "results": results,
                "Qte_deuxiéme_choix": Qte_deuxiéme_choix,
                "Qte_entrante": Qte_entrante,
                "Qte_embalee": Qte_embalee,
                "pie_chart_categorie_default": f"data:image/png;base64,{pie_chart_categorie_default}" if pie_chart_categorie_default else "",
                "pie_chart_default": f"data:image/png;base64,{pie_chart_default}" if pie_chart_default else "",
                "bar_chart_encours": f"data:image/png;base64,{bar_chart_encours}" if bar_chart_encours else "",
                "current_date": datetime.datetime.now().strftime('%d/%m/%Y à %H:%M')
            })
                
            # Generate PDF from HTML
            pdf = await self._html_to_pdf(html_content)
            
            return {
                "pdf": pdf,
                "filename": f"order_{command_id}.pdf",
                "command_id": command_id
            }
        except ValueError as e:
            logger.warning(f"PDF generation validation error: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Erreur lors de la génération du PDF: {str(e)}"}
        
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
            title = options.get('title', f'Graphique pour: {y_column} par {x_column}')
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
            
    async def process_sql_question(self, question: str, columns: list, result: list):
        """Process SQL question and stream response."""
        try:
            prompt = self.get_resphrase_sql_answer_prompt(question, columns, result)
            # Stream response from Ollama
            response = self.client.chat(model=self.settings.LLM_MODEL, messages=[{"role": "user", "content": prompt}], stream=True)
            for chunk in response:
                yield chunk['message']['content']
        except Exception as e:
            logger.error(f"Error processing SQL question: {e}")
            logger.error(traceback.format_exc())
            yield f"Erreur: {str(e)}"


    async def process_general_question(self, history):
        try:
            prompt = self.get_general_question_answer_prompt()
            messages = [{"role": "system", "content": prompt}]
            messages.extend(history)
            # Stream response from Ollama
            response = self.client.chat(model="mistral", messages=messages, stream=True)
            for chunk in response:
                yield chunk['message']['content']
        except Exception as e:
            logger.error(f"Error processing general question: {e}")
            yield e

    def clear_messages(self):
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
       result, columns =chatbot.query_to_sql(question)
       response = chatbot.process_sql_question(question, columns, result)
    else:
        response = chatbot.process_general_question(messages)
    # Set headers based on whether Excel data is included
    headers = {"X-Has-Excel": "true" if chatbot.last_query_excel_data is not None else "false"}
    # Now return the response as streaming, starting with the first chunk.
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
        data = chatbot.last_query_excel_data
        
        if data is None or data.empty:
            return JSONResponse(
                status_code=400,
                content={"error": "Données pas suffisantes pour générer un graphique."}
            )
                
        # Generate chart
        chart_image = await chatbot.generate_chart(question, data)
        
        if chart_image.startswith("Erreur"):
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
            content={"error": "Aucune requête n'a été exécutée."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors de la génération du graphique: {str(e)}"}
        )
   
@app.get("/generate_pdf/")
async def generate_pdf(question: str):
    try:        
        # Call the generate_pdf method
        result = await chatbot.generate_pdf(question)
        
        if "error" in result:
            return Response(status_code=404, content="Une erreur est survenue lors de géneration de PDF, Veuillez vérifier que la commande existe bien dans la base")
        
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
        print(e)
        return Response("Couldn't retrain model on new examples verify your file is in the correct format", status_code=404)

    
@app.get("/clear_messages/")
async def clear_messages():
    chatbot.clear_messages()
    return Response(status_code=200)
