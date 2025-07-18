import logging
import json
import os
import pyodbc
import numpy as np
from sentence_transformers import SentenceTransformer
from connection_pooling import  DatabaseCursor
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load env variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

EXAMPLES_PATH: str = os.environ.get("EXAMPLES_PATH", "Data/examples.json")
EMBEDDINGS_PATH: str = os.environ.get("EXAMPLES_PATH", "Data/embeddings.npy")

class CosineSimilarityExampleSelector:
    def __init__(self, examples_file, model, embeddings=None, k=3):
        """
        Initialize the selector, either with pre-computed embeddings or by calculating them.
        """
        self.examples = self._load_examples(examples_file)
        self.k = k
        self.embedding_model = model
        
        if embeddings is not None:
            self.example_embeddings = embeddings
        else:
            self.example_texts = [ex["question"] for ex in self.examples]
            self.example_embeddings = np.array(self.embedding_model.encode(self.example_texts))
        
    def _load_examples(self, examples_file):
        # Load examples from JSON
        with open(examples_file, "r") as f:
            data = json.load(f)
            examples = data["examples"]
        return examples
    
    def select_examples(self, query):
        query_embedding = np.array(self.embedding_model.encode(query)).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.example_embeddings)[0]
        top_k_indices = similarities.argsort()[-self.k:][::-1]
        return [self.examples[i] for i in top_k_indices]
    
    def save(self, embeddings_file):
        """Save embeddings separately."""
            
        # Save embeddings to numpy file
        np.save(embeddings_file, self.example_embeddings)

    @classmethod
    def load(cls, examples_file, model, embeddings_file=None, k=3):
        """Load examples and embeddings from files."""
        # Load embeddings from numpy file
        if embeddings_file:
            embeddings = np.load(embeddings_file)
            instance = cls(examples_file=examples_file, model=model, embeddings=embeddings, k=k)
        else:
            instance = cls(examples_file=examples_file, model=model, k=3)
        
        return instance

class SqlAgent:
    def __init__(self, client, model, dialect="Microsoft SQL"):
        self.client = client
        self.model = model
        self.messages = []
        self.dialect = dialect
        self.ddl_list = []
        self.question_sql_pairs = []
        
    def add_ddl(self, ddl: str) -> None:
        # Store DDL for future reference
        self.ddl_list.append(ddl)
                
    def add_question_sql(self, question: str, sql: str) -> None:
        # Store question-SQL pairs for training
        self.question_sql_pairs.append({"question": question, "query": sql})
        
    def get_related_ddl(self, similar_questions_sql: List[Dict]) -> list[dict]:
        related_ddl_list = []
        added_indexes = set()

        keywords = {
            "brands": 0,
            "categories": 1,
            "customers": 2,
            "order_items": 3,
            "orders": 4,
            "products": 5,
            "staffs": 6,
            "stocks": 7,
            "stores": 8
        }
        sql_commands = [example['query'].lower() for example in similar_questions_sql[:2]]
        for table_name, index in keywords.items():
            for sql in sql_commands:
                if table_name.lower() in sql and index not in added_indexes:
                    related_ddl_list.append(self.ddl_list[index])
                    added_indexes.add(index)

        index_to_name = {v: k for k, v in keywords.items()}
        logger.info(f"Selected tables: {[index_to_name[i] for i in added_indexes]}")
        return related_ddl_list
    
    def get_similar_question_sql(self, question: str) -> list[tuple[str, str]]:
        # Retrieve similar question-SQL pairs
        embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        example_selector = CosineSimilarityExampleSelector.load(EXAMPLES_PATH, embedding_model,EMBEDDINGS_PATH, k=3)
        selected_examples = example_selector.select_examples(question)
        return selected_examples
        
    def calculate_embeddings(self, examples_file):
        embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        example_selector = CosineSimilarityExampleSelector(examples_file,embedding_model)
        example_selector.save(EMBEDDINGS_PATH)

    def submit_prompt(self) -> str:
        # Submit prompt to OpenAI via LangChain and get response
        try:
            response = self.client.invoke(self.messages)
            return response.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return None

    def get_sql_prompt(self, examples: List[Dict], related_ddl: List[str]) -> str:
        # Generate SQL prompt with context
        prompt = f"You are a {self.dialect} expert, your task is to generate a syntactically correct {self.dialect} query."
        prompt += "Here is the relevant DDL for the database:\n"
        prompt += "\n".join(related_ddl)
        prompt += "\nHere are some correct example question-SQL pairs that might help you answering user question:\n"
        for ex in examples:
            prompt += f"Q: {ex['question']}\nSQL: {ex['query']}\n"
        prompt += "\n**Only a SQL query** should always be returned in plain text format, without SQL markdown or any string formats, explanations, or additional details."
        prompt += f"\n Use the {self.dialect} TOP clause to limit the number of rows returned in the result set when necessary. (e.g, SELECT TOP 10 * FROM [table_name])"
        prompt += "\n To avoid errors in the generated SQL query, please make sure to include tables names and columns names between brackets (e.g., [table_name])."
        prompt += "\n There is no typos in the tables and column names. Never modify any column name or table name."
        prompt += "\nbegin!"
        return prompt
    
    def set_system_message(self, question: str):
        similar_questions_sql = self.get_similar_question_sql(question)
        print(similar_questions_sql)
        related_ddl = self.get_related_ddl(similar_questions_sql)
        prompt = self.get_sql_prompt(similar_questions_sql, related_ddl)
        
        # Convert to LangChain message format
        if self.messages:
            self.messages[0] = SystemMessage(content=prompt)
        else:
            self.messages = [SystemMessage(content=prompt)]

    def clear_messages(self):
        self.messages = []

    def generate_sql(self, question: str) -> str:
        # Generate SQL from a natural language question
        try:
            self.set_system_message(question)
            self.messages.append(HumanMessage(content=question))
            sql_query = self.submit_prompt()
            return sql_query
        except Exception as e:
            logger.error(f"Error has occurred while generating SQL Query: {str(e)}")
            return None

    def execute_sql(self, sql: str):
        # Execute SQL command using a connection from the pool
        try:
            with DatabaseCursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()
                self.messages.append(AIMessage(content=sql))
                return [list(row) for row in result]
        except pyodbc.Error as e:
            logger.error(f"Error has occurred while executing SQL Query: {str(e)}")
            self.messages.append(AIMessage(content=str(e)))
            return None