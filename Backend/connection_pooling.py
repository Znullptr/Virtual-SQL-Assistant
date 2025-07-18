import pyodbc
import queue
import time
import threading
import logging
from dotenv import load_dotenv
import os

# Set up logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConnectionPool:
    """A simple connection pool for SQL Server connections."""
    
    def __init__(self, server, database, username, password, max_connections=10, timeout=30):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.max_connections = max_connections
        self.timeout = timeout
        self.pool = queue.Queue(maxsize=max_connections)
        self.connection_count = 0
        self.lock = threading.RLock()
        self._test_connection()
    
    def _create_connection(self):
        """Create a new database connection."""
        tries = 3
        for attempt in range(tries):
            try:
                connection_string = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={self.server};"
                    f"DATABASE={self.database};"
                    f"UID={self.username};"
                    f"PWD={self.password};"
                    f"Connection Timeout=30;"
                )
                conn = pyodbc.connect(connection_string)
                return conn
            except pyodbc.Error as e:
                logger.warning(f"Connection attempt {attempt+1}/{tries} failed: {str(e)}")
                if attempt < tries - 1:
                    time.sleep(1) 
                else:
                    raise

    def _test_connection(self):
        """Add a new connection to the pool."""
        with self.lock:
            if self.connection_count < self.max_connections:
                try:
                    conn = self._create_connection()
                    self.pool.put(conn)
                    self.connection_count += 1
                    logger.info(f"Establishing connection in pool")
                except Exception as e:
                    logger.error(f"Failed to add connection to pool: {e}")
    
    def get_connection(self):
        """Get a connection from the pool or create a new one if needed."""
        try:
            # Try to get from the pool first
            conn = self.pool.get(block=True, timeout=1)
            
            # Test the connection to make sure it's still valid
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchall()
                cursor.close()
                return conn
            except pyodbc.Error:
                # Connection is dead, create a new one
                logger.warning("Replacing dead connection in pool")
                with self.lock:
                    self.connection_count -= 1
                conn.close()
                return self._create_connection()
                
        except queue.Empty:
            # Pool is empty, create a new connection if possible
            with self.lock:
                if self.connection_count < self.max_connections:
                    conn = self._create_connection()
                    self.connection_count += 1
                    logger.info(f"Pool empty, created new connection. Total: {self.connection_count}")
                    return conn
                else:
                    # Wait for a connection to become available
                    logger.warning("Connection pool exhausted, waiting...")
                    try:
                        return self.pool.get(block=True, timeout=self.timeout)
                    except queue.Empty:
                        raise ConnectionError("Timeout waiting for database connection")
    
    def return_connection(self, conn):
        """Return a connection to the pool."""
        if conn is None:
            return
            
        try:
            # Check if the connection is still usable
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchall()
            cursor.close()
            # Put it back in the pool
            self.pool.put(conn, block=False)
        except (pyodbc.Error, queue.Full) as e:
            # Connection is either dead or pool is full
            logger.warning(f"Not returning connection to pool: {str(e)}")
            try:
                conn.close()
            except:
                pass
            with self.lock:
                self.connection_count -= 1
    
    def close_all(self):
        """Close all connections in the pool."""
        while not self.pool.empty():
            try:
                conn = self.pool.get(block=False)
                try:
                    conn.close()
                except:
                    pass
            except queue.Empty:
                break
        
        with self.lock:
            self.connection_count = 0
        logger.info("All connections in the pool have been closed")


# Global connection pool
pool = None

def initialize_connection_pool():
    global pool
    if pool is None:
        load_dotenv()
        pool = ConnectionPool(
            server=os.getenv("DB_HOST"),
            username=os.getenv("DB_USER"),
            database=os.getenv("DB_NAME"),
            password=os.getenv("DB_PASSWORD"),
            max_connections=int(os.getenv("DB_MAX_CONNECTIONS", '3'))
        )
    return pool

def get_db_connection():
    """Get a connection from the pool."""
    global pool
    if pool is None:
        initialize_connection_pool()
    return pool.get_connection()

class DatabaseCursor:
    """Context manager for database cursors."""
    
    def __init__(self):
        self.conn = None
        self.cursor = None
    
    def __enter__(self):
        self.conn = get_db_connection()
        self.cursor = self.conn.cursor()
        return self.cursor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            if exc_type is not None:
                # If there was an exception, rollback
                try:
                    self.conn.rollback()
                except:
                    pass
            else:
                # Otherwise commit
                try:
                    self.conn.commit()
                except:
                    pass
            # Return connection to pool
            pool.return_connection(self.conn)