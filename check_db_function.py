import psycopg2
from hubert.config import settings

def check_function_exists(function_name):
    """Checks if a function exists in the PostgreSQL database."""
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=settings.db_name,
            user=settings.db_username,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port
        )
        cursor = conn.cursor()
        
        # Query to check for the function in pg_proc
        query = """
        SELECT EXISTS (
            SELECT 1 
            FROM pg_proc 
            WHERE proname = %s
        );
        """
        
        cursor.execute(query, (function_name,))
        exists = cursor.fetchone()[0]
        
        if exists:
            print(f"Function '{function_name}' exists in the database.")
        else:
            print(f"Function '{function_name}' DOES NOT exist in the database.")
            
        cursor.close()
    except psycopg2.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_function_exists('bm25_search') 