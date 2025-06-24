#!/usr/bin/env python3
"""
Script to check users in the database and test database connection.
"""

from hubert.db.postgres_storage import PostgresStorage
from hubert.config import settings

def main():
    print("--- Database User Check ---")
    
    try:
        # Initialize storage
        print("Connecting to database...")
        storage = PostgresStorage()
        print("✓ Database connection successful")
        
        # Test query to check if users table exists and has data
        print("\nChecking users table...")
        
        # Get a connection and check users table
        conn = storage.connect()
        try:
            with conn.cursor() as cursor:
                # Check if users table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'users'
                    );
                """)
                table_exists = cursor.fetchone()[0]
                
                if not table_exists:
                    print("❌ Users table does not exist!")
                    return
                
                print("✓ Users table exists")
                
                # Count users
                cursor.execute("SELECT COUNT(*) FROM users;")
                user_count = cursor.fetchone()[0]
                print(f"✓ Found {user_count} users in database")
                
                # List all users
                if user_count > 0:
                    cursor.execute("SELECT id, username, role FROM users;")
                    users = cursor.fetchall()
                    print("\nUsers in database:")
                    for user in users:
                        print(f"  - ID: {user[0]}, Username: {user[1]}, Role: {user[2]}")
                else:
                    print("❌ No users found in database")
                    
        finally:
            storage.pool.putconn(conn)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Database settings: host={settings.db_host}, port={settings.db_port}, dbname={settings.db_name}, user={settings.db_username}")

if __name__ == "__main__":
    main() 