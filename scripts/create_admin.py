from hubert.db.postgres_storage import PostgresStorage
from werkzeug.security import generate_password_hash

def main():
    """
    A command-line script to create a new admin user in the database.
    """
    print("--- Create New Admin User ---")
    
    # Get user input
    username = input("Enter username: ")
    password = input("Enter password: ")
    
    try:
        # Initialize storage
        storage = PostgresStorage()
        
        # Check if user already exists
        if storage.get_user_by_username(username):
            print(f"Error: User '{username}' already exists.")
            return
            
        # Create the user with the 'admin' role
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        storage.create_user(username, password_hash, role='admin')
        
        print(f"Admin user '{username}' created successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()