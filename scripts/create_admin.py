from hubert.db.postgres_storage import PostgresStorage
from hubert.db.models import User
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
        user = User(username=username, password_hash=password_hash, role='admin')
        user_id = storage.add_user(user)
        
        print(f"Admin user '{username}' created successfully with ID: {user_id}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()