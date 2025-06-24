#!/usr/bin/env python3
"""
Script to test the full user registration and login flow.
"""

import sys
import os
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from hubert.db.postgres_storage import PostgresStorage
from hubert.db.models import User
from werkzeug.security import generate_password_hash, check_password_hash

def test_user_flow():
    print("=== User Registration and Login Flow Test ===")
    
    test_username = "testuser123"
    test_password = "testpass123"
    
    try:
        # Initialize storage
        print("\n1. Testing database connection...")
        storage = PostgresStorage()
        print("✓ Database connection successful")
        
        # Clean up - remove test user if exists
        print("\n2. Cleaning up existing test user...")
        existing_user = storage.get_user_by_username(test_username)
        if existing_user:
            print(f"Found existing test user: {existing_user.username}")
            print(f"Using existing user: {existing_user.username}")
            user_id = existing_user.id
            saved_user = existing_user
        else:
            print("No existing test user found")
            
            # Test password hashing
            print(f"\n3. Testing password hashing for password: '{test_password}'")
            password_hash = generate_password_hash(test_password, method='pbkdf2:sha256')
            print(f"✓ Password hash generated: {len(password_hash)} characters")
            print(f"  Hash preview: {password_hash[:50]}...")
            
            # Test hash verification
            verify_result = check_password_hash(password_hash, test_password)
            print(f"✓ Hash verification test: {verify_result}")
            
            if not verify_result:
                print("❌ Password hashing is not working correctly!")
                return
            
            # Create user object
            print(f"\n4. Creating user object...")
            new_user = User(
                username=test_username,
                password_hash=password_hash,
                role='user'
            )
            print(f"✓ User object created: {new_user.username}, role: {new_user.role}")
            
            # Save to database
            print(f"\n5. Saving user to database...")
            user_id = storage.add_user(new_user)
            print(f"✓ User saved with ID: {user_id}")
            
            # Add a small delay and force a new connection
            time.sleep(0.1)
            print(f"Waiting 0.1 seconds and creating new storage connection...")
            
            # Create a NEW storage instance to test if it's a connection issue
            storage2 = PostgresStorage()
            print(f"✓ New storage connection created")
            
            # Verify user was saved/exists
            print(f"\n6. Verifying user in database...")
            saved_user = storage2.get_user_by_username(test_username)
            if saved_user:
                print(f"✓ User found: {saved_user.username}, role: {saved_user.role}, ID: {saved_user.id}")
                print(f"  Stored hash length: {len(saved_user.password_hash)}")
                print(f"  Stored hash preview: {saved_user.password_hash[:50]}...")
            else:
                print("❌ User not found in database!")
                
                # Additional debugging - try to query the database directly
                print("\nDEBUG: Checking if user exists with direct database query...")
                try:
                    conn = storage2.connect()
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT id, username, role FROM users WHERE username = %s", (test_username,))
                        direct_result = cursor.fetchone()
                        if direct_result:
                            print(f"✓ Direct query found user: ID={direct_result[0]}, username={direct_result[1]}, role={direct_result[2]}")
                        else:
                            print("❌ Direct query also shows no user!")
                            
                        # List all users
                        cursor.execute("SELECT id, username, role FROM users")
                        all_users = cursor.fetchall()
                        print(f"All users in database: {len(all_users)}")
                        for user_row in all_users:
                            print(f"  - ID: {user_row[0]}, Username: {user_row[1]}, Role: {user_row[2]}")
                            
                    storage2.pool.putconn(conn)
                    
                except Exception as debug_error:
                    print(f"❌ Debug query failed: {debug_error}")
                
                return
        
        # Test login with correct password
        print(f"\n7. Testing login with correct password...")
        login_result = check_password_hash(saved_user.password_hash, test_password)
        print(f"✓ Login test result: {login_result}")
        
        # Test login with wrong password
        print(f"\n8. Testing login with wrong password...")
        wrong_login_result = check_password_hash(saved_user.password_hash, test_password + "_wrong")
        print(f"✓ Wrong password test result: {wrong_login_result}")
        
        if login_result and not wrong_login_result:
            print(f"\n✅ All tests passed! User flow is working correctly for user: {test_username}")
            print(f"   You can now try logging in with username: '{test_username}' and password: '{test_password}'")
        else:
            print(f"\n❌ Tests failed! There's an issue with the user flow.")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_user_flow() 