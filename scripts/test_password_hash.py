#!/usr/bin/env python3
"""
Script to test password hashing functionality.
"""

from werkzeug.security import generate_password_hash, check_password_hash

def test_password_hashing():
    print("=== Password Hashing Test ===")
    
    # Test passwords
    test_passwords = ['password123', 'admin', 'test', '1234', 'mypassword']
    
    for password in test_passwords:
        print(f"\nTesting password: '{password}'")
        
        # Generate hash
        try:
            password_hash = generate_password_hash(password, method='pbkdf2:sha256')
            print(f"✓ Hash generated successfully")
            print(f"  Hash length: {len(password_hash)}")
            print(f"  Hash preview: {password_hash[:50]}...")
            
            # Test verification with correct password
            correct_verify = check_password_hash(password_hash, password)
            print(f"✓ Correct password verification: {correct_verify}")
            
            # Test verification with wrong password
            wrong_verify = check_password_hash(password_hash, password + '_wrong')
            print(f"✓ Wrong password verification: {wrong_verify}")
            
            if correct_verify and not wrong_verify:
                print("✅ Password hashing working correctly!")
            else:
                print("❌ Password hashing has issues!")
                
        except Exception as e:
            print(f"❌ Error during password hashing: {e}")

if __name__ == "__main__":
    test_password_hashing() 