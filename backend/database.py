import logging
import re
import sqlite3
from datetime import datetime
import os
from cryptography.fernet import Fernet
from typing import Optional
import json
import base64
from hipaa_security import HIPAAEncryption, HIPAAAuditLogger, phi_redactor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = 'chatbot.db', encryption_key: Optional[str] = None):
        self.db_path = db_path
        self.encryption_enabled = True
        self.fernet_key = self._get_or_create_encryption_key()
        self.fernet = Fernet(self.fernet_key) if self.fernet_key else None
        self.conn = self._create_connection()
        self.create_tables()

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create database encryption key"""
        key_file = 'data/db_key.key'
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        if os.path.exists(key_file):
            # Load existing key
            with open(key_file, 'rb') as f:
                key = f.read()
            logger.info("Loaded existing database encryption key")
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions (owner read/write only)
            try:
                os.chmod(key_file, 0o600)
            except OSError:
                # Windows doesn't support chmod the same way
                logger.warning("Could not set file permissions on Windows")
            logger.info("Generated new database encryption key")
        
        return key
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create SQLite connection with custom encryption support"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=True)
            logger.info("Database connection established with custom encryption")
            return conn
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    def create_tables(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password BLOB NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    bot_response TEXT,
                    timestamp DATETIME NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            ''')
            self.conn.commit()
            logger.info("Database tables created successfully")

    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self.fernet or not data:
            return data
        try:
            encrypted = self.fernet.encrypt(data.encode('utf-8'))
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self.fernet or not encrypted_data:
            return encrypted_data
        try:
            decoded = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.fernet.decrypt(decoded)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def add_user(self, username: str, hashed_password: bytes):
        if not re.match(r'^[a-zA-Z0-9_]{3,50}$', username):
            logger.error(f"Invalid username format: {username}")
            raise ValueError("Invalid username format. Use 3-50 alphanumeric characters or underscores.")
        with self.conn:
            cursor = self.conn.execute('INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)', 
                          (username, hashed_password))
            self.conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"User added successfully: {username}")
            else:
                logger.warning(f"User already exists: {username}")

    def get_user(self, username: str):
        cursor = self.conn.cursor()
        try:
            cursor.execute('SELECT username, password FROM users WHERE username = ?', (username,))
            user_result = cursor.fetchone()
            logger.debug(f"Retrieved user: {username if user_result else 'None'}")
            return user_result
        finally:
            cursor.close()

    def update_password(self, username: str, hashed_password: bytes):
        cursor = self.conn.cursor()
        try:
            cursor.execute('UPDATE users SET password = ? WHERE username = ?', 
                          (hashed_password, username))
            self.conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Password updated for user: {username}")
            else:
                logger.warning(f"User not found for password update: {username}")
        finally:
            cursor.close()

    def add_chat(self, username: str, user_message: str, bot_response: str):
        cursor = self.conn.cursor()
        try:
            # Encrypt sensitive chat data
            encrypted_user_message = self._encrypt_data(user_message)
            encrypted_bot_response = self._encrypt_data(bot_response)
            
            cursor.execute('INSERT INTO chats (username, user_message, bot_response, timestamp) VALUES (?, ?, ?, ?)',
                          (username, encrypted_user_message, encrypted_bot_response, datetime.now()))
            self.conn.commit()
            logger.info(f"Chat saved for user: {username}")
        finally:
            cursor.close()

    def get_chat_history(self, username: str):
        cursor = self.conn.cursor()
        try:
            cursor.execute('SELECT id, username, user_message, bot_response, timestamp FROM chats WHERE username = ? ORDER BY timestamp', (username,))
            raw_history = cursor.fetchall()
            
            # Decrypt the chat history
            history = []
            for row in raw_history:
                decrypted_row = (
                    row[0],  # id
                    row[1],  # username
                    self._decrypt_data(row[2]),  # user_message
                    self._decrypt_data(row[3]),  # bot_response
                    row[4]   # timestamp
                )
                history.append(decrypted_row)
            
            logger.info(f"Retrieved {len(history)} chat messages for user: {username}")
            return history
        finally:
            cursor.close()

    def add_upload(self, username: str, filename: str, filepath: str):
        cursor = self.conn.cursor()
        try:
            cursor.execute('INSERT INTO uploads (username, filename, filepath, timestamp) VALUES (?, ?, ?, ?)',
                          (username, filename, filepath, datetime.now()))
            self.conn.commit()
            logger.info(f"Upload saved for user: {username}, file: {filename}")
        finally:
            cursor.close()

    def backup_database(self, backup_path: str) -> bool:
        """Create encrypted backup of database"""
        try:
            # Create backup connection
            backup_conn = sqlite3.connect(backup_path, check_same_thread=True)
            
            # Perform backup
            self.conn.backup(backup_conn)
            backup_conn.close()
            
            # Set restrictive permissions
            try:
                os.chmod(backup_path, 0o600)
            except OSError:
                logger.warning("Could not set file permissions on Windows")
            
            logger.info(f"Database backup created: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def verify_encryption(self) -> bool:
        """Verify that database encryption is properly configured"""
        try:
            # Check if Fernet encryption is available
            if not self.fernet:
                logger.warning("Fernet encryption not initialized - data is vulnerable!")
                return False
            
            # Test encryption/decryption
            test_data = "test_sensitive_medical_data"
            encrypted = self._encrypt_data(test_data)
            decrypted = self._decrypt_data(encrypted)
            
            if decrypted == test_data and encrypted != test_data:
                logger.info("Database encryption verified - data is encrypted at application level")
                return True
            else:
                logger.warning("Encryption test failed - data may not be properly encrypted!")
                return False
                
        except Exception as e:
            logger.error(f"Encryption verification failed: {e}")
            return False
    
    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("Database connection closed successfully")
            self.conn = None

    def __del__(self):
        self.close()