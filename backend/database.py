import logging
import re
import sqlite3
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('chatbot.db', check_same_thread=True)
        self.create_tables()

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
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT username, password FROM users WHERE username = ?', (username,))
            user_result = cursor.fetchone()
            logger.debug(f"Retrieved user: {username if user_result else 'None'}")
            return user_result

    def update_password(self, username: str, hashed_password: bytes):
        with self.conn.cursor() as cursor:
            cursor.execute('UPDATE users SET password = ? WHERE username = ?', 
                          (hashed_password, username))
            self.conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Password updated for user: {username}")
            else:
                logger.warning(f"User not found for password update: {username}")

    def add_chat(self, username: str, user_message: str, bot_response: str):
        with self.conn.cursor() as cursor:
            cursor.execute('INSERT INTO chats (username, user_message, bot_response, timestamp) VALUES (?, ?, ?, ?)',
                          (username, user_message, bot_response, datetime.now()))
            self.conn.commit()
            logger.info(f"Chat saved for user: {username}")

    def get_chat_history(self, username: str):
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT id, username, user_message, bot_response, timestamp FROM chats WHERE username = ? ORDER BY timestamp', (username,))
            history = cursor.fetchall()
            logger.info(f"Retrieved {len(history)} chat messages for user: {username}")
            return history

    def add_upload(self, username: str, filename: str, filepath: str):
        with self.conn.cursor() as cursor:
            cursor.execute('INSERT INTO uploads (username, filename, filepath, timestamp) VALUES (?, ?, ?, ?)',
                          (username, filename, filepath, datetime.now()))
            self.conn.commit()
            logger.info(f"Upload saved for user: {username}, file: {filename}")

    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("Database connection closed successfully")
            self.conn = None

    def __del__(self):
        self.close()