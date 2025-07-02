import sqlite3
from datetime import datetime

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('chatbot.db', check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password BLOB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                user_message TEXT,
                bot_response TEXT,
                timestamp DATETIME
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                filename TEXT,
                filepath TEXT,
                timestamp DATETIME
            )
        ''')
        self.conn.commit()

    def add_user(self, username: str, hashed_password: bytes):
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                      (username, hashed_password))
        self.conn.commit()

    def get_user(self, username: str):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        return cursor.fetchone()

    def update_password(self, username: str, hashed_password: bytes):
        cursor = self.conn.cursor()
        cursor.execute('UPDATE users SET password = ? WHERE username = ?', 
                      (hashed_password, username))
        self.conn.commit()

    def add_chat(self, username: str, user_message: str, bot_response: str):
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO chats (username, user_message, bot_response, timestamp) VALUES (?, ?, ?, ?)',
                      (username, user_message, bot_response, datetime.now()))
        self.conn.commit()

    def get_chat_history(self, username: str):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM chats WHERE username = ? ORDER BY timestamp', (username,))
        return cursor.fetchall()

    def add_upload(self, username: str, filename: str, filepath: str):
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO uploads (username, filename, filepath, timestamp) VALUES (?, ?, ?, ?)',
                      (username, filename, filepath, datetime.now()))
        self.conn.commit()

    def __del__(self):
        self.conn.close()