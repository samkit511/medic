import streamlit as st
import bcrypt
from database import Database
import os

class AuthManager:
    def __init__(self, db: Database):
        self.db = db

    def hash_password(self, password: str) -> bytes:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    def verify_password(self, password: str, hashed: bytes) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), hashed)

    def login_page(self):
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            user = self.db.get_user(username)
            if user and self.verify_password(password, user[2]):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.session_id = f"session_{username}_{os.urandom(8).hex()}"
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    def register_page(self):
        st.header("Register")
        username = st.text_input("Username", key="reg_username")
        password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Register"):
            if password != confirm_password:
                st.error("Passwords do not match")
                return
            if self.db.get_user(username):
                st.error("Username already exists")
                return
            hashed_password = self.hash_password(password)
            self.db.add_user(username, hashed_password)
            st.success("Registered successfully! Please login.")
            st.rerun()

    def forgot_password_page(self):
        st.header("Forgot Password")
        username = st.text_input("Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        if st.button("Reset Password"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
                return
            if not self.db.get_user(username):
                st.error("Username not found")
                return
            hashed_password = self.hash_password(new_password)
            self.db.update_password(username, hashed_password)
            st.success("Password reset successfully! Please login.")
            st.rerun()

    def logout(self):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.session_id = None
        st.rerun()