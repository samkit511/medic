import streamlit as st
from auth import AuthManager
from database import Database
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize database and auth manager
db = Database()
auth_manager = AuthManager(db)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.session_id = f"session_{st.session_state.get('username', 'guest')}_{os.urandom(8).hex()}"
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Backend API configuration
BACKEND_URL = "http://localhost:8000"

# Set page config
st.set_page_config(page_title="Clinical Diagnostics Chatbot", layout="wide", page_icon="🏥")

# Custom CSS for original soothing color theme with medical gradient header
st.markdown("""
<style>
.stApp {
    background-color: #F4EDE4; /* Warm beige */
    color: #2D3748; /* Dark gray for text */
}
.main-header {
    background: linear-gradient(90deg, #2E86AB, #A23B72);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.main-header h1, .main-header p {
    color: white;
}
.stButton>button {
    background-color: #89B4FA; /* Soft pastel blue */
    color: #2D3748;
    border-radius: 8px;
}
.stTextInput>div>div>input {
    background-color: #D1D9E6; /* Muted blue-gray */
    border-radius: 8px;
    color: #2D3748;
}
.chat-message {
    background-color: #D1D9E6; /* Muted blue-gray */
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
    color: #2D3748;
}
.emergency-alert {
    background: #ff4444;
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    font-weight: bold;
}
.urgency-emergency {
    border-left: 5px solid #ff4444;
    background: #ffe6e6;
    padding: 1rem;
}
.urgency-high {
    border-left: 5px solid #ffa500;
    background: #fff3cd;
    padding: 1rem;
}
.urgency-moderate {
    border-left: 5px solid #ffc107;
    background: #fff3cd;
    padding: 1rem;
}
.urgency-low {
    border-left: 5px solid #28a745;
    background: #e6ffe6;
    padding: 1rem;
}
.footer {
    text-align: center;
    margin-top: 2rem;
    padding: 1rem;
    background: #56585C;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

def submit_query(query):
    """Submit query to backend and return response"""
    try:
        response = requests.post(f"{BACKEND_URL}/api/query", json={
            "message": query,
            "session_id": st.session_state.session_id
        })
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"success": False, "response": f"Error: Failed to connect to backend - {str(e)}", "confidence": 0.0, "explanation": "Backend connection failed"}

def upload_document(file):
    """Upload document to backend and return response"""
    try:
        files = {"file": (file.name, file, file.type)}
        data = {"session_id": st.session_state.session_id}
        response = requests.post(f"{BACKEND_URL}/api/upload", files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"success": False, "error": f"Upload failed: {str(e)}"}

def main():
    # Check login status
    if not st.session_state.logged_in:
        choice = st.sidebar.selectbox("Menu", ["Login", "Register", "Forgot Password"])
        
        if choice == "Login":
            auth_manager.login_page()
        elif choice == "Register":
            auth_manager.register_page()
        else:
            auth_manager.forgot_password_page()
    else:
        # Main interface
        st.markdown("""
        <div class="main-header">
            <h1>Clinical Diagnostics Chatbot</h1>
            <p>Your AI-powered medical assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar for document upload and logout
        with st.sidebar:
            st.header(f"Welcome, {st.session_state.username}")
            st.subheader("Upload Medical Document")
            uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
            if uploaded_file:
                result = upload_document(uploaded_file)
                if result.get("success"):
                    analysis = result.get("analysis", {})
                    response_text = (
                        f"Document processed: {result['filename']}\n\n"
                        f"{analysis.get('response', 'No analysis available.')}\n\n"
                        f"**Confidence**: {analysis.get('confidence', 0.0):.2f}\n"
                        f"**Explanation**: {analysis.get('explanation', 'N/A')}\n"
                        f"**Urgency**: {analysis.get('urgency_level', 'N/A')}"
                    )
                    if analysis.get("possible_conditions"):
                        response_text += f"\n**Possible Conditions**: {', '.join(analysis['possible_conditions'])}"
                    urgency_level = analysis.get('urgency_level', 'low')
                    if analysis.get("emergency"):
                        response_text = f'<div class="emergency-alert">EMERGENCY: SEEK IMMEDIATE MEDICAL ATTENTION!</div>\n{response_text}'
                    response_text = f'<div class="urgency-{urgency_level}">{response_text}</div>'
                    st.session_state.messages.append({
                        "role": "bot",
                        "content": response_text,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                else:
                    st.session_state.messages.append({
                        "role": "bot",
                        "content": f"Error: {result.get('error', 'Failed to process document')}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.messages = []
                st.session_state.session_id = f"session_guest_{os.urandom(8).hex()}"
                st.rerun()
        
        # Chat interface
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-message">You: {msg["content"]}<br><small>{msg["timestamp"]}</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message">Bot: {msg["content"]}<br><small>{msg["timestamp"]}</small></div>', unsafe_allow_html=True)
        
        # Query input
        query = st.text_input("Enter your medical query:", placeholder="Type your symptoms here...")
        if st.button("Send"):
            if query:
                st.session_state.messages.append({
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                result = submit_query(query)
                if result.get("success"):
                    response_text = (
                        f"{result['response']}\n\n"
                        f"**Confidence**: {result.get('confidence', 0.0):.2f}\n"
                        f"**Explanation**: {result.get('explanation', 'N/A')}\n"
                        f"**Urgency**: {result.get('urgency_level', 'N/A')}"
                    )
                    if result.get("disclaimer"):
                        response_text += f"\n**Disclaimer**: {result['disclaimer']}"
                    if result.get("urgency_level") == "emergency":
                        response_text = f'<div class="emergency-alert">EMERGENCY: {result["disclaimer"]}</div>\n{response_text}'
                    response_text = f'<div class="urgency-{result.get("urgency_level", "low")}">{response_text}</div>'
                else:
                    response_text = f"Error: {result.get('response', 'Failed to process query')}"
                st.session_state.messages.append({
                    "role": "bot",
                    "content": response_text,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()
        
        # Footer
        st.markdown("""
        <div class="footer">
            Powered by Clinical Diagnostics AI
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()