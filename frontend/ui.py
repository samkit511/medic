import streamlit as st
import requests
import json
from datetime import datetime
from media_handler import MediaHandler
from database import Database

API_BASE_URL = "http://localhost:8000"

class ChatbotUI:
    def __init__(self, auth_manager, db: Database):
        self.auth_manager = auth_manager
        self.db = db
        self.media_handler = MediaHandler()

    def check_backend_health(self) -> bool:
        """Check if FastAPI backend is running"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def call_chat_api(self, message: str, session_id: str) -> dict:
        """Call the FastAPI chat endpoint"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"message": message, "session_id": session_id},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            return response.json() if response.status_code == 200 else {
                "success": False,
                "error": f"API Error: {response.status_code}",
                "response": "Sorry, I'm having trouble processing your request."
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "response": "Cannot connect to the medical chatbot service."
            }

    def call_upload_api(self, file, session_id: str) -> dict:
        """Call the FastAPI upload endpoint"""
        try:
            files = {"file": (file.name, file, file.type)}
            data = {"session_id": session_id}
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files,
                data=data,
                timeout=60
            )
            return response.json() if response.status_code == 200 else {
                "success": False,
                "error": f"Upload failed: {response.status_code}"
            }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def display_message(self, message: dict, is_user: bool = False):
        """Display a chat message with proper formatting"""
        urgency_colors = {
            "emergency": "🔴 Emergency",
            "high": "🟠 High",
            "moderate": "🟡 Moderate",
            "low": "🟢 Low"
        }
        with st.container():
            css_class = "chat-message-user" if is_user else "chat-message-assistant"
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            if is_user:
                st.markdown(f'<p class="text-sm text-black">You ({message["timestamp"]}):</p>{message["content"]}', unsafe_allow_html=True)
            else:
                if message.get("emergency", False):
                    st.markdown(
                        '<div class="emergency-alert font-bold text-black">🚨 MEDICAL EMERGENCY DETECTED</div>',
                        unsafe_allow_html=True
                    )
                urgency = message.get("urgency_level", "moderate")
                st.markdown(
                    f'<div class="urgency-{urgency} p-2 mb-2 rounded">'
                    f'<span class="font-semibold text-black">Urgency Level:</span> {urgency_colors.get(urgency, "🟡 Moderate")}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f'<p class="font-semibold text-black">Medical Analysis:</p>{message["response"]}', unsafe_allow_html=True)
                if "possible_conditions" in message and message["possible_conditions"]:
                    st.markdown('<p class="font-semibold text-black mt-2">Possible Conditions:</p>', unsafe_allow_html=True)
                    for condition in message["possible_conditions"][:5]:
                        st.markdown(f'<li class="ml-4 text-black">{condition}</li>', unsafe_allow_html=True)
                if "disclaimer" in message:
                    st.markdown(f'<p class="text-sm text-black mt-2">⚠️ {message["disclaimer"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    def render(self):
        # Header
        st.markdown("""
            <div class="gradient-header text-black p-6 rounded-lg mb-6 text-center shadow-lg">
                <h1 class="text-4xl font-bold text-black">🏥 Clinical Diagnostics Chatbot</h1>
                <p class="text-lg mt-2 text-black">AI-powered medical consultation and document analysis</p>
            </div>
        """, unsafe_allow_html=True)

        # Check backend health
        if not self.check_backend_health():
            st.error("⚠️ **Backend service is not available.** Please ensure the FastAPI server is running on http://localhost:8000")
            st.info("To start the backend, run: `cd backend && python main.py`")
            return

        # Sidebar
        with st.sidebar:
            st.markdown('<div class="sidebar p-4">', unsafe_allow_html=True)
            st.markdown('<h2 class="text-xl font-bold text-black">📋 Session Information</h2>', unsafe_allow_html=True)
            st.markdown(f'<p class="text-black"><strong>Username:</strong> {st.session_state.username}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="text-black"><strong>Session ID:</strong> {st.session_state.session_id}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="text-black"><strong>Messages:</strong> {len(self.db.get_chat_history(st.session_state.username))}</p>', unsafe_allow_html=True)
            if st.button("Logout", key="logout", help="End your session"):
                self.auth_manager.logout()

            st.markdown('<h2 class="text-xl font-bold text-black mt-4">📁 Document Upload</h2>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload medical document or image",
                type=["jpg", "png", "pdf", "xlsx", "txt", "docx"],
                help="Upload medical reports, lab results, or images (300 DPI required for images)",
                key="file_uploader"
            )
            if uploaded_file:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension in ["jpg", "png"]:
                    file_path = self.media_handler.handle_upload(uploaded_file, st.session_state.username)
                    if file_path:
                        self.db.add_upload(st.session_state.username, uploaded_file.name, file_path)
                        st.success("Image uploaded and processed successfully!", icon="✅")
                    else:
                        st.error("Image processing failed. Ensure it meets DPI requirements.", icon="❌")
                elif file_extension in ["pdf", "txt", "docx"]:
                    if st.button("Process Document", key="process_doc"):
                        with st.spinner("Processing document..."):
                            result = self.call_upload_api(uploaded_file, st.session_state.session_id)
                            if result.get("success"):
                                self.db.add_upload(st.session_state.username, uploaded_file.name, f"backend/{result['filename']}")
                                st.success(f"✅ {result['message']}")
                                if "extracted_info" in result:
                                    st.json(result["extracted_info"])
                                doc_message = {
                                    "role": "system",
                                    "content": f"📄 Document '{uploaded_file.name}' processed. Ask questions about it.",
                                    "timestamp": datetime.now().isoformat()
                                }
                                self.db.add_chat(st.session_state.username, doc_message["content"], doc_message["content"])
                            else:
                                st.error(f"❌ {result.get('error', 'Upload failed')}")
                else:
                    st.error("Unsupported file type.", icon="❌")

            st.markdown('<h2 class="text-xl font-bold text-black mt-4">ℹ️ Instructions</h2>', unsafe_allow_html=True)
            st.markdown("""
                <div class="bg-white p-4 rounded-lg shadow">
                    <p class="font-semibold text-black">How to use:</p>
                    <ol class="list-decimal ml-6 text-black">
                        <li>Describe your symptoms in detail</li>
                        <li>Upload medical documents or images</li>
                        <li>Ask follow-up questions</li>
                        <li>Always consult healthcare professionals</li>
                    </ol>
                    <p class="font-semibold text-black mt-2">Emergency:</p>
                    <p class="text-black">If you have severe symptoms, call emergency services immediately!</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Main Content
        st.markdown('<h2 class="text-2xl font-bold text-black mb-4">💬 Medical Consultation</h2>', unsafe_allow_html=True)
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            chat_history = self.db.get_chat_history(st.session_state.username)
            for chat in chat_history:
                message = {
                    "content": chat[2],
                    "role": "user",
                    "timestamp": chat[4]
                }
                self.display_message(message, is_user=True)
                response = {
                    "response": chat[3],
                    "role": "assistant",
                    "urgency_level": "moderate",
                    "timestamp": chat[4]
                }
                self.display_message(response)

        # Chat input
        with st.container():
            st.markdown('<div class="chat-input border-t border-gray-200">', unsafe_allow_html=True)
            if 'chat_input_key' not in st.session_state:
                st.session_state.chat_input_key = 0
            user_input = st.text_input(
                "Describe your symptoms or ask a medical question...",
                key=f"chat_input_{st.session_state.chat_input_key}",
                placeholder="Enter your message here",
                label_visibility="collapsed"
            )
            if user_input:
                with st.container():
                    st.markdown('<div class="chat-message-user">', unsafe_allow_html=True)
                    st.markdown(f'<p class="text-sm text-black">You ({datetime.now().isoformat()}):</p>{user_input}', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with st.spinner("Analyzing your symptoms..."):
                    response = self.call_chat_api(user_input, st.session_state.session_id)
                    if response.get("success"):
                        assistant_message = {
                            "response": response["response"],
                            "urgency_level": response.get("urgency_level", "moderate"),
                            "emergency": response.get("emergency", False),
                            "possible_conditions": response.get("possible_conditions", []),
                            "disclaimer": response.get("disclaimer", ""),
                            "timestamp": datetime.now().isoformat()
                        }
                        self.display_message(assistant_message)
                        self.db.add_chat(st.session_state.username, user_input, assistant_message["response"])
                    else:
                        error_message = {
                            "response": response["response"],
                            "urgency_level": "low",
                            "timestamp": datetime.now().isoformat()
                        }
                        self.display_message(error_message)
                        self.db.add_chat(st.session_state.username, user_input, error_message["response"])
                    # Clear input by incrementing key
                    st.session_state.chat_input_key += 1
            st.markdown('</div>', unsafe_allow_html=True)

        # Footer with Medical Disclaimer
        st.markdown("""
            <div class="footer text-center mt-6">
                <p class="font-bold text-black">⚠️ Important Medical Disclaimer:</p>
                <p class="text-sm text-black">This chatbot provides informational content only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.</p>
                <p class="text-sm font-semibold text-black">In case of emergency, call your local emergency services immediately.</p>
            </div>
        """, unsafe_allow_html=True)