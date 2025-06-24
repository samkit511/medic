
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Any
import os

st.set_page_config(
    page_title="Clinical Diagnostics Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000" 

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .emergency-alert {
        background: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .urgency-high {
        border-left: 5px solid #ff4444;
        background: #ffe6e6;
        padding: 1rem;
    }
    .urgency-moderate {
        border-left: 5px solid #ffa500;
        background: #fff3cd;
        padding: 1rem;
    }
    .urgency-low {
        border-left: 5px solid #28a745;
        background: #e6ffe6;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def call_chat_api(message: str, session_id: str) -> Dict[str, Any]:
    """Call the FastAPI chat endpoint (based on search results pattern)"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": message,
                "session_id": session_id
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"API Error: {response.status_code}",
                "response": "Sorry, I'm having trouble processing your request. Please try again."
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout",
            "response": "The request is taking too long. Please try again with a shorter message."
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Connection error",
            "response": "Cannot connect to the medical chatbot service. Please ensure the backend is running."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response": "An unexpected error occurred. Please try again."
        }

def call_upload_api(file, session_id: str) -> Dict[str, Any]:
    """Call the FastAPI upload endpoint"""
    try:
        files = {"file": file}
        data = {"session_id": session_id}
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"Upload failed: {response.status_code}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def check_backend_health() -> bool:
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def display_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper formatting"""
    if is_user:
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            # Check for emergency
            if message.get("emergency", False):
                st.markdown(
                    '<div class="emergency-alert">🚨 MEDICAL EMERGENCY DETECTED</div>',
                    unsafe_allow_html=True
                )
            
            # Display urgency level
            urgency = message.get("urgency_level", "moderate")
            urgency_colors = {
                "emergency": "🔴",
                "high": "🟠", 
                "moderate": "🟡",
                "low": "🟢"
            }
            
            st.markdown(f"**Urgency Level:** {urgency_colors.get(urgency, '🟡')} {urgency.title()}")
            
            # Display main response
            st.markdown("**Medical Analysis:**")
            st.markdown(message["content"])
            
            # Display possible conditions if available
            if "possible_conditions" in message and message["possible_conditions"]:
                st.markdown("**Possible Conditions:**")
                for condition in message["possible_conditions"][:5]:  # Limit to 5
                    st.markdown(f"• {condition}")
            
            
            if "disclaimer" in message:
                st.caption(f"⚠️ {message['disclaimer']}")

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🏥 Clinical Diagnostics Chatbot</h1>
        <p style="color: white; margin: 0;">AI-powered medical consultation and document analysis</p>
    </div>
    """, unsafe_allow_html=True)
 
    if not check_backend_health():
        st.error("⚠️ **Backend service is not available.** Please ensure the FastAPI server is running on http://localhost:8000")
        st.info("To start the backend, run: `cd backend && python main.py`")
        return
    
   
    with st.sidebar:
        st.header("📋 Session Information")
        st.info(f"**Session ID:** {st.session_state.session_id}")
        st.info(f"**Messages:** {len(st.session_state.messages)}")
        
        st.header("📁 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload medical document",
            type=["pdf", "txt", "docx"],
            help="Upload medical reports, lab results, or other relevant documents"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    result = call_upload_api(uploaded_file, st.session_state.session_id)
                    
                    if result.get("success", False):
                        st.success(f"✅ {result['message']}")
                        
                        # Display extracted information
                        if "extracted_info" in result:
                            st.json(result["extracted_info"])
                            
                        # Add document processing message to chat
                        doc_message = {
                            "role": "system",
                            "content": f"📄 Document '{uploaded_file.name}' has been processed and analyzed. You can now ask questions about it.",
                            "timestamp": datetime.now()
                        }
                        st.session_state.messages.append(doc_message)
                        st.rerun()
                    else:
                        st.error(f"❌ {result.get('error', 'Upload failed')}")
        
        st.header("ℹ️ Instructions")
        st.markdown("""
        **How to use:**
        1. Describe your symptoms in detail
        2. Upload medical documents if needed
        3. Ask follow-up questions
        4. Always consult healthcare professionals
        
        **Emergency:** If you have severe symptoms, call emergency services immediately!
        """)
    
    # Chat Interface
    st.header("💬 Medical Consultation")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            display_message(message, is_user=True)
        else:
            display_message(message)
    
    # Chat input
    if prompt := st.chat_input("Describe your symptoms or ask a medical question..."):
        # Add user message to history
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        }
        st.session_state.messages.append(user_message)
        
        # Display user message
        display_message(user_message, is_user=True)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your symptoms..."):
                # Call API
                response = call_chat_api(prompt, st.session_state.session_id)
                
                if response.get("success", False):
                    # Create assistant message
                    assistant_message = {
                        "role": "assistant",
                        "content": response["response"],
                        "urgency_level": response.get("urgency_level", "moderate"),
                        "emergency": response.get("emergency", False),
                        "possible_conditions": response.get("possible_conditions", []),
                        "disclaimer": response.get("disclaimer", ""),
                        "timestamp": datetime.now()
                    }
                    
                    # Display response
                    display_message(assistant_message)
                    
                    # Add to history
                    st.session_state.messages.append(assistant_message)
                    
                else:
                    # Display error
                    error_message = {
                        "role": "assistant",
                        "content": response["response"],
                        "urgency_level": "low",
                        "timestamp": datetime.now()
                    }
                    display_message(error_message)
                    st.session_state.messages.append(error_message)
        
        # Rerun to update the interface
        st.rerun()

# Emergency Quick Check Section
def emergency_section():
    st.header("🚨 Quick Emergency Check")
    
    emergency_symptoms = st.text_input("Quick symptom check for emergency:")
    
    if emergency_symptoms and st.button("Check Emergency Status"):
        try:
            response = requests.get(
                f"{API_BASE_URL}/emergency-check",
                params={"symptoms": emergency_symptoms}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result["is_emergency"]:
                    st.error(f"🚨 **{result['message']}**")
                else:
                    st.success(f"✅ {result['message']}")
            else:
                st.error("Failed to check emergency status")
                
        except Exception as e:
            st.error(f"Error: {e}")

# Run the app
if __name__ == "__main__":
    main()
    

    st.markdown("---")
    emergency_section()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #56585C; border-radius: 10px;">
        <p><strong>⚠️ Important Medical Disclaimer:</strong></p>
        <p>This chatbot provides informational content only and is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.</p>
        <p><strong>In case of emergency, call your local emergency services immediately.</strong></p>
    </div>
    """, unsafe_allow_html=True)
