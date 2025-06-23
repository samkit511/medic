# 🏥 Clinical Diagnostics Chatbot

An AI-powered medical consultation chatbot using RAG (Retrieval Augmented Generation), few-Shot Prompting, and Free APIs (Groq + HuggingFace). Built with FastAPI backend and Streamlit frontend.

1)Features

Medical Symptom Analysis : using AI-powered RAG 
Emergency Detection :with immediate alerts for critical symptoms  
Document Upload & Processing :(PDF, TXT, DOCX)
Conversation Memory : with session management
Multi-API Support : (Groq + HuggingFace with fallbacks)
Real-time Chat Interface: with medical safety protocols
Free to Use:no premium API requirements

2) 📁 Project Structure( more files to be added later)
clinical_diagnostics_chatbot/
├── 📁 backend/
│ ├── main.py # FastAPI application(High-performance API framework,Fast AI inference (free tier))
│ ├── services.py # Core medical AI services
│ ├── config.py # Configuration management
│ └── requirements.txt # Backend dependencies
├── 📁 frontend/
│ ├── app.py # Streamlit application
│ └── requirements.txt # Frontend dependencies
├── 📁 data/
│ ├── 📁 documents/ # Medical knowledge base
│ └── 📁 uploads/ # User uploaded documents
├── .env # Environment variables
├── .gitignore # Git ignore rules
└── README.md # This file

3. Set Up Virtual Environment
   go to the terminal of vs code ->python -m venv venv->venv\Scripts\activate
4.Install Dependencies
cd backend
pip install fastapi uvicorn python-multipart python-dotenv groq huggingface-hub PyPDF2 requests

5.cd frontend
pip install streamlit requests python-multipart python-dotenv
 6.API Keys (Get free keys from respective platforms)
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
GROQ_API_KEY=your_groq_api_key_here
   
