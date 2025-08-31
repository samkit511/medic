Clinical Diagnostics Chatbot
Overview
The Clinical Diagnostics Chatbot, named CURA, is an AI-powered medical assistant designed to provide preliminary symptom analysis, document processing, and general health guidance. Built with a FastAPI backend and a React frontend, it leverages advanced language models (HuggingFace and Groq) for natural language processing, medical knowledge retrieval, and emotional intelligence. The application supports multilingual queries, document uploads (PDF, TXT, JPG, PNG), and session-based conversation tracking, with a strong emphasis on privacy and user trust.
This project aims to bridge healthcare gaps, particularly for underserved communities, by offering accessible, 24/7 medical assistance. It includes features like symptom analysis, document processing, and empathetic responses, all while adhering to medical disclaimers and urging users to consult healthcare professionals.
Features

Symptom Analysis: Processes user-reported symptoms to identify possible conditions, urgency levels, and recommendations using Retrieval-Augmented Generation (RAG) and few-shot prompting.
Document Upload: Supports uploading medical documents (PDF, TXT) and images (JPG, PNG) for analysis, extracting medical information such as symptoms, conditions, and medications.
Multilingual Support: Translates non-English queries to English using Google Translate for broader accessibility.
Emotional Intelligence: Uses sentiment analysis to adjust response tone (empathetic or professional) based on user input.
Session Management: Tracks conversation history and context per session for personalized responses.
Privacy-First Design: Stores user data securely in a SQLite database and ensures compliance with regulations like HIPAA and GDPR.
Google OAuth Login: Allows users to authenticate via Google for a personalized experience.
Responsive UI: Built with React, Tailwind CSS, and Bootstrap for a user-friendly interface across devices.
Error Handling: Provides standardized error responses and logging for debugging and reliability.

Prerequisites
Before setting up the application, ensure you have the following installed:

Python 3.8+ (for backend)
Node.js 18+ and npm 8+ (for frontend)
Git (for cloning the repository)
SQLite (for database, included with Python)
API keys for:
HuggingFace: For language and vision models
Groq: For fallback AI responses
Google OAuth: For user authentication


Optional: Tesseract OCR (for image text extraction)

Setup Instructions
Follow these steps to set up and run the application locally.
1. Clone the Repository
git clone <repository-url>
cd medic-landing

2. Set Up the Backend

Navigate to the backend directory:
cd backend


Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install backend dependencies:
pip install -r requirements.txt

Configure environment variables:Create a .env file in the backend/ directory with the following:
HUGGINGFACE_API_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key

Replace your_huggingface_token and your_groq_api_key with your actual API keys.

Start the FastAPI server:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

The backend will be available at http://localhost:8000. API documentation is at http://localhost:8000/docs.


3. Set Up the Frontend

Navigate to the frontend directory:
cd medic-landing


Install frontend dependencies:
npm install


Configure Google OAuth:Update src/components/auth.jsx with your Google OAuth client ID:
const clientId = 'your_google_oauth_client_id';


Start the React development server:
npm run dev

The frontend will be available at http://localhost:5173 (or another port if specified).


4. Install Tesseract OCR (Optional, for Image Processing)
If you want to process images (JPG, PNG), install Tesseract OCR:

Windows:
Download and install from Tesseract at UB Mannheim.
Add Tesseract to your system PATH.


macOS:brew install tesseract


Linux:sudo apt-get install tesseract-ocr



5. Test the Application

Run backend tests:
cd backend
python test_services.py


Access the application:

Open http://localhost:5173 in your browser.
Navigate to /login to sign in with Google OAuth.
After logging in, access the chatbot at /chatbot.



How the Application Works
Backend (FastAPI)
The backend is built with FastAPI and provides a robust API for handling medical queries, document uploads, and session management. Key components include:

Configuration (config.py):

Manages settings like API keys, file paths, and model configurations using Pydantic.
Defines emergency keywords and medical disclaimers.


Database (database.py):

Uses SQLite to store user data, chat history, and uploaded files.
Tables: users (username, password), chats (conversation history), uploads (file metadata).


Models (modells.py):

Defines Pydantic models for request/response validation (e.g., ChatRequest, ChatResponse, DocumentUploadResponse).
Includes enums for urgency levels and file types.


Services (services.py):

MedicalRAGService: Implements Retrieval-Augmented Generation to fetch relevant medical knowledge and possible conditions.
HuggingFaceService/GroqService: Interfaces with HuggingFace and Groq APIs for AI responses, with fallback to Groq if HuggingFace fails.
DocumentProcessingService: Extracts text from PDFs and images (using PyPDF2 and Tesseract OCR) and performs medical information extraction.
ContextService: Manages session-based conversation context.


API Endpoints (main.py):

/api/query: Processes medical queries with AI and RAG.
/api/upload: Handles document uploads (PDF, TXT).
/api/upload_image: Processes medical images (JPG, PNG).
/api/save_chat: Saves chat history to the database.
/api/get_chat_history: Retrieves chat history for a user.
/session/{session_id}: Manages session creation and retrieval.
/health: Checks API health status.
/emergency-check: Evaluates if symptoms indicate an emergency.


Testing (test_services.py):

Tests emergency detection, non-emergency queries, multilingual support, emotional responses, document processing, and cache performance.



Frontend (React)
The frontend is a React application using TypeScript, Tailwind CSS, and Bootstrap for styling. Key components include:

Landing Page (landing.jsx):
Displays a welcome message and a call-to-action to try CURA.


Authentication (auth.jsx):
Uses Google OAuth for user login, storing the username in local storage.


Chatbot (chatbot.jsx):
Provides a chat interface for submitting medical queries and uploading documents.
Displays responses with urgency levels, confidence scores, and disclaimers.
Supports file uploads via a hidden input triggered by a "+" button.


Feedback (feedback.jsx):
Embeds a Google Form for user feedback.


Routing (App.tsx):
Defines routes for landing, about, recognition, login, feedback, and chatbot pages.



Workflow

User Interaction:

Users log in via Google OAuth or use the app as a guest.
They submit medical queries or upload documents through the chatbot interface.
The frontend sends requests to the backend via Axios.


Query Processing:

The backend receives the query, extracts symptoms using MedicalRAGService, and generates a response using HuggingFaceService or GroqService.
Emergency keywords trigger urgent responses with appropriate warnings.
Responses include possible conditions, recommendations, urgency levels, and disclaimers.


Document Processing:

Uploaded files are processed by DocumentProcessingService.
PDFs and text files are parsed for text; images undergo OCR.
Medical information (symptoms, conditions, medications) is extracted using regex and NER (Named Entity Recognition).


Session Management:

ContextService tracks conversation history and uploaded documents per session.
Sessions are identified by unique IDs, ensuring context-aware responses.


Response Rendering:

The frontend displays AI responses with styled urgency indicators (e.g., red for emergencies).
Timestamps and disclaimers are included for transparency.



Usage Example

Log in:
Go to http://localhost:5173/login and sign in with Google.


Submit a query:
Navigate to /chatbot, enter "I have chest pain and shortness of breath", and click "Send".
The response will flag an emergency, list possible conditions (e.g., heart attack, angina), and recommend immediate medical attention.


Upload a document:
Click the "+" button, select a PDF or image, and upload.
The chatbot will analyze the document and return extracted medical information.



Notes

Medical Disclaimer: The chatbot emphasizes that it is not a substitute for professional medical advice. Emergency cases prompt immediate action recommendations.
API Keys: Ensure valid HuggingFace and Groq API keys are set in the .env file to avoid service failures.
Limitations: The chatbot relies on predefined medical knowledge and examples (medical_examples.json). Expand these for broader coverage.
Security: User data is stored in SQLite, and file uploads are sanitized to prevent malicious inputs.
