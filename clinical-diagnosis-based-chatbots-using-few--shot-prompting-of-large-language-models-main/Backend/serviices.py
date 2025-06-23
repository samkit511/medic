
import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
from groq import Groq
from huggingface_hub import InferenceClient
import PyPDF2
from confiig import settings, is_emergency_symptom 

class MedicalRAGService:
    """
    RAG (Retrieval Augmented Generation) service for medical knowledge
    This service retrieves relevant medical information and generates responses
    """
    
    def __init__(self):
        self.medical_knowledge = self._load_medical_knowledge()
        self.symptom_database = self._load_symptom_database()
        
    def _load_medical_knowledge(self) -> Dict[str, str]:
        """Load basic medical knowledge base"""
        return {
            "chest_pain": "Chest pain can indicate heart problems, lung issues, or muscle strain. Seek immediate medical attention if severe.",
            "headache": "Headaches can be caused by stress, dehydration, tension, or more serious conditions. Monitor frequency and severity.",
            "fever": "Fever indicates infection or inflammation. Rest, hydration, and monitoring temperature are important.",
            "cough": "Persistent cough may indicate respiratory infection, allergies, or other lung conditions.",
            "shortness_of_breath": "Difficulty breathing can indicate serious heart or lung problems. Seek medical attention immediately.",
            "nausea": "Nausea can be caused by digestive issues, infections, medications, or other conditions.",
            "fatigue": "Persistent fatigue may indicate underlying health conditions, stress, or lifestyle factors."
        }
    
    def _load_symptom_database(self) -> Dict[str, List[str]]:
        """Load symptom-to-condition mapping"""
        return {
            "chest_pain": ["Heart attack", "Angina", "Pulmonary embolism", "Anxiety", "Muscle strain"],
            "shortness_of_breath": ["Asthma", "Heart failure", "Pneumonia", "COPD", "Anxiety"],
            "headache": ["Tension headache", "Migraine", "Cluster headache", "Sinusitis", "High blood pressure"],
            "fever": ["Viral infection", "Bacterial infection", "COVID-19", "Flu", "Food poisoning"],
            "cough": ["Common cold", "Bronchitis", "Pneumonia", "Allergies", "GERD"],
            "nausea": ["Food poisoning", "Gastritis", "Pregnancy", "Medication side effect", "Anxiety"],
            "fatigue": ["Anemia", "Depression", "Sleep disorders", "Thyroid problems", "Chronic fatigue syndrome"]
        }
    
    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """Get relevant medical context for a query (main RAG function)"""
      
        extracted_symptoms = self._extract_symptoms(query)
        
     
        relevant_knowledge = []
        possible_conditions = []
        
        for symptom in extracted_symptoms:
            if symptom in self.medical_knowledge:
                relevant_knowledge.append(self.medical_knowledge[symptom])
            
            if symptom in self.symptom_database:
                possible_conditions.extend(self.symptom_database[symptom])
        
        return {
            "extracted_symptoms": extracted_symptoms,
            "relevant_knowledge": relevant_knowledge,
            "possible_conditions": list(set(possible_conditions)),
            "emergency_detected": any(is_emergency_symptom(symptom) for symptom in extracted_symptoms)
        }
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from user text"""
        text_lower = text.lower()
        extracted = []
        
        for symptom in self.medical_knowledge.keys():
           
            symptom_words = symptom.replace("_", " ").split()
            if all(word in text_lower for word in symptom_words):
                extracted.append(symptom)
        
        return extracted

class HuggingFaceService:
    """
    HuggingFace API service for medical text generation with few-shot prompting
    """
    
    def __init__(self):
        self.client = InferenceClient(token=settings.huggingface_api_token)
        self.medical_model = settings.hf_medical_model
        self.few_shot_examples = self._load_medical_examples()  # Added few-shot examples
        
    def _load_medical_examples(self) -> List[Dict[str, str]]:
        """Load few-shot medical examples"""
        return [
            {
                "input": "I have chest pain and shortness of breath",
                "output": "Analysis: These symptoms are concerning for potential cardiac issues.\nRecommendations: Seek immediate medical attention. Call emergency services.\nUrgency: Emergency"
            },
            {
                "input": "I have a headache and feel nauseous",
                "output": "Analysis: This combination suggests possible migraine or tension headache.\nRecommendations: Rest in a dark room, stay hydrated. Monitor symptoms.\nUrgency: Moderate"
            },
            {
                "input": "I have been feeling very tired and weak lately",
                "output": "Analysis: Persistent fatigue can indicate various conditions including anemia, depression, or thyroid issues.\nRecommendations: Consider blood tests and consult your primary care physician.\nUrgency: Low"
            },
            {
                "input": "I have a persistent cough and fever",
                "output": "Analysis: These symptoms suggest a respiratory infection, possibly viral or bacterial.\nRecommendations: Rest, fluids, monitor temperature. See doctor if symptoms worsen.\nUrgency: Moderate"
            }
        ]
        
    async def generate_medical_response(self, prompt: str, context: Dict = None) -> Dict[str, Any]:
        """Generate medical response using HuggingFace with few-shot prompting"""
        try:
            # Create few-shot enhanced prompt
            medical_prompt = self._create_few_shot_prompt(prompt, context)
            
            response = self.client.text_generation(
                medical_prompt,
                model=self.medical_model,
                max_new_tokens=settings.max_response_tokens,
                temperature=settings.medical_temperature,
                do_sample=True
            )
            
            return {
                "success": True,
                "response": response,
                "model_used": self.medical_model,
                "provider": "huggingface",
                "method": "few-shot"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "provider": "huggingface"
            }
    
    def _create_few_shot_prompt(self, user_query: str, context: Dict = None) -> str:
        """Create few-shot medical prompt"""
        # Build few-shot examples section
        examples_text = "Here are examples of medical symptom analysis:\n\n"
        
        for i, example in enumerate(self.few_shot_examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Patient: {example['input']}\n"
            examples_text += f"Medical Analysis: {example['output']}\n\n"
        
        context_text = ""
        if context and context.get("relevant_knowledge"):
            context_text = f"Relevant Medical Knowledge: {', '.join(context['relevant_knowledge'])}\n\n"
        
        full_prompt = f"""You are a medical AI assistant. Analyze patient symptoms using the examples below as guidance.

{examples_text}

{context_text}Now analyze this case following the same format:

Patient: {user_query}
Medical Analysis: 

Please provide:
- Analysis: Clinical assessment of the symptoms
- Recommendations: Suggested next steps
- Urgency: Level of medical urgency (Emergency/High/Moderate/Low)

Remember: Always recommend consulting healthcare professionals for proper diagnosis."""

        return full_prompt

class GroqService:
    """
    Groq API service for fast medical reasoning with few-shot prompting
    """
    
    def __init__(self):
        if settings.groq_api_key:
            self.client = Groq(api_key=settings.groq_api_key)
            self.few_shot_examples = self._load_medical_examples()  # Added few-shot examples
        else:
            self.client = None
    
    def _load_medical_examples(self) -> List[Dict[str, str]]:
        """Load few-shot medical examples (same as HuggingFace)"""
        return [
            {
                "input": "I have chest pain and shortness of breath",
                "output": "Analysis: These symptoms are concerning for potential cardiac issues.\nRecommendations: Seek immediate medical attention. Call emergency services.\nUrgency: Emergency"
            },
            {
                "input": "I have a headache and feel nauseous",
                "output": "Analysis: This combination suggests possible migraine or tension headache.\nRecommendations: Rest in a dark room, stay hydrated. Monitor symptoms.\nUrgency: Moderate"
            },
            {
                "input": "I have been feeling very tired and weak lately",
                "output": "Analysis: Persistent fatigue can indicate various conditions including anemia, depression, or thyroid issues.\nRecommendations: Consider blood tests and consult your primary care physician.\nUrgency: Low"
            }
        ]
    
    async def generate_medical_response(self, prompt: str, context: Dict = None) -> Dict[str, Any]:
        """Generate medical response using Groq with few-shot prompting"""
        if not self.client:
            return {"success": False, "error": "Groq API key not configured"}
        
        try:
           
            messages = self._create_few_shot_messages(prompt, context)
            
            response = self.client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                max_tokens=settings.max_response_tokens,
                temperature=settings.medical_temperature
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model_used": settings.groq_model,
                "provider": "groq",
                "method": "few-shot"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "provider": "groq"
            }
    
    def _create_few_shot_messages(self, user_query: str, context: Dict = None) -> List[Dict]:
        """Create few-shot conversation messages for Groq"""
        messages = [
            {
                "role": "system",
                "content": """You are a medical AI assistant. Use the following examples to guide your responses:

Example 1:
Patient: I have chest pain and shortness of breath
Assistant: Analysis: These symptoms are concerning for potential cardiac issues.
Recommendations: Seek immediate medical attention. Call emergency services.
Urgency: Emergency

Example 2:
Patient: I have a headache and feel nauseous
Assistant: Analysis: This combination suggests possible migraine or tension headache.
Recommendations: Rest in a dark room, stay hydrated. Monitor symptoms.
Urgency: Moderate

Follow this format: provide Analysis, Recommendations, and Urgency level."""
            }
        ]
        
     
        if context and context.get("relevant_knowledge"):
            messages.append({
                "role": "system", 
                "content": f"Additional medical context: {', '.join(context['relevant_knowledge'])}"
            })
        
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        return messages

class DocumentProcessingService:
    """
    Service for processing uploaded medical documents
    """
    
    def __init__(self):
        self.supported_types = settings.allowed_file_types
        
    async def process_document(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Process uploaded medical document"""
        try:
            if file_type.lower() == "pdf":
                text = self._extract_pdf_text(file_path)
            elif file_type.lower() == "txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                return {"success": False, "error": f"Unsupported file type: {file_type}"}
            
          
            medical_info = self._extract_medical_info(text)
            
            return {
                "success": True,
                "extracted_text": text[:500] + "..." if len(text) > 500 else text,
                "medical_info": medical_info,
                "file_type": file_type
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            raise Exception(f"Failed to extract PDF text: {e}")
        return text
    
    def _extract_medical_info(self, text: str) -> Dict[str, List[str]]:
        """Extract medical information from document text"""
        info = {
            "medications": [],
            "conditions": [],
            "lab_values": [],
            "dates": []
        }
        
    
        med_patterns = [
            r'\b\w+cillin\b', r'\b\w+mycin\b', r'\b\w+pril\b',
            r'\baspirin\b', r'\bibuprofen\b', r'\bmetformin\b'  
        ]
        for pattern in med_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            info["medications"].extend(matches)
        
        # Lab values
        lab_patterns = [
            r'(\d+(?:\.\d+)?)\s*mg/dL',
            r'WBC:?\s*(\d+(?:\.\d+)?)',
            r'Hemoglobin:?\s*(\d+(?:\.\d+)?)'
        ]
        for pattern in lab_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            info["lab_values"].extend(matches)
        
        # Dates
        date_pattern = r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}'
        dates = re.findall(date_pattern, text)
        info["dates"].extend(dates)
        
        return info

class ContextService:
    """
    Service for managing conversation context
    """
    
    def __init__(self):
        self.sessions = {}  
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create new conversation session"""
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "messages": [],
            "medical_context": {
                "symptoms_mentioned": [],
                "conditions_discussed": [],
                "urgency_level": "normal"
            }
        }
        return self.sessions[session_id] 
    
    def add_message(self, session_id: str, user_message: str, ai_response: str, metadata: Dict = None):
        """Add message to conversation history"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        message_data = {
            "timestamp": datetime.now(),
            "user_message": user_message,
            "ai_response": ai_response,
            "metadata": metadata or {}
        }
        
        self.sessions[session_id]["messages"].append(message_data)
    
        if metadata and "symptoms" in metadata:
            self.sessions[session_id]["medical_context"]["symptoms_mentioned"].extend(metadata["symptoms"])
    
    def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation context"""
        return self.sessions.get(session_id)
    
    def get_conversation_summary(self, session_id: str) -> str:
        """Get summary of conversation"""
        context = self.get_context(session_id)
        if not context:
            return ""
        
        messages = context["messages"]
        if not messages:
            return "New conversation"
        
        symptoms = context["medical_context"]["symptoms_mentioned"]
        return f"Discussed symptoms: {', '.join(set(symptoms[:3]))} ({len(messages)} messages)"

class MedicalChatbotService:
    """
    Main orchestrator service that combines all other services
    """
    
    def __init__(self):
        self.rag_service = MedicalRAGService()
        self.hf_service = HuggingFaceService()
        self.groq_service = GroqService()
        self.doc_service = DocumentProcessingService()
        self.context_service = ContextService()
    
    async def process_medical_query(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """Main method to process medical queries (this is what your API will call)"""
        try:
         
            if is_emergency_symptom(message):
                return {
                    "success": True,
                    "response": settings.emergency_message,
                    "urgency_level": "emergency",
                    "emergency": True,
                    "session_id": session_id, 
                    "disclaimer": "SEEK IMMEDIATE MEDICAL ATTENTION!"
                }
            
          
            rag_context = self.rag_service.get_context_for_query(message)
            
         
            conversation_context = self.context_service.get_context(session_id)
            
          
            if self.groq_service.client:
                ai_response = await self.groq_service.generate_medical_response(message, rag_context)
            else:
                ai_response = await self.hf_service.generate_medical_response(message, rag_context)
            
            if not ai_response["success"]:
               
                response_text = self._create_fallback_response(rag_context)
            else:
                response_text = ai_response["response"]
            
       
            final_response = f"{response_text}\n\n{settings.medical_disclaimer}"
            
         
            self.context_service.add_message(
                session_id, message, final_response, 
                {"symptoms": rag_context["extracted_symptoms"]}
            )
            
            return {
                "success": True,
                "response": final_response,
                "urgency_level": "high" if rag_context["emergency_detected"] else "moderate",
                "emergency": rag_context["emergency_detected"],
                "possible_conditions": rag_context["possible_conditions"],
                "session_id": session_id,
                "disclaimer": settings.medical_disclaimer
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "I'm having trouble processing your request. Please consult a healthcare professional.",
                "session_id": session_id,  # Added session_id
                "disclaimer": settings.medical_disclaimer
            }
    
    def _create_fallback_response(self, rag_context: Dict) -> str:
        """Create fallback response when AI fails"""
        if rag_context["possible_conditions"]:
            conditions = ", ".join(rag_context["possible_conditions"][:3])
            return f"Based on your symptoms, possible conditions might include: {conditions}. Please consult a healthcare professional for proper evaluation."
        else:
            return "I understand you have health concerns. Please describe your symptoms in more detail or consult with a healthcare professional for proper medical advice."


medical_chatbot = MedicalChatbotService()

# Test function
if __name__ == "__main__":
    import asyncio
    
    async def test_service():
        print("🏥 Testing Medical Chatbot Services")
        
        # Test symptom analysis
        result = await medical_chatbot.process_medical_query("I have a headache and fever")
        print(f"✅ Response: {result['response'][:100]}...")
        print(f"Urgency: {result['urgency_level']}")
        
  
    asyncio.run(test_service())
