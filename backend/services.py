import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import requests
from groq import Groq
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from deep_translator import GoogleTranslator
from cachetools import TTLCache
import PyPDF2
from config import settings, is_emergency_symptom

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize cache for CoT responses
cache = TTLCache(maxsize=100, ttl=3600)

class MedicalRAGService:
    """RAG service for medical knowledge"""
    
    def __init__(self):
        self.medical_knowledge = self._load_medical_knowledge()
        self.symptom_database = self._load_symptom_database()
        self.embedding_model = SentenceTransformer(settings.hf_embedding_model)
        
    def _load_medical_knowledge(self) -> Dict[str, str]:
        """Load basic medical knowledge base"""
        return {
            "chest_pain": "Chest pain may indicate heart attack, angina, or muscle strain. Seek immediate attention if severe.",
            "headache": "Headaches can stem from stress, dehydration, or serious conditions. Monitor severity.",
            "fever": "Fever suggests infection or inflammation. Hydrate and monitor temperature.",
            "cough": "Persistent cough may indicate infection or allergies.",
            "shortness_of_breath": "Difficulty breathing may indicate heart or lung issues. Seek immediate care.",
            "nausea": "Nausea can result from digestive issues or infections.",
            "fatigue": "Persistent fatigue may indicate anemia or thyroid issues.",
            "rash": "Rashes may indicate allergies, infections, or autoimmune conditions.",
            "body_pain": "Generalized body pain may indicate fibromyalgia, infection, or chronic fatigue syndrome.",
            "dizziness": "Dizziness can result from dehydration, low blood pressure, or neurological issues."
        }
    
    def _load_symptom_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load symptom-to-condition mapping with weights"""
        return {
            "chest_pain": [
                {"condition": "Heart attack", "weight": 0.9},
                {"condition": "Angina", "weight": 0.8},
                {"condition": "Pulmonary embolism", "weight": 0.7},
                {"condition": "Muscle strain", "weight": 0.5}
            ],
            "shortness_of_breath": [
                {"condition": "Asthma", "weight": 0.8},
                {"condition": "Pneumonia", "weight": 0.7},
                {"condition": "COPD", "weight": 0.6},
                {"condition": "Heart failure", "weight": 0.5}
            ],
            "headache": [
                {"condition": "Migraine", "weight": 0.8},
                {"condition": "Tension headache", "weight": 0.7},
                {"condition": "Sinusitis", "weight": 0.6}
            ],
            "fever": [
                {"condition": "Viral infection", "weight": 0.8},
                {"condition": "Bacterial infection", "weight": 0.7},
                {"condition": "Flu", "weight": 0.6}
            ],
            "cough": [
                {"condition": "Common cold", "weight": 0.8},
                {"condition": "Bronchitis", "weight": 0.7},
                {"condition": "Pneumonia", "weight": 0.6}
            ],
            "nausea": [
                {"condition": "Gastritis", "weight": 0.8},
                {"condition": "Food poisoning", "weight": 0.7},
                {"condition": "Pregnancy", "weight": 0.5}
            ],
            "fatigue": [
                {"condition": "Chronic fatigue syndrome", "weight": 0.8},
                {"condition": "Anemia", "weight": 0.7},
                {"condition": "Thyroid issues", "weight": 0.6}
            ],
            "rash": [
                {"condition": "Viral exanthem", "weight": 0.8},
                {"condition": "Eczema", "weight": 0.7},
                {"condition": "Psoriasis", "weight": 0.6}
            ],
            "body_pain": [
                {"condition": "Fibromyalgia", "weight": 0.9},
                {"condition": "Chronic fatigue syndrome", "weight": 0.8},
                {"condition": "Viral infection", "weight": 0.6}
            ],
            "dizziness": [
                {"condition": "Dehydration", "weight": 0.8},
                {"condition": "Hypotension", "weight": 0.7},
                {"condition": "Vestibular disturbance", "weight": 0.6}
            ]
        }
    
    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """Get relevant medical context for a query"""
        extracted_symptoms = self._extract_symptoms(query)
        relevant_knowledge = [self.medical_knowledge[s] for s in extracted_symptoms if s in self.medical_knowledge]
        possible_conditions = []
        for symptom in extracted_symptoms:
            if symptom in self.symptom_database:
                possible_conditions.extend([c["condition"] for c in sorted(
                    self.symptom_database[symptom], key=lambda x: x["weight"], reverse=True
                )[:3]])  # Limit to top 3 conditions per symptom
        return {
            "extracted_symptoms": extracted_symptoms,
            "relevant_knowledge": relevant_knowledge,
            "possible_conditions": list(set(possible_conditions)),
            "emergency_detected": any(is_emergency_symptom(s) for s in extracted_symptoms)
        }
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from user text"""
        text_lower = text.lower()
        return [s for s in self.medical_knowledge.keys() if all(w in text_lower for w in s.replace("_", " ").split())]

class HuggingFaceService:
    """HuggingFace API service with CoT and Few-Shot prompting"""
    
    def __init__(self):
        self.client = InferenceClient(token=settings.huggingface_api_token)
        self.medical_model = settings.hf_medical_model
        self.embedding_model = SentenceTransformer(settings.hf_embedding_model)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.translator = GoogleTranslator()
        self.few_shot_examples = self._load_medical_examples()
        
    def _load_medical_examples(self) -> List[Dict[str, str]]:
        """Load few-shot medical examples from JSON"""
        try:
            with open(os.path.join(os.path.dirname(__file__), "medical_examples.json"), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("medical_examples.json not found. Using empty example set.")
            return []
    
    def _select_relevant_examples(self, query: str, max_examples: int = 3) -> List[Dict[str, str]]:
        """Select relevant few-shot examples based on similarity"""
        if not self.few_shot_examples:
            return []
        query_embedding = self.embedding_model.encode([query])[0]
        example_embeddings = self.embedding_model.encode([ex["input"] for ex in self.few_shot_examples])
        similarities = [sum(a * b for a, b in zip(query_embedding, ex_emb)) / (
            sum(a * a for a in query_embedding) ** 0.5 * sum(b * b for b in ex_emb) ** 0.5
        ) for ex_emb in example_embeddings]
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:max_examples]
        return [self.few_shot_examples[i] for i in top_indices]
    
    def _analyze_emotion(self, text: str) -> str:
        """Analyze emotional tone of user input"""
        try:
            result = self.sentiment_analyzer(text)[0]
            return "empathetic" if result["label"] == "NEGATIVE" and result["score"] > 0.7 else "neutral"
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return "neutral"
    
    def _create_cot_prompt(self, user_query: str, context: Dict = None) -> str:
        """Create Chain-of-Thought prompt with Few-Shot examples"""
        examples = self._select_relevant_examples(user_query)
        examples_text = "Examples of medical analysis:\n\n"
        for i, ex in enumerate(examples, 1):
            examples_text += f"Example {i}:\nPatient: {ex['input']}\nMedical Analysis: {ex['output']}\n\n"
        
        context_text = f"Relevant Knowledge: {', '.join(context['relevant_knowledge'])}\n" if context and context.get("relevant_knowledge") else ""
        urgency_text = f"Urgency: {'high' if context and context['emergency_detected'] else 'moderate'}\n"
        possible_conditions = f"Possible Conditions: {', '.join(context['possible_conditions'])}\n" if context and context.get("possible_conditions") else ""
        emotion = self._analyze_emotion(user_query)
        tone = "Use an empathetic tone for concerning symptoms." if emotion == "empathetic" else "Use a professional tone."
        
        return f"""You are a medical AI assistant. {tone} Follow this reasoning process:

1. Organize symptoms, severity, and relevant context.
2. List ONLY the provided possible conditions: {possible_conditions if possible_conditions else 'use provided medical knowledge'}.
3. Provide actionable recommendations and urgency level.
4. Do not add conditions outside the provided list.

{examples_text}
{context_text}
{possible_conditions}
{urgency_text}

Patient: {user_query}

Reasoning:
Step 1: Symptom Organization - [List symptoms, severity, history]
Step 2: Possible Conditions - [Select ONLY from: {possible_conditions if possible_conditions else 'provided medical knowledge'}]
Step 3: Recommendations - [Actionable advice, urgency]

Medical Analysis:
- Analysis: [Detailed clinical assessment]
- Recommendations: [Specific next steps]
- Urgency: [{'high' if context and context['emergency_detected'] else 'moderate'}]
- Confidence: [0-1 scale]

Always recommend consulting healthcare professionals for proper diagnosis."""
    
    async def generate_medical_response(self, prompt: str, context: Dict = None) -> Dict[str, Any]:
        """Generate medical response with CoT and Few-Shot"""
        cache_key = f"{prompt}:{context.get('emergency_detected', False)}:{context.get('possible_conditions', [])}"
        cached_response = cache.get(cache_key)
        if cached_response:
            logger.info(f"Retrieved cached response for prompt: {prompt[:50]}...")
            return {
                "success": True,
                "response": cached_response,
                "model_used": self.medical_model,
                "provider": "huggingface",
                "method": "cot_few_shot",
                "confidence": 0.8,
                "explanation": f"Reasoning based on cached response, {len(context['relevant_knowledge'] if context else 0)} knowledge sources, and {len(self._select_relevant_examples(prompt))} examples."
            }
        
        try:
            translated_prompt = self._translate_query(prompt)
            medical_prompt = self._create_cot_prompt(translated_prompt, context)
            logger.info(f"Generating HuggingFace response for: {translated_prompt[:50]}...")
            response = self.client.text_generation(
                medical_prompt,
                model=self.medical_model,
                max_new_tokens=settings.max_response_tokens,
                temperature=settings.medical_temperature,
                do_sample=True
            )
            refined_response = await self._refine_diagnosis(translated_prompt, response, context.get("session_id", "default") if context else "default", context)
            
            cache[cache_key] = refined_response
            return {
                "success": True,
                "response": refined_response,
                "model_used": self.medical_model,
                "provider": "huggingface",
                "method": "cot_few_shot",
                "confidence": 0.8,
                "explanation": f"Reasoning based on {len(context['relevant_knowledge'] if context else 0)} knowledge sources and {len(self._select_relevant_examples(prompt))} examples."
            }
        except Exception as e:
            logger.error(f"HuggingFace error: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "huggingface",
                "response": "Failed to generate response. Please try again.",
                "explanation": f"Reasoning based on fallback due to HuggingFace error: {str(e)}"
            }
    
    def _translate_query(self, query: str, target_lang: str = "en") -> str:
        """Translate query to English if needed"""
        try:
            if any(ord(char) > 127 for char in query):
                translated = self.translator.translate(query, dest=target_lang)
                logger.info(f"Translated query to English: {translated[:50]}...")
                return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
        return query
    
    async def _refine_diagnosis(self, user_query: str, response: str, session_id: str, context: Dict = None) -> str:
        """Refine diagnosis with follow-up questions if needed"""
        urgency = "high" if context and context.get("emergency_detected") else "moderate"
        # Validate response contains correct urgency and conditions
        if f"Urgency: {urgency}" not in response:
            response = response.replace(f"Urgency: {'moderate' if urgency == 'high' else 'high'}", f"Urgency: {urgency}")
        if context and context.get("possible_conditions"):
            valid_conditions = set(context["possible_conditions"])
            response_lines = response.split("\n")
            for i, line in enumerate(response_lines):
                if "Possible Conditions:" in line:
                    conditions = [c.strip() for c in line.split(":")[1].split(",") if c.strip()]
                    filtered_conditions = [c for c in conditions if c in valid_conditions]
                    response_lines[i] = f"Possible Conditions: {', '.join(filtered_conditions)}"
            response = "\n".join(response_lines)
        if len(response.split()) < 50 or "unspecified" in response.lower():
            clarifying_prompt = f"""Patient input: {user_query}\nInitial response: {response}\nAsk a clarifying question to refine diagnosis (e.g., duration, severity, or associated symptoms).\nUrgency: {urgency}\nPossible Conditions: {', '.join(context['possible_conditions'])}"""
            try:
                logger.info(f"Refining diagnosis for session: {session_id}")
                clarification = self.client.text_generation(
                    clarifying_prompt,
                    model=self.medical_model,
                    max_new_tokens=100,
                    temperature=0.5
                )
                return f"{response}\nClarification needed: {clarification}\n\nUrgency: {urgency}"
            except Exception as e:
                logger.error(f"Clarification error: {e}")
                return f"{response}\n\nUrgency: {urgency}"
        return f"{response}\n\nUrgency: {urgency}"

class GroqService:
    """Groq API service with CoT and Few-Shot prompting"""
    
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None
        self.embedding_model = SentenceTransformer(settings.hf_embedding_model)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.translator = GoogleTranslator()
        self.few_shot_examples = self._load_medical_examples()
    
    def _load_medical_examples(self) -> List[Dict[str, str]]:
        """Load few-shot medical examples from JSON"""
        try:
            with open(os.path.join(os.path.dirname(__file__), "medical_examples.json"), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("medical_examples.json not found. Using empty example set.")
            return []
    
    def _select_relevant_examples(self, query: str, max_examples: int = 3) -> List[Dict[str, str]]:
        """Select relevant few-shot examples based on similarity"""
        if not self.few_shot_examples:
            return []
        query_embedding = self.embedding_model.encode([query])[0]
        example_embeddings = self.embedding_model.encode([ex["input"] for ex in self.few_shot_examples])
        similarities = [sum(a * b for a, b in zip(query_embedding, ex_emb)) / (
            sum(a * a for a in query_embedding) ** 0.5 * sum(b * b for b in ex_emb) ** 0.5
        ) for ex_emb in example_embeddings]
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:max_examples]
        return [self.few_shot_examples[i] for i in top_indices]
    
    def _analyze_emotion(self, text: str) -> str:
        """Analyze emotional tone of user input"""
        try:
            result = self.sentiment_analyzer(text)[0]
            return "empathetic" if result["label"] == "NEGATIVE" and result["score"] > 0.7 else "neutral"
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return "neutral"
    
    def _create_few_shot_messages(self, user_query: str, context: Dict = None) -> List[Dict]:
        """Create few-shot conversation messages with CoT for Groq"""
        examples = self._select_relevant_examples(user_query)
        emotion = self._analyze_emotion(user_query)
        tone = "Use an empathetic tone for concerning symptoms." if emotion == "empathetic" else "Use a professional tone."
        
        examples_text = ""
        for ex in examples:
            examples_text += f"Patient: {ex['input']}\nAssistant: {ex['output']}\n\n"
        
        urgency_text = f"Urgency: {'high' if context and context['emergency_detected'] else 'moderate'}\n"
        possible_conditions = f"Possible Conditions: {', '.join(context['possible_conditions'])}\n" if context and context.get("possible_conditions") else ""
        
        messages = [{
            "role": "system",
            "content": f"""{tone} You are a medical AI assistant. Follow this reasoning process:
1. Organize symptoms, severity, and relevant context.
2. List ONLY the provided possible conditions: {possible_conditions if possible_conditions else 'use provided medical knowledge'}.
3. Provide actionable recommendations and urgency level.
4. Do not add conditions outside the provided list.

Examples:
{examples_text}
{possible_conditions}
{urgency_text}
Always recommend consulting healthcare professionals for proper diagnosis."""
        }]
        
        if context and context.get("relevant_knowledge"):
            messages.append({
                "role": "system",
                "content": f"Additional medical context: {', '.join(context['relevant_knowledge'])}"
            })
        
        messages.append({"role": "user", "content": user_query})
        return messages
    
    async def generate_medical_response(self, prompt: str, context: Dict = None) -> Dict[str, Any]:
        """Generate medical response with CoT and Few-Shot"""
        if not self.client:
            logger.warning("Groq API key not configured")
            return {
                "success": False,
                "error": "Groq API key not configured",
                "response": "Groq service unavailable.",
                "explanation": f"Reasoning based on fallback due to missing Groq API key."
            }
        
        cache_key = f"{prompt}:{context.get('emergency_detected', False)}:{context.get('possible_conditions', [])}"
        cached_response = cache.get(cache_key)
        if cached_response:
            logger.info(f"Retrieved cached response for prompt: {prompt[:50]}...")
            return {
                "success": True,
                "response": cached_response,
                "model_used": settings.groq_model,
                "provider": "groq",
                "method": "cot_few_shot",
                "confidence": 0.8,
                "explanation": f"Reasoning based on cached response, {len(context['relevant_knowledge'] if context else 0)} knowledge sources, and {len(self._select_relevant_examples(prompt))} examples."
            }
        
        try:
            translated_prompt = self._translate_query(prompt)
            messages = self._create_few_shot_messages(translated_prompt, context)
            logger.info(f"Generating Groq response for: {translated_prompt[:50]}...")
            response = self.client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                max_tokens=settings.max_response_tokens,
                temperature=settings.medical_temperature
            )
            refined_response = await self._refine_diagnosis(translated_prompt, response.choices[0].message.content, context.get("session_id", "default") if context else "default", context)
            
            cache[cache_key] = refined_response
            return {
                "success": True,
                "response": refined_response,
                "model_used": settings.groq_model,
                "provider": "groq",
                "method": "cot_few_shot",
                "confidence": 0.8,
                "explanation": f"Reasoning based on {len(context['relevant_knowledge'] if context else 0)} knowledge sources and {len(self._select_relevant_examples(prompt))} examples."
            }
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "groq",
                "response": "Failed to generate response. Please try again.",
                "explanation": f"Reasoning based on fallback due to Groq error: {str(e)}"
            }
    
    def _translate_query(self, query: str, target_lang: str = "en") -> str:
        """Translate query to English if needed"""
        try:
            if any(ord(char) > 127 for char in query):
                translated = self.translator.translate(query, dest=target_lang)
                logger.info(f"Translated query to English: {translated[:50]}...")
                return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
        return query
    
    async def _refine_diagnosis(self, user_query: str, response: str, session_id: str, context: Dict = None) -> str:
        """Refine diagnosis with follow-up questions if needed"""
        urgency = "high" if context and context.get("emergency_detected") else "moderate"
        # Validate response contains correct urgency and conditions
        if f"Urgency: {urgency}" not in response:
            response = response.replace(f"Urgency: {'moderate' if urgency == 'high' else 'high'}", f"Urgency: {urgency}")
        if context and context.get("possible_conditions"):
            valid_conditions = set(context["possible_conditions"])
            response_lines = response.split("\n")
            for i, line in enumerate(response_lines):
                if "Possible Conditions:" in line:
                    conditions = [c.strip() for c in line.split(":")[1].split(",") if c.strip()]
                    filtered_conditions = [c for c in conditions if c in valid_conditions]
                    response_lines[i] = f"Possible Conditions: {', '.join(filtered_conditions)}"
            response = "\n".join(response_lines)
        if len(response.split()) < 50 or "unspecified" in response.lower():
            clarifying_prompt = f"""Patient input: {user_query}\nInitial response: {response}\nAsk a clarifying question to refine diagnosis (e.g., duration, severity, or associated symptoms).\nUrgency: {urgency}\nPossible Conditions: {', '.join(context['possible_conditions'])}"""
            try:
                logger.info(f"Refining diagnosis for session: {session_id}")
                clarification = self.client.chat.completions.create(
                    model=settings.groq_model,
                    messages=[{"role": "user", "content": clarifying_prompt}],
                    max_tokens=100,
                    temperature=0.5
                )
                return f"{response}\nClarification needed: {clarification.choices[0].message.content}\n\nUrgency: {urgency}"
            except Exception as e:
                logger.error(f"Clarification error: {e}")
                return f"{response}\n\nUrgency: {urgency}"
        return f"{response}\n\nUrgency: {urgency}"

class DocumentProcessingService:
    """Service for processing uploaded medical documents"""
    
    def __init__(self, rag_service, ai_service):
        self.supported_types = settings.allowed_file_types
        self.ner = None
        self.rag_service = rag_service
        self.ai_service = ai_service
        try:
            self.ner = pipeline("ner", model="dmis-lab/biobert-v1.1", tokenizer="dmis-lab/biobert-v1.1", aggregation_strategy="simple")
            logger.info("BioBERT initialized successfully")
        except Exception as e:
            logger.error(f"BioBERT initialization error: {e}")
    
    async def process_document(self, file_path: str, file_type: str, session_id: str = "default") -> Dict[str, Any]:
        """Process uploaded medical document and generate medical analysis"""
        try:
            if file_type.lower() == "pdf":
                text = self._extract_pdf_text(file_path)
            elif file_type.lower() == "txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                return {"success": False, "error": f"Unsupported file type: {file_type}"}
            
            # Retry BioBERT initialization if not ready
            if not self.ner:
                try:
                    self.ner = pipeline("ner", model="dmis-lab/biobert-v1.1", tokenizer="dmis-lab/biobert-v1.1", aggregation_strategy="simple")
                    logger.info("BioBERT initialized on retry")
                except Exception as e:
                    logger.error(f"BioBERT retry initialization error: {e}")
            
            medical_info = self._extract_medical_info(text)
            logger.info(f"Extracted medical info: {medical_info}")
            
            # Generate medical analysis using extracted symptoms
            symptoms = medical_info["conditions"] + medical_info["symptoms"]
            if symptoms:
                symptom_query = ", ".join(symptoms)
                rag_context = self.rag_service.get_context_for_query(symptom_query)
                ai_response = await self.ai_service.generate_medical_response(symptom_query, rag_context)
                if ai_response["success"]:
                    urgency_level = "high" if rag_context["emergency_detected"] else "moderate"
                    analysis = {
                        "response": f"{ai_response['response']}",
                        "urgency_level": urgency_level,
                        "emergency": rag_context["emergency_detected"],
                        "possible_conditions": rag_context["possible_conditions"],
                        "confidence": ai_response.get("confidence", 0.8),
                        "explanation": ai_response.get("explanation", "N/A")
                    }
                else:
                    analysis = {
                        "response": f"Failed to analyze document: {ai_response['error']}\n\nUrgency: normal",
                        "urgency_level": "normal",
                        "emergency": False,
                        "possible_conditions": [],
                        "confidence": 0.0,
                        "explanation": "Analysis failed due to AI service error."
                    }
            else:
                analysis = {
                    "response": "No medical symptoms identified in the document. Please consult a healthcare professional for further evaluation.\n\nUrgency: normal",
                    "urgency_level": "normal",
                    "emergency": False,
                    "possible_conditions": [],
                    "confidence": 0.5,
                    "explanation": "No relevant symptoms detected for analysis."
                }
            
            return {
                "success": True,
                "extracted_text": text[:500] + "..." if len(text) > 500 else text,
                "medical_info": medical_info,
                "analysis": analysis,
                "file_type": file_type,
                "filename": os.path.basename(file_path)
            }
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise Exception(f"Failed to extract PDF text: {e}")
        return text
    
    def _extract_medical_info(self, text: str) -> Dict[str, List[str]]:
        """Extract medical information using BioBERT and regex"""
        info = {
            "medications": [],
            "conditions": [],
            "symptoms": [],
            "lab_values": [],
            "dates": []
        }
        
        # Regex-based symptom and condition extraction
        symptom_patterns = [
            r'\bgeneralized body pain\b',
            r'\bshortness of breath\b',
            r'\bheadache\b',
            r'\bdizziness\b',
            r'\bfatigue\b',
            r'\bcough\b',
            r'\bfever\b',
            r'\bnausea\b',
            r'\brash\b',
            r'\bchest pain\b'
        ]
        for pattern in symptom_patterns:
            info["symptoms"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        condition_patterns = [
            r'\bdiabetes\b',
            r'\bhypertension\b',
            r'\basthma\b',
            r'\bcopd\b',
            r'\bpneumonia\b',
            r'\bmigraine\b',
            r'\bfibromyalgia\b',
            r'\bchronic fatigue syndrome\b'
        ]
        for pattern in condition_patterns:
            info["conditions"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        medication_patterns = [
            r'\bmetformin\b',
            r'\binsulin\b',
            r'\balbuterol\b',
            r'\blisinopril\b'
        ]
        for pattern in medication_patterns:
            info["medications"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        if self.ner:
            try:
                # Process text in chunks to avoid token length issues
                chunk_size = 512
                text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                valid_conditions = {c["condition"] for s in self.rag_service.symptom_database.values() for c in s}
                for chunk in text_chunks:
                    entities = self.ner(chunk)
                    for e in entities:
                        if e["entity_group"] == "DISEASE" and e["score"] > 0.8 and e["word"] in valid_conditions:  # Stricter threshold
                            info["conditions"].append(e["word"])
                        elif e["entity_group"] == "DRUG" and e["score"] > 0.8:
                            info["medications"].append(e["word"])
                        elif e["entity_group"] == "SYMPTOM" and e["score"] > 0.8:
                            info["symptoms"].append(e["word"])
            except Exception as e:
                logger.error(f"BioBERT processing error: {e}")
        
        lab_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:mg/dL|mmHg|bpm|breaths/min|%)',
            r'WBC:?\s*(\d+(?:\.\d+)?)',
            r'Hemoglobin:?\s*(\d+(?:\.\d+)?)'
        ]
        for pattern in lab_patterns:
            info["lab_values"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        date_pattern = r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}'
        info["dates"].extend(re.findall(date_pattern, text))
        
        # Remove duplicates
        for key in info:
            info[key] = list(set(info[key]))
        
        return info

class ContextService:
    """Service for managing conversation context"""
    
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
    """Main orchestrator service"""
    
    def __init__(self):
        self.rag_service = MedicalRAGService()
        self.hf_service = HuggingFaceService()
        self.groq_service = GroqService()
        self.doc_service = DocumentProcessingService(self.rag_service, self.groq_service if self.groq_service.client else self.hf_service)
        self.context_service = ContextService()
    
    async def process_medical_query(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """Process medical queries with CoT and RAG"""
        try:
            if is_emergency_symptom(message):
                logger.info(f"Emergency detected for query: {message[:50]}...")
                return {
                    "success": True,
                    "response": settings.emergency_message,
                    "urgency_level": "emergency",
                    "emergency": True,
                    "session_id": session_id,
                    "disclaimer": "SEEK IMMEDIATE MEDICAL ATTENTION!",
                    "confidence": 1.0,
                    "explanation": "Reasoning based on emergency keyword detection."
                }
            
            rag_context = self.rag_service.get_context_for_query(message)
            conversation_context = self.context_service.get_context(session_id)
            
            if self.groq_service.client:
                logger.info("Using Groq service")
                ai_response = await self.groq_service.generate_medical_response(message, rag_context)
            else:
                logger.info("Using HuggingFace service")
                ai_response = await self.hf_service.generate_medical_response(message, rag_context)
            
            if not ai_response["success"]:
                response_text = self._create_fallback_response(rag_context)
                confidence = 0.5
                explanation = f"Reasoning based on fallback response due to AI service failure: {ai_response['error']}"
            else:
                response_text = ai_response["response"]
                confidence = ai_response.get("confidence", 0.8)
                explanation = ai_response.get("explanation", "Reasoning based on AI response.")
            
            final_response = f"{response_text}\n\n{settings.medical_disclaimer}"
            self.context_service.add_message(
                session_id, message, final_response,
                {"symptoms": rag_context["extracted_symptoms"]}
            )
            
            logger.info(f"Generated response for session {session_id}: {final_response[:50]}...")
            return {
                "success": True,
                "response": final_response,
                "urgency_level": "high" if rag_context["emergency_detected"] else "moderate",
                "emergency": rag_context["emergency_detected"],
                "possible_conditions": rag_context["possible_conditions"],
                "session_id": session_id,
                "disclaimer": settings.medical_disclaimer,
                "confidence": confidence,
                "explanation": explanation
            }
        except Exception as e:
            logger.error(f"MedicalChatbotService error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I'm having trouble processing your request. Please consult a healthcare professional.",
                "session_id": session_id,
                "disclaimer": settings.medical_disclaimer,
                "confidence": 0.0,
                "explanation": f"Reasoning based on error handling: {str(e)}"
            }
    
    def _create_fallback_response(self, rag_context: Dict) -> str:
        """Create fallback response when AI fails"""
        if rag_context["possible_conditions"]:
            conditions = ", ".join(rag_context["possible_conditions"][:3])
            return f"Based on your symptoms, possible conditions might include: {conditions}. Please consult a healthcare professional for proper evaluation.\n\nUrgency: {'high' if rag_context['emergency_detected'] else 'moderate'}"
        return "Please describe your symptoms in more detail or consult a healthcare professional.\n\nUrgency: normal"

# Clear cache to ensure fresh responses
cache.clear()

medical_chatbot = MedicalChatbotService()

if __name__ == "__main__":
    async def test_service():
        logger.info("🏥 Testing Medical Chatbot Services")
        result = await medical_chatbot.process_medical_query("chest pain, shortness of breath")
        logger.info(f"✅ Response: {result['response'][:100]}...")
        logger.info(f"Urgency: {result['urgency_level']}")
        logger.info(f"Confidence: {result['confidence']}")
        logger.info(f"Explanation: {result['explanation']}")
    
    asyncio.run(test_service())
