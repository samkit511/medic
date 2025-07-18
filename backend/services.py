import os
import re
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
from groq import Groq
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from deep_translator import GoogleTranslator
from cachetools import TTLCache
from PIL import Image
import pytesseract
import PyPDF2
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import settings, is_emergency_symptom

CLINICAL_QUESTIONS = [
    {"key": "name", "question": "Can I have your full name, please?"},
    {"key": "age", "question": "How old are you currently?"},
    {"key": "gender", "question": "What is your gender or how do you identify?"},
    {"key": "symptom_description", "question": "Can you explain in detail what you're feeling or experiencing?"},
    {"key": "onset", "question": "When did you first notice these symptoms?"},
    {"key": "duration", "question": "Have the symptoms been continuous, or do they come and go? And for how long?"},
    {"key": "location", "question": "Can you point to or describe exactly where the problem is?"},
    {"key": "character", "question": "How would you describe the nature of the symptom—like is it sharp, dull, burning, or something else?"},
    {"key": "severity", "question": "On a scale from 1 to 10, how intense or painful is it?"},
    {"key": "progression", "question": "Have your symptoms been improving, worsening, or staying the same since they began?"},
    {"key": "associated", "question": "Have you noticed any other symptoms happening along with this one—such as fever, chills, nausea, or fatigue?"},
    {"key": "aggravating", "question": "Have you found anything that makes the symptoms better or worse, like movement, food, or medications?"},
]

def get_next_question(session_data):
    for item in CLINICAL_QUESTIONS:
        if item["key"] not in session_data or not session_data[item["key"]]:
            return item["question"], item["key"]
    return None, None 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cache = TTLCache(maxsize=100, ttl=3600)

# Global model cache to prevent redundant loading
_model_cache = {}

class Singleton:
    def __init__(self, cls):
        self._cls = cls
        self._instance = None

    def __call__(self, *args, **kwargs):
        if self._instance is None:
            self._instance = self._cls(*args, **kwargs)
        return self._instance

@Singleton
class MedicalRAGService:
    def __init__(self):
        self.medical_knowledge = self._load_medical_knowledge()
        self.symptom_database = self._load_symptom_database()
        self.embedding_model = self._load_embedding_model()
        logger.info("MedicalRAGService initialized")

    def _load_embedding_model(self):
        model_key = "sentence_transformer"
        if model_key not in _model_cache:
            logger.info(f"Loading SentenceTransformer: {settings.hf_embedding_model}")
            _model_cache[model_key] = SentenceTransformer(settings.hf_embedding_model)
        else:
            logger.info(f"Using cached SentenceTransformer: {settings.hf_embedding_model}")
        return _model_cache[model_key]

    def _load_medical_knowledge(self) -> Dict[str, str]:
        return {
            "chest pain": "Chest pain may indicate heart attack, angina, or muscle strain. Seek immediate attention if severe.",
            "headache": "Headaches can stem from stress, dehydration, or serious conditions. Monitor severity.",
            "fever": "Fever suggests infection or inflammation. Hydrate and monitor temperature.",
            "cough": "Persistent cough may indicate infection or allergies.",
            "difficulty breathing": "Difficulty breathing may indicate heart or lung issues. Seek immediate care.",
            "nausea": "Nausea can result from digestive issues or infections.",
            "fatigue": "Persistent fatigue may indicate anemia or thyroid issues.",
            "rash": "Rashes may indicate allergies or autoimmune conditions.",
            "body pain": "Generalized body pain may indicate fibromyalgia or infection.",
            "dizziness": "Dizziness can result from dehydration or neurological issues."
        }

    def _load_symptom_database(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "chest pain": [
                {"condition": "Heart attack", "weight": 0.9},
                {"condition": "Angina", "weight": 0.8},
                {"condition": "Pulmonary embolism", "weight": 0.7},
                {"condition": "Muscle strain", "weight": 0.5}
            ],
            "difficulty breathing": [
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
            "body pain": [
                {"condition": "Fibromyalgia", "weight": 0.9},
                {"condition": "Chronic fatigue syndrome", "weight": 0.8},
                {"condition": "Viral infection", "weight": 0.6}
            ],
            "dizziness": [
                {"condition": "Dehydration", "weight": 0.8},
                {"condition": "Hypotension", "weight": 0.7},
                {"condition": "Vestibular disorder", "weight": 0.6}
            ]
        }

    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        extracted_symptoms = self._extract_symptoms(query)
        relevant_knowledge = [self.medical_knowledge[s] for s in extracted_symptoms if s in self.medical_knowledge]
        possible_conditions = []
        for symptom in extracted_symptoms:
            if symptom in self.symptom_database:
                possible_conditions.extend([c["condition"] for c in sorted(
                    self.symptom_database[symptom], key=lambda x: x["weight"], reverse=True
                )[:3]])
        logger.info(f"Extracted symptoms: {extracted_symptoms}")
        return {
            "extracted_symptoms": extracted_symptoms,
            "relevant_knowledge": relevant_knowledge,
            "possible_conditions": list(set(possible_conditions)),
            "emergency_detected": is_emergency_symptom(query)
        }

    def _extract_symptoms(self, text: str) -> List[str]:
        text_lower = text.lower()
        symptoms = []
        # Direct matches for medical knowledge keys
        for symptom in self.medical_knowledge.keys():
            if all(w in text_lower for w in symptom.split()):
                symptoms.append(symptom)
        # Handle synonyms for specific symptoms
        synonym_map = {
            "shortness of breath": "difficulty breathing",
            "trouble breathing": "difficulty breathing",
            "breathing difficulty": "difficulty breathing"
        }
        for synonym, canonical_symptom in synonym_map.items():
            if synonym in text_lower and canonical_symptom not in symptoms:
                symptoms.append(canonical_symptom)
        logger.debug(f"Symptoms extracted: {symptoms}")
        return symptoms

@Singleton
class HuggingFaceService:
    def __init__(self):
        self.client = InferenceClient(token=settings.huggingface_api_token) if settings.huggingface_api_token else None
        self.text_model = "microsoft/DialoGPT-large"  # Conversational model for text
        self.vision_model = settings.hf_medical_model  # Vision model for multimodal
        self.embedding_model = self._load_embedding_model()
        self.sentiment_analyzer = self._load_sentiment_analyzer()
        self.translator = GoogleTranslator(source='auto', target='en')

        self.few_shot_examples = self._load_medical_examples()
        logger.info(f"HuggingFaceService initialized, client available: {bool(self.client)}")
    def _translate_query(self, prompt):
        return self.translator.translate(prompt)


    def _load_embedding_model(self):
        model_key = "sentence_transformer"
        if model_key not in _model_cache:
            logger.info(f"Loading SentenceTransformer: {settings.hf_embedding_model}")
            _model_cache[model_key] = SentenceTransformer(settings.hf_embedding_model)
        else:
            logger.info(f"Using cached SentenceTransformer: {settings.hf_embedding_model}")
        return _model_cache[model_key]

    def _load_sentiment_analyzer(self):
        model_key = "sentiment_analyzer"
        if model_key not in _model_cache:
            logger.info("Loading sentiment analyzer: distilbert-base-uncased-finetuned-sst-2-english")
            _model_cache[model_key] = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        else:
            logger.info("Using cached sentiment analyzer")
        return _model_cache[model_key]

    def _load_medical_examples(self) -> List[Dict[str, str]]:
        try:
            with open(os.path.join(os.path.dirname(__file__), "medical_examples.json"), "r") as f:
                examples = json.load(f)
                logger.info(f"Loaded {len(examples)} medical examples")
                return examples
        except FileNotFoundError as e:
            logger.warning(f"medical_examples.json not found: {e}")
            return []

    def _select_relevant_examples(self, query: str, max_examples: int = 3) -> List[Dict[str, str]]:
        if not self.few_shot_examples:
            return []
        query_embedding = self.embedding_model.encode([query])[0]
        example_embeddings = self.embedding_model.encode([ex["input"] for ex in self.few_shot_examples])
        similarities = [sum(a * b for a, b in zip(query_embedding, ex_emb)) / (
            sum(a * a for a in query_embedding) ** 0.5 * sum(b * b for b in ex_emb) ** 0.5
        ) for ex_emb in example_embeddings]
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:max_examples]
        selected_examples = [self.few_shot_examples[i] for i in top_indices]
        logger.debug(f"Selected {len(selected_examples)} relevant examples")
        return selected_examples

    def _analyze_emotion(self, text: str) -> str:
        try:
            result = self.sentiment_analyzer(text)[0]
            tone = "empathetic" if result["label"] == "NEGATIVE" and result["score"] > 0.7 else "neutral"
            logger.info(f"Emotion analysis: {text[:50]}... -> {tone}")
            return tone
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return "neutral"

    def _create_cot_prompt(self, user_query: str, context: Dict = None) -> str:
        examples = self._select_relevant_examples(user_query)
        examples_text = "Examples of medical analysis:\n\n"
        for i, ex in enumerate(examples, 1):
            examples_text += f"Example {i}:\nPatient: {ex['input']}\nMedical Advice: {ex['output']}\n\n"
        
        context_text = f"Relevant Knowledge: {', '.join(context['relevant_knowledge'])}\n" if context and context.get("relevant_knowledge") else ""
        urgency_text = f"Urgency: {'emergency' if context and context['emergency_detected'] else 'normal'}\n"
        possible_conditions = f"Possible Conditions: {', '.join(context['possible_conditions'])}\n" if context and context.get("possible_conditions") else ""
        emotion = self._analyze_emotion(user_query)
        tone = "Use an empathetic tone for concerning symptoms." if emotion == "empathetic" else "Use a professional tone."
        
        prompt = f"""You are a medical AI assistant. {tone} Follow this reasoning process:

1. Organize symptoms, severity, and relevant context.
2. List ONLY the provided possible conditions: {possible_conditions if possible_conditions else 'use provided medical knowledge'}.
3. Provide actionable recommendations and urgency level.
4. Do not add conditions outside the provided list.
5. Assess confidence based on available information and ask clarifying questions if needed.

{examples_text}
{context_text}
{possible_conditions}
{urgency_text}

Patient: {user_query}

Reasoning:
Step 1: Symptom Organization - [List symptoms, severity, history]
Step 2: Possible Conditions - [Select from: {possible_conditions if possible_conditions else 'provided medical knowledge'}]
Step 3: Recommendations - [Actionable advice]
Step 4: Confidence Assessment - [Rate confidence as High (85-95%), Medium (70-84%), or Low (50-69%) based on symptom clarity and information completeness]

Medical Advice:
- Analysis: [Detailed assessment with personalized context]
- Recommendations: [Specific steps]
- Urgency: [{'emergency' if context and context['emergency_detected'] else 'normal'}]
- Confidence: High (85%) [REQUIRED: Must include confidence level with percentage in this exact format]
- Additional Questions: [If confidence is low, ask 1-2 specific clarifying questions]

IMPORTANT: Always include a confidence assessment in the format "Confidence: High (85%)" or "Confidence: Medium (72%)" or "Confidence: Low (55%)". Base the percentage on:
- High (80-95%): Clear symptoms, complete information, well-defined conditions
- Medium (60-79%): Some symptoms identified, partial information, moderate clarity
- Low (40-59%): Vague symptoms, incomplete information, unclear presentation

Always recommend consulting healthcare professionals."""
        logger.debug(f"CoT prompt created: {prompt[:100]}...")
        return prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.HTTPError, Exception))
    )
    async def generate_medical_response(self, prompt: str, context: Dict = None, is_multimodal: bool = False) -> Dict[str, Any]:
        if not self.client:
            logger.warning("HuggingFace API token not configured")
            return {
                "success": False,
                "error": "HuggingFace API token not configured",
                "response": "HuggingFace service unavailable.",
                "explanation": "Reasoning based on fallback due to missing API token."
            }
        
        cache_key = f"{prompt}:{context.get('emergency_detected', False)}:{context.get('possible_conditions', [])}"
        cached_response = cache.get(cache_key)
        if cached_response:
            logger.info(f"Retrieved cached response for prompt: {prompt[:50]}...")
            return {
                "success": True,
                "response": cached_response,
                "model_used": self.vision_model if is_multimodal else self.text_model,
                "provider": "huggingface",
                "method": "cot_few_shot",
                "confidence": 0.8,
                "explanation": f"Reasoning based on cached response, {len(context['relevant_knowledge'] if context else 0)} knowledge sources, and {len(self._select_relevant_examples(prompt))} examples."
            }
        
        try:
            translated_prompt = self._translate_query(prompt)
            medical_prompt = self._create_cot_prompt(translated_prompt, context)
            model = self.vision_model if is_multimodal else self.text_model
            logger.info(f"Generating HuggingFace response for: {translated_prompt[:50]}... with model: {model}")
            response = self.client.text_generation(
                medical_prompt,
                model=model,
                max_new_tokens=settings.max_response_tokens,
                temperature=settings.medical_temperature,
                return_full_text=False
            )
            refined_response = await self._refine_diagnosis(translated_prompt, response, context.get("session_id", "default") if context else "default", context)
            
            cache[cache_key] = refined_response
            logger.info(f"Response generated: {refined_response[:50]}...")
            return {
                "success": True,
                "response": refined_response,
                "model_used": model,
                "provider": "huggingface",
                "method": "cot_few_shot",
                "confidence": 0.8,
                "explanation": f"Reasoning based on {len(context['relevant_knowledge'] if context else 0)} knowledge sources and {len(self._select_relevant_examples(prompt))} examples."
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("HuggingFace API token is invalid or expired")
                return {
                    "success": False,
                    "error": "HuggingFace API token is invalid or expired",
                    "response": "Failed to generate response due to authentication error.",
                    "explanation": "HuggingFace API authentication failed."
                }
            elif e.response.status_code == 403:
                logger.error("HuggingFace API token lacks sufficient permissions")
                return {
                    "success": False,
                    "error": "HuggingFace API token lacks sufficient permissions",
                    "response": "Failed to generate response due to permission error.",
                    "explanation": "HuggingFace API permission error."
                }
            logger.error(f"HuggingFace error: {e}")
            raise
        except Exception as e:
            logger.error(f"HuggingFace error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "Failed to generate response. Please try again.",
                "explanation": f"Reasoning based on fallback due to error: {str(e)}"
            }

   
    def _translate_query(self, query: str) -> str:
        try:
        # Always attempt translation (deep_translator will skip if already Englissssh)
            translated = self.translator.translate(query)
            logger.info(f"Translated query to English: {translated[:50]}...")
            return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
        return query


    async def _refine_diagnosis(self, user_query: str, response: str, session_id: str, context: Dict = None) -> str:
        urgency = "emergency" if context and context.get("emergency_detected") else "normal"
        if f"Urgency: {urgency}" not in response:
            response = response.replace(f"Urgency: {'normal' if urgency == 'emergency' else 'low'}", f"Urgency: {urgency}")
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
            clarifying_prompt = f"""Patient input: {user_query}\nInitial response: {response}\nAsk a clarifying question to refine diagnosis (e.g., duration, severity, associated symptoms).\nUrgency: {urgency}\nPossible Conditions: {', '.join(context['possible_conditions'])}"""
            try:
                logger.info(f"Refining diagnosis for session: {session_id}")
                clarification = self.client.text_generation(
                    clarifying_prompt,
                    model=self.text_model,
                    max_new_tokens=100,
                    temperature=settings.medical_temperature
                )
                return f"{response}\nClarification needed: {clarification}\n\nUrgency: {urgency}"
            except Exception as e:
                logger.error(f"Clarification error: {e}")
                return f"{response}\n\nUrgency: {urgency}"
        return f"{response}\n\nUrgency: {urgency}"

@Singleton
class GroqService:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None
        self.embedding_model = self._load_embedding_model()
        self.sentiment_analyzer = self._load_sentiment_analyzer()
       
        self.translator = GoogleTranslator(source='auto', target='en')
        self.few_shot_examples = self._load_medical_examples()
        logger.info(f"GroqService initialized, client available: {bool(self.client)}")
    def _translate_query(self, prompt):
        return self.translator.translate(prompt)

    def _load_embedding_model(self):
        model_key = "sentence_transformer"
        if model_key not in _model_cache:
            logger.info(f"Loading SentenceTransformer: {settings.hf_embedding_model}")
            _model_cache[model_key] = SentenceTransformer(settings.hf_embedding_model)
        else:
            logger.info(f"Using cached SentenceTransformer: {settings.hf_embedding_model}")
        return _model_cache[model_key]

    def _load_sentiment_analyzer(self):
        model_key = "sentiment_analyzer"
        if model_key not in _model_cache:
            logger.info("Loading sentiment analyzer: distilbert-base-uncased-finetuned-sst-2-english")
            _model_cache[model_key] = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        else:
            logger.info("Using cached sentiment analyzer")
        return _model_cache[model_key]

    def _load_medical_examples(self) -> List[Dict[str, str]]:
        try:
            with open(os.path.join(os.path.dirname(__file__), "medical_examples.json"), "r") as f:
                examples = json.load(f)
                logger.info(f"Loaded {len(examples)} medical examples for Groq")
                return examples
        except FileNotFoundError as e:
            logger.warning(f"medical_examples.json not found: {e}")
            return []

    def _select_relevant_examples(self, query: str, max_examples: int = 3) -> List[Dict[str, str]]:
        if not self.few_shot_examples:
            return []
        query_embedding = self.embedding_model.encode([query])[0]
        example_embeddings = self.embedding_model.encode([ex["input"] for ex in self.few_shot_examples])
        similarities = [sum(a * b for a, b in zip(query_embedding, ex_emb)) / (
            sum(a * a for a in query_embedding) ** 0.5 * sum(b * b for b in ex_emb) ** 0.5
        ) for ex_emb in example_embeddings]
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:max_examples]
        selected_examples = [self.few_shot_examples[i] for i in top_indices]
        logger.debug(f"Selected {len(selected_examples)} examples for Groq")
        return selected_examples

    def _analyze_emotion(self, text: str) -> str:
        try:
            result = self.sentiment_analyzer(text)[0]
            tone = "empathetic" if result["label"] == "NEGATIVE" and result["score"] > 0.7 else "neutral"
            logger.info(f"Emotion analysis: {text[:50]}... -> {tone}")
            return tone
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return "neutral"

    def _create_few_shot_messages(self, user_query: str, context: Dict = None) -> List[Dict]:
        examples = self._select_relevant_examples(user_query)
        emotion = self._analyze_emotion(user_query)
        tone = "Use an empathetic tone for concerning symptoms." if emotion == "empathetic" else "Use a professional tone."
        
        examples_text = ""
        for ex in examples:
            examples_text += f"Patient: {ex['input']}\nAssistant: {ex['output']}\n\n"
        
        urgency_text = f"Urgency: {'emergency' if context and context['emergency_detected'] else 'normal'}"
        possible_conditions = f"Possible Conditions: {', '.join(context['possible_conditions'])}" if context and context.get("possible_conditions") else ""
        
        messages = [{
            "role": "system",
            "content": f"""{tone}
You are a medical AI assistant. Follow these guidelines:

1. Organize symptoms, severity, and context.
2. List ONLY provided conditions: {possible_conditions if possible_conditions else 'use medical knowledge base'}.
3. Provide actionable recommendations and urgency level.
4. Do not include unlisted conditions.

Examples:
{examples_text}
{possible_conditions}
{urgency_text}

Always recommend consulting healthcare providers."""
        }]
        
        if context and context.get("relevant_knowledge"):
            messages.append({
                "role": "system",
                "content": f"Additional context: {', '.join(context['relevant_knowledge'])}"
            })
        
        messages.append({"role": "user", "content": user_query})
        logger.debug(f"Created {len(messages)} messages for Groq")
        return messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def generate_medical_response(self, prompt: str, context: Dict = None) -> Dict[str, Any]:
        if not self.client:
            logger.warning("Groq API key not configured")
            return {
                "success": False,
                "error": "Groq API key not configured",
                "response": "Groq service unavailable.",
                "explanation": "Reasoning based on fallback due to missing API key."
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
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=settings.groq_model,
                    messages=messages,
                    max_tokens=settings.max_response_tokens,
                    temperature=settings.medical_temperature
                )
            )
            refined_response = await self._refine_diagnosis(translated_prompt, response.choices[0].message.content, context.get("session_id", "default") if context else "default", context)
            
            cache[cache_key] = refined_response
            logger.info(f"Generated response: {refined_response[:50]}...")
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
                "response": "Failed to generate response.",
                "explanation": f"Failed to reason due to: {str(e)}"
            }

    def _translate_query(self, query: str) -> str:
        try:
            translated = self.translator.translate(query)
            logger.info(f"Translated query: {query[:50]}... -> {translated[:50]}...")
            return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
        return query


    async def _refine_diagnosis(self, user_query: str, response: str, session_id: str, context: Dict = None) -> str:
        urgency = "emergency" if context and context.get("emergency_detected") else "normal"
        urgency_emoji = "🚨" if urgency == "emergency" else "ℹ️"
        
        # Extract dynamic confidence score from AI response
        dynamic_confidence = self._extract_confidence_from_response(response)
        
        # Format the response with better structure
        formatted_response = self._format_medical_response(response, urgency, context, dynamic_confidence)
        
        if len(response.split()) < 50 or "unspecified" in response.lower():
            prompt = f"""Patient input: {user_query}\nInitial response: {response}\nAsk a clarifying question to refine diagnosis (e.g., duration, severity).\nUrgency: {urgency}\nPossible Conditions: {', '.join(context['possible_conditions']) if context and context.get('possible_conditions') else 'N/A'}"""
            try:
                logger.info(f"Refining diagnosis for session: {session_id}")
                clarification = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model=settings.groq_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100,
                        temperature=settings.medical_temperature
                    )
                )
                formatted_response += f"\n\n**📋 Additional Information Needed:**\n{clarification.choices[0].message.content}"
            except Exception as e:
                logger.error(f"Clarification error: {e}")
        
        return formatted_response
    
    def _format_medical_response(self, response: str, urgency: str, context: Dict = None, dynamic_confidence: str = None) -> str:
        """Format medical response with enhanced readability and structure"""
        try:
            # Initialize urgency formatting
            urgency_emoji = "🚨" if urgency == "emergency" else "ℹ️"
            urgency_color = "🚨 EMERGENCY" if urgency == "emergency" else "ℹ️ NORMAL"
            
            # Extract key components from response
            lines = response.split('\n')
            formatted_lines = []
            current_section = None
            section_content = []
            
            # Define section mappings
            section_mappings = {
                'analysis': {'emoji': '🔍', 'title': 'MEDICAL ANALYSIS'},
                'recommendations': {'emoji': '📋', 'title': 'RECOMMENDATIONS'},
                'conditions': {'emoji': '🩺', 'title': 'POSSIBLE CONDITIONS'},
                'symptoms': {'emoji': '🩹', 'title': 'SYMPTOMS IDENTIFIED'},
                'diagnosis': {'emoji': '⚕️', 'title': 'DIAGNOSIS'},
                'treatment': {'emoji': '💊', 'title': 'TREATMENT OPTIONS'},
                'follow_up': {'emoji': '📅', 'title': 'FOLLOW-UP CARE'},
                'confidence': {'emoji': '🎯', 'title': 'AI CONFIDENCE ASSESSMENT'}
            }
            
            def flush_section():
                """Add current section to formatted lines"""
                if current_section and section_content:
                    section_info = section_mappings.get(current_section, {'emoji': '📝', 'title': current_section.upper()})
                    formatted_lines.append(f"\n{section_info['emoji']} **{section_info['title']}**\n")
                    
                    # Format content based on section type
                    if current_section in ['recommendations', 'treatment', 'follow_up']:
                        for item in section_content:
                            if item.strip():
                                formatted_lines.append(f"   • {item.strip()}")
                    else:
                        formatted_lines.extend(section_content)
                    
                    formatted_lines.append("")  # Add spacing after section
            
            # Process each line
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect section headers
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ['analysis:', 'medical analysis']):
                    flush_section()
                    current_section = 'analysis'
                    section_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                elif any(keyword in line_lower for keyword in ['recommendations:', 'recommend']):
                    flush_section()
                    current_section = 'recommendations'
                    section_content = []
                elif any(keyword in line_lower for keyword in ['possible conditions:', 'conditions:', 'differential']):
                    flush_section()
                    current_section = 'conditions'
                    section_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                elif any(keyword in line_lower for keyword in ['symptoms:', 'symptoms identified']):
                    flush_section()
                    current_section = 'symptoms'
                    section_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                elif any(keyword in line_lower for keyword in ['diagnosis:', 'preliminary diagnosis']):
                    flush_section()
                    current_section = 'diagnosis'
                    section_content = [line.split(':', 1)[1].strip() if ':' in line else line]
                elif any(keyword in line_lower for keyword in ['treatment:', 'treatment options']):
                    flush_section()
                    current_section = 'treatment'
                    section_content = []
                elif any(keyword in line_lower for keyword in ['follow-up:', 'follow up']):
                    flush_section()
                    current_section = 'follow_up'
                    section_content = []
                elif line.startswith('Confidence:'):
                    flush_section()
                    current_section = 'confidence'
                    confidence_text = line.replace('Confidence:', '').strip()
                    section_content = [confidence_text]
                elif line.startswith(('Urgency:', 'Emergency:')):
                    # Skip urgency lines as we handle them separately
                    continue
                else:
                    # Handle list items and bullet points
                    if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                        # Remove number and add as bullet point
                        content = line.split('.', 1)[1].strip() if '.' in line else line
                        section_content.append(content)
                    elif line.startswith(('*', '-', '•')):
                        # Remove bullet and add content
                        content = line[1:].strip()
                        section_content.append(content)
                    elif line.startswith('   '):
                        # Indented content, likely part of a list
                        section_content.append(line.strip())
                    else:
                        # Regular content
                        if current_section:
                            section_content.append(line)
                        else:
                            # No current section, add as general content
                            formatted_lines.append(line)
            
            # Flush any remaining section
            flush_section()
            
            # Build formatted response
            formatted_response = "\n".join(formatted_lines)
            
            # Clean up extra whitespace
            formatted_response = '\n'.join(line for line in formatted_response.split('\n') if line.strip() or not line)
            
            # Add patient profile if available
            if context and context.get('user_data'):
                user_data = context['user_data']
                profile_lines = ["\n👤 **PATIENT PROFILE:**"]
                
                if user_data.get('age'):
                    profile_lines.append(f"\n   • Age: {user_data['age']}")
                if user_data.get('gender'):
                    profile_lines.append(f"\n   • Gender: {user_data['gender']}")
                if user_data.get('medical_history'):
                    profile_lines.append(f"\n   • Medical History: {user_data['medical_history']}")
                if user_data.get('medications'):
                    profile_lines.append(f"\n   • Current Medications: {user_data['medications']}")
                if user_data.get('allergies'):
                    profile_lines.append(f"\n   • Known Allergies: {user_data['allergies']}")
                if user_data.get('symptoms_duration'):
                    profile_lines.append(f"\n   • Symptoms Duration: {user_data['symptoms_duration']}")
                
                if len(profile_lines) > 1:  # Only add if there's actual data
                    formatted_response += "\n" + "\n".join(profile_lines)
            
            # Add confidence and urgency status
            status_lines = []
            
            # Dynamic confidence score
            if dynamic_confidence:
                status_lines.append(f"\n🎯 **CONFIDENCE LEVEL:** {dynamic_confidence}")
            else:
                status_lines.append(f"\n🎯 **CONFIDENCE LEVEL:** Not available")
            
            # Urgency status with appropriate formatting
            if urgency == "emergency":
                status_lines.append(f"\n🚨 **URGENCY STATUS:** {urgency_color}")
                status_lines.append(f"\n⚠️ **IMPORTANT:** Seek immediate medical attention!")
            else:
                status_lines.append(f"\n{urgency_emoji} **URGENCY STATUS:** {urgency_color}")
            
            formatted_response += "\n" + "\n".join(status_lines)
            
            # Add contextual disclaimers
            disclaimer_lines = []
            disclaimer_lines.append("\n⚠️ **MEDICAL DISCLAIMER:**")
            disclaimer_lines.append("\n  • This is AI-generated medical information for educational purposes only")
            disclaimer_lines.append("\n   • Always consult qualified healthcare professionals for diagnosis and treatment")
            disclaimer_lines.append("\n  • In case of emergency, contact emergency services immediately")
            
            if urgency == "emergency":
                disclaimer_lines.append("   • 🚨 EMERGENCY DETECTED - Seek immediate medical care!")
            
            formatted_response += "\n" + "\n".join(disclaimer_lines)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting medical response: {e}")
            # Fallback to simple formatting
            fallback_response = f"{urgency_emoji} **MEDICAL RESPONSE**\n\n{response}\n\n"
            if dynamic_confidence:
                fallback_response += f"🎯 **CONFIDENCE:** {dynamic_confidence}\n\n"
            fallback_response += f"⚠️ **URGENCY:** {urgency_color}\n\n"
            fallback_response += "⚠️ **DISCLAIMER:** This is AI-generated medical information. Consult a healthcare professional.\n"
            return fallback_response
    
    def _extract_confidence_from_response(self, response: str) -> str:
        """Extract dynamic confidence score from AI response"""
        import re
        
        # Log the full response for debugging
        logger.info(f"=== CONFIDENCE EXTRACTION DEBUG ===")
        logger.info(f"Full response length: {len(response)}")
        logger.info(f"Response content: {response}")
        logger.info(f"=== END RESPONSE ===")
        
        # Enhanced confidence patterns - more comprehensive and flexible
        confidence_patterns = [
            # Standard patterns - case insensitive
            r'Confidence\s*:?\s*(High|Medium|Low)(?:\s*\((\d+)%\))?',
            r'(?:My\s+)?[Cc]onfidence(?:\s+(?:level|score|rating|assessment))?\s*(?:is|:)?\s*(High|Medium|Low)(?:\s*\((\d+)%\))?',
            r'(?:Overall\s+)?[Cc]onfidence\s*(?:level|score|rating)?\s*(?:is|:)?\s*(High|Medium|Low)(?:\s*\((\d+)%\))?',
            r'Assessment\s+[Cc]onfidence\s*:?\s*(High|Medium|Low)(?:\s*\((\d+)%\))?',
            
            # More flexible patterns
            r'I\s+(?:am|have)\s*(High|Medium|Low)\s+confidence(?:\s*\((\d+)%\))?',
            r'(?:With\s+)?(?:a\s+)?(High|Medium|Low)\s+(?:level\s+of\s+)?confidence(?:\s*\((\d+)%\))?',
            r'Confidence\s*[-:]\s*(High|Medium|Low)(?:\s*\((\d+)%\))?',
            r'(?:The\s+)?confidence\s+(?:is|level\s+is)\s*(High|Medium|Low)(?:\s*\((\d+)%\))?',
            
            # Percentage-first patterns
            r'(\d+)%\s+confidence',
            r'confidence\s*(?:of|is|:)\s*(\d+)%',
            r'I\s+am\s+(\d+)%\s+confident',
            r'(?:At\s+)?(\d+)%\s+confidence',
            r'confidence\s*(?:rating|score|level)?\s*(?:of|is|:)?\s*(\d+)%',
            
            # Fallback patterns with confidence words
            r'(?:very\s+|quite\s+|reasonably\s+)?confident(?:\s*\((\d+)%\))?',
            r'(?:not\s+very\s+|low\s+|limited\s+)?confident(?:\s*\((\d+)%\))?',
            r'(?:high|medium|low)\s+certainty',
            r'(?:very\s+|quite\s+|reasonably\s+)?certain(?:\s*\((\d+)%\))?',
            
            # Line-based patterns (for when confidence is on its own line)
            r'^\s*-\s*Confidence\s*:?\s*(High|Medium|Low)(?:\s*\((\d+)%\))?',
            r'^\s*Confidence\s*:?\s*(High|Medium|Low)(?:\s*\((\d+)%\))?',
        ]
        
        # Try each pattern
        for i, pattern in enumerate(confidence_patterns):
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                logger.info(f"✓ Confidence pattern {i+1} matched: '{match.group(0)}'")
                logger.info(f"  Match groups: {match.groups()}")
                
                # Handle different pattern types
                if i < 8:  # High/Medium/Low patterns
                    confidence_level = match.group(1).capitalize()
                    percentage = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                    
                    if percentage:
                        result = f"{confidence_level} ({percentage}%)"
                    else:
                        # Map confidence levels to default percentages
                        default_percentages = {
                            'High': '85%',
                            'Medium': '70%',
                            'Low': '50%'
                        }
                        result = f"{confidence_level} ({default_percentages.get(confidence_level, '70%')})"
                    
                    logger.info(f"✓ Extracted confidence: {result}")
                    return result
                    
                elif i < 13:  # Percentage-only patterns
                    percentage = int(match.group(1))
                    if percentage >= 80:
                        level = "High"
                    elif percentage >= 60:
                        level = "Medium"
                    else:
                        level = "Low"
                    
                    result = f"{level} ({percentage}%)"
                    logger.info(f"✓ Extracted confidence from percentage: {result}")
                    return result
                    
                else:  # Fallback patterns
                    full_match = match.group(0).lower()
                    if 'not' in full_match or 'low' in full_match or 'limited' in full_match:
                        result = "Low (50%)"
                    elif 'very' in full_match or 'quite' in full_match or 'high' in full_match:
                        result = "High (85%)"
                    else:
                        result = "Medium (70%)"
                    
                    logger.info(f"✓ Extracted confidence from fallback: {result}")
                    return result
        
        # Try to find any percentage that might be confidence-related
        percentage_matches = re.findall(r'(\d+)%', response)
        if percentage_matches:
            logger.info(f"Found percentages: {percentage_matches}")
            # Use the first reasonable percentage (40-100%)
            for pct_str in percentage_matches:
                percentage = int(pct_str)
                if 40 <= percentage <= 100:
                    if percentage >= 80:
                        level = "High"
                    elif percentage >= 60:
                        level = "Medium"
                    else:
                        level = "Low"
                    
                    result = f"{level} ({percentage}%)"
                    logger.info(f"✓ Extracted confidence from percentage fallback: {result}")
                    return result
        
        # Final fallback - check for confidence-related words without specific patterns
        confidence_words = re.findall(r'\b(confident|confidence|certain|certainty|sure|unsure)\b', response, re.IGNORECASE)
        if confidence_words:
            logger.info(f"Found confidence words: {confidence_words}")
            # Basic heuristic based on presence of confidence words
            if any(word.lower() in ['unsure', 'uncertain'] for word in confidence_words):
                result = "Low (50%)"
            elif any(word.lower() in ['confident', 'certain', 'sure'] for word in confidence_words):
                result = "Medium (70%)"
            else:
                result = "Medium (70%)"
            
            logger.info(f"✓ Extracted confidence from word heuristic: {result}")
            return result
        
        # If no confidence pattern found, return None to use static scoring
        logger.warning("⚠️ No confidence pattern found in response - will use static fallback")
        return None

class DocumentProcessingService:
    def __init__(self, rag_service, ai_service):
        self.supported_types = settings.allowed_file_types
        self.ner = None
        self.rag_service = rag_service
        self.ai_service = ai_service
        try:
            self.ner = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER", aggregation_strategy="simple")
            logger.info("BERT NER initialized")
        except Exception as e:
            logger.error(f"BERT NER initialization failed: {e}")

    async def process_document(self, file_path: str, file_type: str, session_id: str = "default") -> Dict[str, Any]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"success": False, "error": f"File not found: {file_path}"}
        
        try:
            if file_type.lower() == "pdf":
                text = self._extract_pdf_text(file_path)
            elif file_type.lower() == "txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_type.lower() in ["jpg", "png"]:
                try:
                    image = Image.open(file_path)
                    dpi = image.info.get('dpi', (72, 72))[0]
                    if dpi < 300:
                        width, height = image.size
                        new_width = int(width * (300 / dpi))
                        new_height = int(height * (300 / dpi))
                        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    text = pytesseract.image_to_string(image)
                    logger.info(f"Extracted text from image: {text[:50]}...")
                except Exception as e:
                    logger.error(f"OCR error: {e}")
                    return {"success": False, "error": f"Failed to extract text from image: {e}"}
            else:
                logger.error(f"Unsupported file type: {file_type}")
                return {"success": False, "error": f"Unsupported file type: {file_type}"}
            
            if not self.ner:
                try:
                    self.ner = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER", aggregation_strategy="simple")
                    logger.info("BERT NER initialized on retry")
                except Exception as e:
                    logger.error(f"BERT NER retry failed: {e}")
            
            medical_info = self._extract_medical_info(text)
            logger.info(f"Extracted medical info: {medical_info}")
            
            symptoms = medical_info["conditions"] + medical_info["symptoms"]
            if symptoms:
                symptom_query = ", ".join(symptoms)
                rag_context = self.rag_service.get_context_for_query(symptom_query)
                # Check if ai_service is GroqService, which doesn't support is_multimodal parameter
                if hasattr(self.ai_service, 'client') and hasattr(self.ai_service.client, 'chat'):
                    # This is GroqService
                    ai_response = await self.ai_service.generate_medical_response(symptom_query, rag_context)
                else:
                    # This is HuggingFaceService
                    ai_response = await self.ai_service.generate_medical_response(symptom_query, rag_context, is_multimodal=(file_type.lower() in ["jpg", "png"]))
                if ai_response["success"]:
                    urgency_level = "emergency" if rag_context["emergency_detected"] else "normal"
                    analysis = {
                        "response": f"{ai_response['response']}\n\n{settings.medical_disclaimer}",
                        "urgency_level": urgency_level,
                        "emergency": rag_context["emergency_detected"],
                        "possible_conditions": rag_context["possible_conditions"],
                        "confidence": ai_response.get("confidence", 0.8),
                        "explanation": ai_response.get("explanation", "N/A")
                    }
                else:
                    analysis = {
                        "response": f"Failed to analyze: {ai_response['error']}\n\n{settings.medical_disclaimer}",
                        "urgency_level": "normal",
                        "emergency": False,
                        "possible_conditions": [],
                        "confidence": 0.0,
                        "explanation": "Analysis failed due to AI service error."
                    }
            else:
                analysis = {
                    "response": f"No symptoms identified. Please consult a healthcare professional.\n\n{settings.medical_disclaimer}",
                    "urgency_level": "normal",
                    "emergency": False,
                    "possible_conditions": [],
                    "confidence": 0.5,
                    "explanation": "No relevant symptoms detected."
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
        if not os.path.exists(file_path):
            logger.error(f"PDF not found: {file_path}")
            raise Exception(f"File not found: {file_path}")
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
            logger.info(f"Extracted text from PDF: {text[:50]}...")
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise Exception(f"Failed to extract PDF text: {e}")

    def _extract_medical_info(self, text: str) -> Dict[str, List[str]]:
        info = {
            "medications": [],
            "conditions": [],
            "symptoms": [],
            "lab_values": [],
            "dates": []
        }
        
        symptom_patterns = [
            r'\bchest pain\b',
            r'\bdifficulty breathing\b',
            r'\bshortness of breath\b',
            r'\btrouble breathing\b',
            r'\bbreathing difficulty\b',
            r'\bheadache\b',
            r'\bdizziness\b',
            r'\bfatigue\b',
            r'\bcough\b',
            r'\bfever\b',
            r'\bnausea\b',
            r'\brash\b',
            r'\bbody pain\b'
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
            r'\baspirin\b',
            r'\bibuprofen\b',
            r'\bparacetamol\b',
            r'\bmetformin\b',
            r'\blisinopril\b'
        ]
        for pattern in medication_patterns:
            info["medications"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        lab_value_patterns = [
            r'\b\d+\.\d+\s*(?:mg/dL|mmol/L|%)',
            r'\b\d+\s*(?:bpm|mmHg)'
        ]
        for pattern in lab_value_patterns:
            info["lab_values"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b'
        ]
        for pattern in date_patterns:
            info["dates"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        if self.ner:
            try:
                entities = self.ner(text)
                for entity in entities:
                    if entity["entity_group"] in ["PER", "ORG", "LOC"]:
                        continue
                    if "symptom" in entity["entity_group"].lower():
                        info["symptoms"].append(entity["word"])
                    elif "condition" in entity["entity_group"].lower():
                        info["conditions"].append(entity["word"])
            except Exception as e:
                logger.error(f"NER extraction error: {e}")
        
        return info

class ContextService:
    def __init__(self):
        self.contexts = {}

    def create_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.contexts:
            self.contexts[session_id] = {
                "created_at": datetime.now(),
                "messages": [],
                "documents": [],
                "medical_context": {
                    "symptoms": [],
                    "conditions": [],
                    "medications": []
                },
                "last_updated": datetime.now()
            }
            logger.info(f"Created new session: {session_id}")
        return self.contexts[session_id]

    def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        context = self.contexts.get(session_id)
        if context:
            logger.debug(f"Retrieved context for session: {session_id}")
        else:
            logger.warning(f"Session not found: {session_id}")
        return context

    def update_context(self, session_id: str, message: str, response: str, medical_info: Dict = None) -> None:
        if session_id not in self.contexts:
            self.create_session(session_id)
        
        self.contexts[session_id]["messages"].append({
            "user_message": message,
            "bot_response": response,
            "timestamp": datetime.now()
        })
        
        if medical_info:
            self.contexts[session_id]["medical_context"]["symptoms"].extend(medical_info.get("symptoms", []))
            self.contexts[session_id]["medical_context"]["conditions"].extend(medical_info.get("conditions", []))
            self.contexts[session_id]["medical_context"]["medications"].extend(medical_info.get("medications", []))
        
        self.contexts[session_id]["last_updated"] = datetime.now()
        logger.info(f"Updated context for session: {session_id}")

    def add_document(self, session_id: str, document_path: str) -> None:
        if session_id not in self.contexts:
            self.create_session(session_id)
        self.contexts[session_id]["documents"].append(document_path)
        self.contexts[session_id]["last_updated"] = datetime.now()
        logger.info(f"Added document to session: {session_id}, document: {document_path}")

    def get_conversation_summary(self, session_id: str) -> str:
        context = self.get_context(session_id)
        if not context:
            return "No conversation history available."
        
        messages = context["messages"]
        if not messages:
            return "No messages in this session."
        
        summary = f"Conversation summary for session {session_id}:\n"
        for msg in messages[-3:]:  # Summarize last 3 messages
            summary += f"User: {msg['user_message'][:50]}...\nBot: {msg['bot_response'][:50]}...\n"
        logger.debug(f"Generated summary for session: {session_id}")
        return summary
class MedicalChatbot:
    def __init__(self):
       

        self.rag_service = MedicalRAGService()
        self.hf_service = HuggingFaceService()
        self.groq_service = GroqService()
        self.doc_service = DocumentProcessingService(
            self.rag_service,
            self.groq_service if settings.groq_api_key and settings.groq_api_key.strip() else self.hf_service
        )
        self.context_service = ContextService()
        logger.info("MedicalChatbot initialized")

    def get_next_clinical_question(self, session_id):
        session = self.context_service.get_context(session_id)
        session_data = session.get("collected_data", {}) if session else {}
        question, key = get_next_question(session_data)
        return question, key

    async def process_user_input(self, user_input, session_id):
        session = self.context_service.get_context(session_id)
        if not session:
            session = self.context_service.create_session(session_id)
        collected_data = session.setdefault("collected_data", {})
        
        # Initialize greeting status and question index if not set
        if "greeting_shown" not in session:
            session["greeting_shown"] = False
        if "current_question_index" not in session:
            session["current_question_index"] = 0
            
        # Handle special initial greeting request
        if user_input == "__INITIAL_GREETING__":
            if not session["greeting_shown"]:
                session["greeting_shown"] = True
                return {"next_question": "Hello! May I know your name?", "session_id": session_id}
            else:
                # If greeting already shown, return current question
                current_index = session["current_question_index"]
                if current_index < len(CLINICAL_QUESTIONS):
                    current_question = CLINICAL_QUESTIONS[current_index]
                    return {"next_question": current_question["question"], "session_id": session_id}
                else:
                    return {"next_question": "Thank you for the information. Please proceed with your diagnosis.", "session_id": session_id}
            
        # If this is the first real interaction (no greeting shown yet), show greeting
        if not session["greeting_shown"]:
            session["greeting_shown"] = True
            return {"next_question": "Hello! May I know your name?", "session_id": session_id}
        
        # Process the user input - save the answer for the current question
        current_index = session["current_question_index"]
        
        # If we have a valid current question index, save the answer
        if current_index < len(CLINICAL_QUESTIONS):
            current_question = CLINICAL_QUESTIONS[current_index]
            collected_data[current_question["key"]] = user_input
            self.context_service.update_context(session_id, user_input, "", medical_info=None)
            
            # Move to next question
            session["current_question_index"] += 1
        
        # Check if we have more questions to ask
        next_index = session["current_question_index"]
        if next_index < len(CLINICAL_QUESTIONS):
            next_question = CLINICAL_QUESTIONS[next_index]
            return {"next_question": next_question["question"], "session_id": session_id}
        else:
        # All questions answered, proceed to diagnosis
            summary = ", ".join([f"{k}: {v}" for k, v in collected_data.items()])
            # Include user data in the medical query context
            diagnosis = await self.process_medical_query(summary, session_id, user_data=collected_data)
            return {"diagnosis": diagnosis, "session_id": session_id}

    async def process_medical_query(self, message: str, session_id: str = "default", user_data: Dict = None) -> dict:
        try:
            logger.info(f"Processing medical query: {message[:50]}... for session: {session_id}")

            rag_context = self.rag_service.get_context_for_query(message)
            
            # Add user_data to the context if provided
            if user_data:
                rag_context['user_data'] = user_data
            
            # Prioritize Groq service over HuggingFace to avoid token issues
            if settings.groq_api_key and settings.groq_api_key.strip():
                ai_service = self.groq_service
                logger.info("Using Groq service as primary AI service")
            elif settings.huggingface_api_token and settings.huggingface_api_token.strip():
                ai_service = self.hf_service
                logger.info("Using HuggingFace service as primary AI service")
            else:
                logger.error("No valid API tokens available")
                raise ValueError("No valid API tokens configured")

            result = await ai_service.generate_medical_response(message, rag_context)

            # Fallback to the other service if the primary fails
            if not result["success"] and ("authentication error" in result["error"].lower() or "invalid" in result["error"].lower()):
                logger.info(f"Primary service failed: {result['error']}. Attempting fallback...")
                
                # Try the other service
                fallback_service = self.hf_service if ai_service == self.groq_service else self.groq_service
                if (fallback_service == self.groq_service and settings.groq_api_key and settings.groq_api_key.strip()) or \
                   (fallback_service == self.hf_service and settings.huggingface_api_token and settings.huggingface_api_token.strip()):
                    logger.info(f"Falling back to {'Groq' if fallback_service == self.groq_service else 'HuggingFace'} service")
                    result = await fallback_service.generate_medical_response(message, rag_context)

            if result["success"]:
                self.context_service.update_context(session_id, message, result["response"], rag_context)
                urgency_level = "emergency" if rag_context["emergency_detected"] else "normal"
                return {
                    "success": True,
                    "response": f"{result['response']}\n\n{settings.medical_disclaimer}",
                    "urgency_level": urgency_level,
                    "emergency": rag_context["emergency_detected"],
                    "possible_conditions": rag_context["possible_conditions"],
                    "session_id": session_id,
                    "confidence": result.get("confidence", 0.8),
                    "explanation": result.get("explanation", "N/A"),
                    "disclaimer": settings.medical_disclaimer
                }
            else:
                logger.error(f"AI service error: {result['error']}")
                return {
                    "success": False,
                    "error": result["error"],
                    "response": f"Failed to process query: {result['error']}\n\n{settings.medical_disclaimer}",
                    "urgency_level": "normal",
                    "emergency": False,
                    "possible_conditions": [],
                    "session_id": session_id,
                    "confidence": 0.0,
                    "explanation": result.get("explanation", "Failed to process due to AI service error.")
                }
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"Error processing query: {str(e)}\n\n{settings.medical_disclaimer}",
                "urgency_level": "normal",
                "emergency": False,
                "possible_conditions": [],
                "session_id": session_id,
                "confidence": 0.0,
                "explanation": f"Failed to process due to: {str(e)}"
            }
        
medical_chatbot = MedicalChatbot()