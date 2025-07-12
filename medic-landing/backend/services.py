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
        self.text_model = "meta-llama/Meta-Llama-3-8B-Instruct"  # Conversational model for text
        self.vision_model = settings.hf_medical_model  # Vision model for multimodal
        self.embedding_model = self._load_embedding_model()
        self.sentiment_analyzer = self._load_sentiment_analyzer()
        self.translator = GoogleTranslator()
        self.few_shot_examples = self._load_medical_examples()
        logger.info(f"HuggingFaceService initialized, client available: {bool(self.client)}")

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

{examples_text}
{context_text}
{possible_conditions}
{urgency_text}

Patient: {user_query}

Reasoning:
Step 1: Symptom Organization - [List symptoms, severity, history]
Step 2: Possible Conditions - [Select from: {possible_conditions if possible_conditions else 'provided medical knowledge'}]
Step 3: Recommendations - [Actionable advice]

Medical Advice:
- Analysis: [Detailed assessment]
- Recommendations: [Specific steps]
- Urgency: [{'emergency' if context and context['emergency_detected'] else 'normal'}]
- Confidence: [0-1 scale]

Always recommend consulting healthcare professionals."""
        logger.debug(f"CoT prompt created: {prompt[:100]}...")
        return prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.HTTPError)
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
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are a medical AI assistant."}, {"role": "user", "content": medical_prompt}],
                max_tokens=settings.max_response_tokens,
                temperature=settings.medical_temperature
            )
            refined_response = await self._refine_diagnosis(translated_prompt, response.choices[0].message.content, context.get("session_id", "default") if context else "default", context)
            
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

    def _translate_query(self, query: str, target_lang: str = "en") -> str:
        try:
            if any(ord(char) > 127 for char in query):
                translated = self.translator.translate(query, dest=target_lang)
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
        self.translator = GoogleTranslator()
        self.few_shot_examples = self._load_medical_examples()
        logger.info(f"GroqService initialized, client available: {bool(self.client)}")

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

    def _translate_query(self, query: str, target_lang: str = "en") -> str:
        try:
            if any(ord(char) > 127 for char in query):
                translated = self.translator.translate(query, dest=target_lang)
                logger.info(f"Translated query: {query[:50]}... -> {translated[:50]}...")
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
            prompt = f"""Patient input: {user_query}\nInitial response: {response}\nAsk a clarifying question to refine diagnosis (e.g., duration, severity).\nUrgency: {urgency}\nPossible Conditions: {', '.join(context['possible_conditions'])}"""
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
                return f"{response}\nClarification needed: {clarification.choices[0].message.content}\n\nUrgency: {urgency}"
            except Exception as e:
                logger.error(f"Clarification error: {e}")
                return f"{response}\n\nUrgency: {urgency}"
        return f"{response}\n\nUrgency: {urgency}"

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
        self.doc_service = DocumentProcessingService(self.rag_service, self.hf_service if settings.huggingface_api_token else self.groq_service)
        self.context_service = ContextService()
        logger.info("MedicalChatbot initialized")

    async def process_medical_query(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        try:
            logger.info(f"Processing medical query: {message[:50]}... for session: {session_id}")
            
            rag_context = self.rag_service.get_context_for_query(message)
            ai_service = self.hf_service if settings.huggingface_api_token else self.groq_service
            
            result = await ai_service.generate_medical_response(message, rag_context)
            
            # Fallback to GroqService if HuggingFace fails due to authentication
            if not result["success"] and "authentication error" in result["error"].lower():
                logger.info("Falling back to GroqService due to HuggingFace authentication failure")
                ai_service = self.groq_service
                result = await ai_service.generate_medical_response(message, rag_context)
            
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

if __name__ == "__main__":
    async def test_chatbot():
        chatbot = MedicalChatbot()
        query = "I have a headache and fever"
        result = await chatbot.process_medical_query(query, session_id="test123")
        print(f"Query: {query}")
        print(f"Response: {result['response']}")
        print(f"Urgency: {result['urgency_level']}")
        print(f"Possible Conditions: {result['possible_conditions']}")
        print(f"Confidence: {result['confidence']}")

    asyncio.run(test_chatbot())