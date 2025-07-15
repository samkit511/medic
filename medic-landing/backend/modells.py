from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class UrgencyLevel(str, Enum):
    """Medical urgency levels"""
    EMERGENCY = "emergency"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"

class FileType(str, Enum):
    """Supported file types"""
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    JPG = "jpg"
    PNG = "png"

class AnalysisType(str, Enum):
    """Types of medical analysis"""
    SYMPTOM_CHECK = "symptom_check"
    DOCUMENT_ANALYSIS = "document_analysis"
    GENERAL_QUERY = "general_query"

class ChatRequest(BaseModel):
    """Request model for chat/symptom analysis"""
    message: str = Field(..., min_length=5, max_length=2000, description="User's message or symptoms")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation tracking")
    analysis_type: AnalysisType = Field(AnalysisType.SYMPTOM_CHECK, description="Type of analysis requested")
    include_context: bool = Field(True, description="Whether to include conversation context")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or just whitespace')
        return v.strip()

class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    session_id: str = Field(..., description="Session identifier")
    file_name: str = Field(..., description="Name of uploaded file")
    file_type: FileType = Field(..., description="Type of uploaded file")
    analysis_question: Optional[str] = Field(None, description="Specific question about the document")

class MedicalAnalysis(BaseModel):
    """Structured medical analysis"""
    possible_conditions: List[str] = Field(default_factory=list, description="List of possible medical conditions")
    recommendations: List[str] = Field(default_factory=list, description="Medical recommendations")
    when_to_seek_help: str = Field("", description="When to seek medical attention")
    self_care_tips: List[str] = Field(default_factory=list, description="Self-care recommendations")
    red_flags: List[str] = Field(default_factory=list, description="Warning symptoms to watch for")

class ChatResponse(BaseModel):
    """Response model for chat/symptom analysis"""
    response: str = Field(..., description="AI-generated medical response")
    urgency_level: UrgencyLevel = Field(UrgencyLevel.MODERATE, description="Urgency level of symptoms")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in analysis (0-1)")
    is_emergency: bool = Field(False, description="Whether emergency care is needed")
    medical_analysis: Optional[MedicalAnalysis] = Field(None, description="Structured medical analysis")
    sources: List[str] = Field(default_factory=list, description="Knowledge sources used")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    disclaimer: str = Field("", description="Medical disclaimer")
    
    @validator('confidence_score')
    def validate_confidence(cls, v):
        return round(v, 2)

class DocumentAnalysis(BaseModel):
    """Structured document analysis"""
    document_type: str = Field("", description="Type of medical document detected")
    key_findings: List[str] = Field(default_factory=list, description="Key medical findings from document")
    extracted_data: Dict[str, Any] = Field(default_factory=dict, description="Structured data extracted")
    summary: str = Field("", description="Document summary")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations based on document")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    success: bool = Field(..., description="Whether upload was successful")
    file_id: str = Field("", description="Unique identifier for uploaded file")
    file_name: str = Field("", description="Name of uploaded file")
    file_size: int = Field(0, description="File size in bytes")
    processing_time: float = Field(0.0, description="Time taken to process document")
    document_analysis: Optional[DocumentAnalysis] = Field(None, description="Analysis of uploaded document")
    session_id: str = Field("", description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    message: str = Field("", description="Success or error message")

class ConversationTurn(BaseModel):
    """Individual conversation turn"""
    user_message: str = Field(..., description="User's message")
    ai_response: str = Field(..., description="AI's response")
    urgency_level: UrgencyLevel = Field(UrgencyLevel.MODERATE, description="Urgency level")
    timestamp: datetime = Field(default_factory=datetime.now, description="Turn timestamp")

class SessionContext(BaseModel):
    """Session context for conversation tracking and clinical data collection"""
    session_id: str = Field(..., description="Unique session identifier")
    conversation_history: List[ConversationTurn] = Field(default_factory=list, description="Conversation history")
    uploaded_documents: List[str] = Field(default_factory=list, description="List of uploaded document IDs")
    medical_summary: str = Field("", description="Summary of medical discussion")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")

    name: Optional[str] = Field(None, description="Patient's name")
    age: Optional[int] = Field(None, description="Patient's age")
    gender: Optional[str] = Field(None, description="Patient's gender")
    chief_complaint: Optional[str] = Field(None, description="Chief complaint")
    symptom_description: Optional[str] = Field(None, description="Description of main symptom")
    onset: Optional[str] = Field(None, description="When symptoms started")
    duration: Optional[str] = Field(None, description="Duration of symptoms")
    location: Optional[str] = Field(None, description="Location of problem")
    character: Optional[str] = Field(None, description="Nature of symptom (sharp, dull, throbbing, etc.)")
    severity: Optional[str] = Field(None, description="Symptom severity (1-10)")
    progression: Optional[str] = Field(None, description="Symptom progression (better, worse, same)")
    associated_symptoms: Optional[str] = Field(None, description="Other symptoms present")
    aggravating_factors: Optional[str] = Field(None, description="Aggravating or relieving factors")
    diet: Optional[str] = Field(None, description="Recent diet or dietary changes")
    recent_meals: Optional[str] = Field(None, description="Meals before symptom onset")
    hydration: Optional[str] = Field(None, description="Water intake")
    sleep_quality: Optional[str] = Field(None, description="Sleep quality")
    physical_activity: Optional[str] = Field(None, description="Physical activity")
    substance_use: Optional[str] = Field(None, description="Substance use (smoking, alcohol, drugs)")
    medical_history: Optional[str] = Field(None, description="Past medical history")
    medications: Optional[str] = Field(None, description="Current medications or supplements")
    allergies: Optional[str] = Field(None, description="Known allergies")
    family_history: Optional[str] = Field(None, description="Family history of disease")
    occupation: Optional[str] = Field(None, description="Occupation")
    recent_travel: Optional[str] = Field(None, description="Recent travel")
    exposures: Optional[str] = Field(None, description="Exposure to sick contacts or environmental risks")


class ErrorDetail(BaseModel):
    """Detailed error information"""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_type: str = Field(..., description="Type of error (validation, api, processing)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class ErrorResponse(BaseModel):
    """Standardized error response"""
    success: bool = Field(False, description="Always false for errors")
    error: ErrorDetail = Field(..., description="Error details")
    session_id: Optional[str] = Field(None, description="Session identifier if available")
    suggested_action: str = Field("", description="Suggested action for user")

class MedicalSource(BaseModel):
    """Medical knowledge source"""
    source_id: str = Field(..., description="Unique source identifier")
    source_type: str = Field(..., description="Type of source (textbook, guidelines, etc.)")
    title: str = Field("", description="Source title")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in source")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance to query")

class APIUsage(BaseModel):
    """API usage tracking"""
    api_provider: str = Field(..., description="API provider (groq, huggingface)")
    requests_made: int = Field(0, description="Number of requests made")
    requests_remaining: Optional[int] = Field(None, description="Requests remaining (if known)")
    reset_time: Optional[datetime] = Field(None, description="When rate limit resets")

def create_error_response(error_code: str, error_message: str, error_type: str = "general", session_id: Optional[str] = None) -> ErrorResponse:
    """Helper function to create standardized error responses"""
    return ErrorResponse(
        error=ErrorDetail(
            error_code=error_code,
            error_message=error_message,
            error_type=error_type
        ),
        session_id=session_id,
        suggested_action="Please try again or contact support if the problem persists."
    )

def create_success_response(message: str, urgency: UrgencyLevel = UrgencyLevel.MODERATE, session_id: str = "default") -> ChatResponse:
    """Helper function to create standardized success responses"""
    return ChatResponse(
        response=message,
        urgency_level=urgency,
        session_id=session_id,
        disclaimer="This analysis is for informational purposes only. Always consult healthcare professionals for medical advice."
    )

if __name__ == "__main__":
    print("🏥 Clinical Diagnostics Models Test")
    
    test_request = ChatRequest(
        message="I have a headache and fever",
        session_id="test123"
    )
    print(f"✅ Chat Request: {test_request.message}")
    
    test_response = create_success_response(
        message="Based on your symptoms, you may have a viral infection.",
        urgency=UrgencyLevel.LOW,
        session_id="test123"
    )
    print(f"✅ Chat Response: {test_response.urgency_level}")
    
    test_error = create_error_response(
        error_code="VALIDATION_ERROR",
        error_message="Message is too short",
        session_id="test123"
    )
    print(f"✅ Error Response: {test_error.error.error_code}")
    
    print("All models validated successfully!")
