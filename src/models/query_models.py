from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class QueryType(str, Enum):
    EMAIL = "email"
    RESEARCH = "research"
    ACADEMIC = "academic"
    REDIRECT = "redirect"
    GENERAL = "general"

class BaseQuery(BaseModel):
    """Base model for all query types"""
    query_type: QueryType = Field(..., description="Type of query to be processed")
    user_message: str = Field(..., description="Original user message")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context for the query")

class EmailQuery(BaseQuery):
    """Model for email composition queries"""
    recipient_type: Optional[str] = Field(None, description="Type of recipient (e.g., professor, advisor)")
    purpose: Optional[str] = Field(None, description="Purpose of the email")
    details: Optional[List[str]] = Field(default=[], description="Additional details to include")
    tone: Optional[str] = Field(default="professional", description="Desired tone of the email")

class ResearchQuery(BaseQuery):
    """Model for research paper queries"""
    paper_topic: Optional[str] = Field(None, description="Main topic of the research paper")
    academic_level: Optional[str] = Field(None, description="Academic level (undergraduate, graduate, doctoral)")
    paper_length: Optional[str] = Field(None, description="Required length of the paper")
    requirements: Optional[List[str]] = Field(default=[], description="Specific requirements for the paper")

class AcademicQuery(BaseQuery):
    """Model for academic concept queries"""
    subject_area: Optional[str] = Field(None, description="Subject area of the concept")
    concept: Optional[str] = Field(None, description="Specific concept to explain")
    difficulty_level: Optional[str] = Field(default="intermediate", description="Desired difficulty level of explanation")
    prerequisites: Optional[List[str]] = Field(default=[], description="Prerequisites for understanding the concept")

class RedirectQuery(BaseQuery):
    """Model for resource redirection queries"""
    resource_type: Optional[str] = Field(None, description="Type of resource being sought")
    specific_need: Optional[str] = Field(None, description="Specific information needed")
    department: Optional[str] = Field(None, description="Relevant department or office")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context for redirection")

class QueryResponse(BaseModel):
    """Model for agent responses"""
    success: bool = Field(..., description="Whether the query was processed successfully")
    content: str = Field(..., description="The response content")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata about the response")
    error: Optional[str] = Field(None, description="Error message if processing failed") 