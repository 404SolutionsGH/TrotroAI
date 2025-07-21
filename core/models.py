"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    session_id: str = "default"
    context: Optional[Dict[str, Any]] = {}


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    confidence: float
    source: str = "ai"
    context: Dict[str, Any] = {}
    suggestions: List[str] = []


class HealthCheck(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


class TrainingRequest(BaseModel):
    """Request model for training AI models"""
    training_data: List[Dict[str, Any]]
    model_type: str = "general"


class PredictionRequest(BaseModel):
    """Request model for demand prediction"""
    route_id: int
    datetime: str
    factors: Optional[Dict[str, Any]] = {}


class RouteOptimizationRequest(BaseModel):
    """Request model for route optimization"""
    origin: str
    destination: str
    preferences: Optional[Dict[str, Any]] = {}
    constraints: Optional[Dict[str, Any]] = {}
