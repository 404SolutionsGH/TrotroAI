#!/usr/bin/env python3
"""
Trotro AI Service - FastAPI Application
A microservice for AI-powered transportation queries and analysis
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import logging
import os
from datetime import datetime
from contextlib import asynccontextmanager

from core.ai_engine import TrotroAIEngine
from core.database import DatabaseManager
from core.models import (
    ChatRequest, 
    ChatResponse, 
    HealthCheck, 
    TrainingRequest,
    PredictionRequest,
    RouteOptimizationRequest
)
from core.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Global variables
ai_engine: Optional[TrotroAIEngine] = None
db_manager: Optional[DatabaseManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global ai_engine, db_manager
    
    # Startup
    logger.info("🚀 Starting Trotro AI Service...")
    
    try:
        # Initialize database connection
        db_manager = DatabaseManager(settings.database_url)
        await db_manager.connect()
        logger.info("✅ Database connected")
        
        # Initialize AI engine
        ai_engine = TrotroAIEngine(
            deepseek_api_key=settings.deepseek_api_key,
            model_cache_dir=settings.model_cache_dir,
            embedding_model=settings.embedding_model
        )
        await ai_engine.initialize()
        logger.info("✅ AI Engine initialized")
        
        # Load training data from database
        await ai_engine.load_transport_data(db_manager)
        logger.info("✅ Transport data loaded")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Trotro AI Service...")
    if db_manager:
        await db_manager.disconnect()
    if ai_engine:
        await ai_engine.cleanup()


# Initialize FastAPI app
app = FastAPI(
    title="Trotro AI Service",
    description="AI-powered transportation query and optimization service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint - health check"""
    return HealthCheck(
        status="healthy",
        service="trotro-ai-service",
        version="1.0.0",
        timestamp=datetime.utcnow()
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check"""
    if not ai_engine or not db_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return HealthCheck(
        status="healthy",
        service="trotro-ai-service",
        version="1.0.0",
        timestamp=datetime.utcnow(),
        details={
            "ai_engine": ai_engine.is_ready(),
            "database": db_manager.is_connected(),
            "models_loaded": ai_engine.get_model_info()
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for AI queries"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not ready")
    
    try:
        response = await ai_engine.process_query(
            query=request.message,
            session_id=request.session_id,
            user_context=request.context
        )
        
        return ChatResponse(
            response=response["answer"],
            confidence=response.get("confidence", 0.0),
            source=response.get("source", "ai"),
            context=response.get("context", {}),
            suggestions=response.get("suggestions", [])
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/demand")
async def predict_demand(request: PredictionRequest):
    """Predict transportation demand"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not ready")
    
    try:
        prediction = await ai_engine.predict_demand(
            route_id=request.route_id,
            datetime=request.datetime,
            additional_factors=request.factors
        )
        
        return {
            "route_id": request.route_id,
            "predicted_demand": prediction["demand"],
            "confidence": prediction["confidence"],
            "factors": prediction["factors"],
            "recommendation": prediction["recommendation"]
        }
        
    except Exception as e:
        logger.error(f"Demand prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/route")
async def optimize_route(request: RouteOptimizationRequest):
    """Optimize route planning"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not ready")
    
    try:
        optimization = await ai_engine.optimize_route(
            origin=request.origin,
            destination=request.destination,
            preferences=request.preferences,
            constraints=request.constraints
        )
        
        return {
            "origin": request.origin,
            "destination": request.destination,
            "optimal_route": optimization["route"],
            "estimated_time": optimization["time"],
            "estimated_cost": optimization["cost"],
            "alternatives": optimization["alternatives"]
        }
        
    except Exception as e:
        logger.error(f"Route optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train AI model with new data"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not ready")
    
    try:
        # Start training in background
        background_tasks.add_task(
            ai_engine.train_model,
            training_data=request.training_data,
            model_type=request.model_type
        )
        
        return {
            "message": "Training started",
            "status": "in_progress",
            "data_points": len(request.training_data)
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status")
async def get_model_status():
    """Get status of all AI models"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not ready")
    
    return {
        "models": ai_engine.get_model_status(),
        "last_updated": ai_engine.get_last_update(),
        "training_progress": ai_engine.get_training_progress()
    }


@app.get("/analytics")
async def get_analytics():
    """Get AI service analytics"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not ready")
    
    return await ai_engine.get_analytics()


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: dict):
    """WhatsApp webhook for chatbot integration"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not ready")
    
    try:
        # Process WhatsApp message
        response = await ai_engine.process_whatsapp_message(request)
        return response
        
    except Exception as e:
        logger.error(f"WhatsApp webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )
