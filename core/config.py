"""
Configuration management for Trotro AI Service
"""

import os
from pydantic import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    service_name: str = "trotro-ai-service"
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8001
    
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:password@localhost/trotro"
    )
    
    # API Keys
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Backend service
    backend_service_url: str = os.getenv(
        "BACKEND_SERVICE_URL", 
        "https://api.trotro.live"
    )
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Model configuration
    model_cache_dir: str = "./models"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_context_length: int = 512
    
    # Performance
    max_workers: int = 4
    timeout_seconds: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
