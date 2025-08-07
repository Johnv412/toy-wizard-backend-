"""
Configuration settings for ToyResaleWizard API
"""

import os
import logging
from typing import List
from pydantic_settings import BaseSettings

# Configure logging for monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Performance monitoring logger
performance_logger = logging.getLogger('performance')
performance_logger.setLevel(logging.INFO)

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ToyResaleWizard"
    
    # Database
    DATABASE_URL: str = "sqlite:///./toyresale.db"
    
    # Redis (for caching and task queue)
    REDIS_URL: str = "redis://localhost:6379"
    
    # ML Models
    YOLO_MODEL_PATH: str = "./models/yolov8n.pt"
    CLIP_MODEL_NAME: str = "ViT-B/32"
    
    # External APIs
    EBAY_APP_ID: str = ""
    AMAZON_API_KEY: str = ""
    
    # CORS - Secure configuration
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8081"
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8081",  # React Native Metro
        "exp://192.168.1.100:19000",  # Expo development
    ]
    
    # Security settings - MUST be set via environment variables in production
    SECRET_KEY: str = ""
    ALLOWED_HOSTS: str = "localhost,127.0.0.1"
    
    # Application settings
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    # Validate critical security settings
    def __post_init__(self):
        if not self.SECRET_KEY:
            if os.getenv("ENVIRONMENT") == "production":
                raise ValueError("SECRET_KEY must be set in production environment")
            else:
                # Generate a random key for development
                import secrets
                self.SECRET_KEY = secrets.token_urlsafe(32)
                print(f"⚠️  WARNING: Using auto-generated SECRET_KEY for development: {self.SECRET_KEY}")
        
        if len(self.SECRET_KEY) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        
        # Validate API keys in production
        if os.getenv("ENVIRONMENT") == "production":
            if not self.EBAY_APP_ID:
                print("⚠️  WARNING: EBAY_APP_ID not set - eBay pricing will be unavailable")
            if not self.AMAZON_API_KEY:
                print("⚠️  WARNING: AMAZON_API_KEY not set - Amazon pricing will be unavailable")
    
    RATE_LIMIT_ENABLED: bool = True
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    
    # File upload settings
    UPLOAD_DIR: str = "./uploads"
    TEMP_DIR: str = "./temp"
    
    # ML Settings
    CONFIDENCE_THRESHOLD: float = 0.6
    MAX_DETECTIONS: int = 10
    
    # Pricing Settings
    CACHE_EXPIRY_HOURS: int = 24
    MAX_PRICE_HISTORY: int = 100
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    model_config = {
        "env_file": ".env",
        "extra": "allow"
    }

settings = Settings()