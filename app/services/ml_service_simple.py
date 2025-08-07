"""
Simplified ML Service for development - without heavy ML dependencies
"""

import logging
from typing import Dict, List, Any, Optional
from PIL import Image
import io

logger = logging.getLogger(__name__)

class MLService:
    """
    Simplified ML Service for development testing
    """
    
    def __init__(self):
        self.initialized = True
        # Mock model attributes for compatibility
        self.yolo_model = "mock_yolo_model"
        self.clip_model = "mock_clip_model"
        self.device = "cpu"
        self.toy_categories = ["Action Figures", "Dolls", "Building Blocks", "Educational", "Electronic"]
        self.brands_list = ["LEGO", "Barbie", "Hot Wheels", "Fisher-Price", "Hasbro"]
        self.brand_patterns = {
            "LEGO": ["lego", "brick", "building"],
            "Barbie": ["barbie", "doll", "mattel"],
            "Hot Wheels": ["hot wheels", "matchbox", "car"],
            "Fisher-Price": ["fisher-price", "little people"],
            "Hasbro": ["hasbro", "transformers", "nerf"]
        }
        logger.info("✅ MLService initialized (development mode)")
    
    async def analyze_toy(self, image_data: bytes) -> Dict[str, Any]:
        """
        Mock toy analysis for development
        """
        try:
            # Validate image
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            
            # Mock analysis results
            result = {
                "toy_name": "Action Figure",
                "toy_type": "Action Figure", 
                "brand": "Generic Toy Co",
                "category": "action_figure",
                "condition_score": 8.5,
                "confidence": 0.95,
                "condition": {
                    "score": 8.5,
                    "description": "Very Good",
                    "defects": ["Minor wear on paint"]
                },
                "detections": [
                    {
                        "class": "toy",
                        "confidence": 0.95,
                        "bbox": [50, 50, width-50, height-50]
                    }
                ],
                "pricing": {
                    "estimated_value": 25.99,
                    "price_range": [15.99, 35.99],
                    "market_confidence": 0.8
                },
                "metadata": {
                    "image_size": f"{width}x{height}",
                    "analysis_time": 0.5
                }
            }
            
            logger.info(f"✅ Mock analysis completed for {width}x{height} image")
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for ML service
        """
        return {
            "status": "healthy",
            "mode": "development",
            "models_loaded": ["mock_yolo", "mock_clip"],
            "initialized": self.initialized
        }

    async def cleanup(self):
        """
        Cleanup resources (for compatibility)
        """
        logger.info("MLService cleanup completed (development mode)")

    async def initialize_models(self):
        """
        Initialize models (for compatibility)
        """
        logger.info("MLService models initialized (development mode)")

    async def analyze_toy_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze toy image (alias for analyze_toy)
        """
        return await self.analyze_toy(image_data)

# Global ML service instance
_ml_service = None

def get_ml_service() -> MLService:
    """
    Get or create ML service instance
    """
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service

async def get_ml_orchestrator():
    """
    Async alias for compatibility
    """
    return get_ml_service()