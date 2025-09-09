"""
AI Service for real toy analysis using OpenAI Vision API
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)

class AIService:
    """Service for AI-powered toy analysis using OpenAI Vision"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def analyze_toy_image(self, image_base64: str) -> Dict[str, Any]:
        """
        Analyze a toy image using OpenAI Vision API

        Args:
            image_base64: Base64 encoded image data

        Returns:
            Dict containing toy analysis results
        """
        if not self.api_key or self.api_key == "your-openai-api-key-here":
            logger.warning("OpenAI API key not configured, returning mock data")
            return self._get_mock_analysis()

        try:
            prompt = """
            Analyze this toy image and provide detailed information:
            1. Exact toy name and brand (be as specific as possible)
            2. Category (action_figure, doll, lego, board_game, stuffed_animal, vehicle, etc.)
            3. Estimated market value in USD (current resale price)
            4. Condition assessment (1-10 scale, where 10 is mint/new)
            5. Rarity level (common, uncommon, rare, very_rare, collectible)
            6. Year/model if identifiable
            7. Any notable features, damage, or missing parts

            Respond ONLY with valid JSON format matching this structure:
            {
                "toy_name": "string",
                "brand": "string",
                "category": "string",
                "estimated_price": number,
                "condition_score": number,
                "rarity": "string",
                "year": "string or null",
                "features": "string description",
                "confidence": number
            }
            """

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "gpt-4-vision-preview",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 500,
                        "temperature": 0.1
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    result = response.json()
                    ai_response = result["choices"][0]["message"]["content"]

                    # Parse the JSON response
                    try:
                        analysis_data = json.loads(ai_response.strip())
                        analysis_data["timestamp"] = datetime.utcnow().isoformat()
                        analysis_data["ai_provider"] = "openai"

                        logger.info(f"Successfully analyzed toy: {analysis_data.get('toy_name', 'Unknown')}")
                        return analysis_data

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse AI response as JSON: {ai_response}")
                        return self._get_mock_analysis()

                else:
                    logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                    return self._get_mock_analysis()

        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            return self._get_mock_analysis()

    def _get_mock_analysis(self) -> Dict[str, Any]:
        """Return mock analysis data when AI is not available"""
        return {
            "toy_name": "Demo Toy - Add OpenAI API Key for Real Analysis",
            "brand": "Demo Brand",
            "category": "demo",
            "estimated_price": 0.00,
            "condition_score": 0.0,
            "rarity": "unknown",
            "year": None,
            "features": "Demo analysis - configure OPENAI_API_KEY for real results",
            "confidence": 0.0,
            "timestamp": datetime.utcnow().isoformat(),
            "ai_provider": "mock"
        }

# Global AI service instance
ai_service = AIService()