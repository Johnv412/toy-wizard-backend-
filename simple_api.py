from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import random
from typing import Dict, Any
import os

app = FastAPI(title="Toy Wizard API")

# Enable CORS for all origins - specifically for PWA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=False,  # Set to False for wildcard origins
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "Toy Wizard API is running!"}

@app.get("/api/health/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "ToyResaleWizard API",
        "version": "1.0.0"
    }

@app.post("/api/analysis/analyze-toy")
async def analyze_toy(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Analyze a toy image and return pricing information."""
    
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read file size
    contents = await file.read()
    file_size = len(contents)
    
    # Simulate AI analysis with realistic toy data
    toy_categories = ["action_figure", "doll", "lego", "board_game", "stuffed_animal", "vehicle"]
    toy_names = [
        "Vintage Star Wars Figure", "Classic Barbie Doll", "LEGO Creator Set",
        "Monopoly Board Game", "Teddy Bear", "Hot Wheels Car", "Transformer Robot",
        "Pokemon Plush", "Nerf Blaster", "Marvel Action Figure"
    ]
    
    # Generate realistic pricing based on "condition"
    base_price = random.uniform(10, 150)
    condition_score = random.uniform(6, 10)
    estimated_price = base_price * (condition_score / 10)
    
    return {
        "status": "success",
        "data": {
            "toy_name": random.choice(toy_names),
            "category": random.choice(toy_categories),
            "estimated_price": round(estimated_price, 2),
            "condition_score": round(condition_score, 1),
            "confidence": round(random.uniform(0.85, 0.99), 2),
            "file_size": file_size,
            "timestamp": datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)