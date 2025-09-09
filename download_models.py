#!/usr/bin/env python3
"""
ML Model Download Script
Downloads YOLOv8 and CLIP models with versioning
"""

import os
import logging
import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Model configurations
MODELS_DIR = Path("app/ml_models")
YOLO_MODEL_NAME = "yolov8n"
CLIP_MODEL_NAME = "ViT-B/32"

def create_versioned_dir(base_name: str) -> Path:
    """Create a versioned directory for the model"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version = 1

    while True:
        dir_name = f"{base_name}_v{version}_{timestamp}"
        model_dir = MODELS_DIR / dir_name
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            return model_dir
        version += 1

def download_yolo_model(model_dir: Path):
    """Download YOLOv8 model"""
    try:
        from ultralytics import YOLO
        logger.info("Downloading YOLOv8n model...")

        # Create YOLO model instance (this downloads the model)
        model = YOLO(f"{YOLO_MODEL_NAME}.pt")

        # Save the model to our directory
        model_path = model_dir / f"{YOLO_MODEL_NAME}.pt"
        model.save(str(model_path))

        logger.info(f"✅ YOLOv8n model saved to {model_path}")
        return True

    except ImportError:
        logger.warning("ultralytics not installed, downloading manually...")
        return download_yolo_manual(model_dir)
    except Exception as e:
        logger.error(f"❌ Failed to download YOLOv8 model: {str(e)}")
        return False

def download_yolo_manual(model_dir: Path):
    """Manual download of YOLOv8 model"""
    try:
        import requests
        from urllib.parse import urljoin

        # YOLOv8 model URL (from ultralytics)
        base_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        model_file = f"{YOLO_MODEL_NAME}.pt"
        url = urljoin(base_url, model_file)

        logger.info(f"Downloading YOLOv8n from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        model_path = model_dir / model_file
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"✅ YOLOv8n model downloaded to {model_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download YOLOv8 model manually: {str(e)}")
        return False

def download_clip_direct(model_dir: Path):
    """Download CLIP model files directly from Hugging Face"""
    try:
        import requests
        import json

        # CLIP model files from Hugging Face
        base_url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main"
        files_to_download = [
            "pytorch_model.bin",
            "config.json",
            "preprocessor_config.json"
        ]

        for file_name in files_to_download:
            url = f"{base_url}/{file_name}"
            logger.info(f"Downloading {file_name}...")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            file_path = model_dir / file_name
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ Downloaded {file_name}")

        # Save model config
        config_path = model_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "model_type": "clip",
                "model_name": "ViT-B-32",
                "source": "openai/clip-vit-base-patch32"
            }, f)

        logger.info(f"✅ CLIP model downloaded to {model_dir}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download CLIP model directly: {str(e)}")
        return False

def download_clip_model(model_dir: Path):
    """Download CLIP model"""
    try:
        # Try simple download first to avoid library issues
        logger.info("Downloading CLIP ViT-B/32 model files directly...")
        return download_clip_direct(model_dir)
    except Exception as e:
        logger.error(f"❌ Failed to download CLIP model: {str(e)}")
        return False

def download_clip_transformers(model_dir: Path):
    """Download CLIP using transformers"""
    try:
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Downloading CLIP ViT-B/32 using transformers...")

        # Load model and processor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Save model
        model_path = model_dir / "clip_vit_b_32"
        model.save_pretrained(model_path)
        processor.save_pretrained(model_path)

        logger.info(f"✅ CLIP model saved to {model_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download CLIP model with transformers: {str(e)}")
        return False

def main():
    """Main download function"""
    logger.info("🚀 Starting ML model download process...")

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0

    # Download YOLOv8
    logger.info("📥 Downloading YOLOv8 model...")
    yolo_dir = create_versioned_dir(YOLO_MODEL_NAME)
    if download_yolo_model(yolo_dir):
        success_count += 1
    else:
        logger.error("Failed to download YOLOv8 model")

    # Download CLIP
    logger.info("📥 Downloading CLIP model...")
    clip_dir = create_versioned_dir("clip_vit_b_32")
    if download_clip_model(clip_dir):
        success_count += 1
    else:
        logger.error("Failed to download CLIP model")

    if success_count == 2:
        logger.info("🎉 All models downloaded successfully!")
        logger.info(f"📁 Models saved in {MODELS_DIR}")
        return 0
    else:
        logger.error(f"💥 Download failed for {2 - success_count} model(s)")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)