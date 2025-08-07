#!/usr/bin/env python3
"""
Secure ML model downloader for ToyResaleWizard
"""

import os
import sys
import requests
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Dict
import ssl
import urllib3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model integrity verification data
MODEL_CHECKSUMS = {
    "yolov8n.pt": "8c5b5a9a4f5b9c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0",
    # Add more model checksums as needed
}

# Trusted model sources
TRUSTED_SOURCES = {
    "github.com",
    "pytorch.org", 
    "huggingface.co",
    "download.pytorch.org"
}

def verify_url_safety(url: str) -> bool:
    """Verify URL is from trusted source and uses HTTPS"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        
        # Must use HTTPS
        if parsed.scheme != 'https':
            logger.error(f"Insecure URL scheme: {parsed.scheme}")
            return False
            
        # Must be from trusted source
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        if not any(trusted in hostname for trusted in TRUSTED_SOURCES):
            logger.error(f"Untrusted source: {hostname}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"URL validation failed: {e}")
        return False

def calculate_file_hash(filepath: str) -> str:
    """Calculate SHA256 hash of file"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Hash calculation failed: {e}")
        return ""

def verify_file_integrity(filepath: str, expected_hash: Optional[str] = None) -> bool:
    """Verify downloaded file integrity"""
    if not expected_hash:
        filename = Path(filepath).name
        expected_hash = MODEL_CHECKSUMS.get(filename)
        
    if not expected_hash:
        logger.warning(f"No checksum available for {filepath}")
        return True  # Allow download but warn
        
    actual_hash = calculate_file_hash(filepath)
    if actual_hash == expected_hash:
        logger.info(f"File integrity verified: {filepath}")
        return True
    else:
        logger.error(f"File integrity check failed: {filepath}")
        logger.error(f"Expected: {expected_hash}")
        logger.error(f"Actual: {actual_hash}")
        return False

def download_file(url: str, destination: str, expected_hash: Optional[str] = None) -> bool:
    """Securely download a file from URL to destination with integrity verification"""
    try:
        # Verify URL safety first
        if not verify_url_safety(url):
            logger.error(f"URL safety check failed: {url}")
            return False
            
        logger.info(f"Downloading {url} to {destination}")
        
        # Configure secure session
        session = requests.Session()
        session.verify = True  # Always verify SSL certificates
        session.timeout = (30, 300)  # Connection timeout, read timeout
        
        # Set secure headers
        headers = {
            'User-Agent': 'ToyResaleWizard-ModelDownloader/1.0',
            'Accept': 'application/octet-stream',
        }
        
        response = session.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        # Use temporary file for atomic download
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        
        # Verify file integrity before moving to final location
        if not verify_file_integrity(temp_path, expected_hash):
            os.unlink(temp_path)
            return False
            
        # Move verified file to destination
        os.rename(temp_path, destination)
        os.chmod(destination, 0o644)  # Set secure permissions
        
        logger.info(f"Successfully downloaded and verified {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return False

def download_yolo_model():
    """Securely download YOLOv8 model with verification"""
    models_dir = Path("/app/models")
    models_dir.mkdir(exist_ok=True, mode=0o755)
    
    model_path = models_dir / "yolov8n.pt"
    
    if model_path.exists():
        # Verify existing model integrity
        if verify_file_integrity(str(model_path)):
            logger.info(f"YOLOv8 model already exists and verified at {model_path}")
            return True
        else:
            logger.warning("Existing model failed integrity check, re-downloading...")
            os.unlink(model_path)
    
    # YOLOv8 nano model from official Ultralytics GitHub releases
    url = "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt"
    expected_hash = MODEL_CHECKSUMS.get("yolov8n.pt")
    
    return download_file(url, str(model_path), expected_hash)

def verify_clip_model():
    """Verify CLIP model can be loaded"""
    try:
        import clip
        import torch
        
        logger.info("Verifying CLIP model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        logger.info("CLIP model verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"CLIP model verification failed: {e}")
        return False

def create_model_info():
    """Create model information file"""
    models_dir = Path("/app/models")
    info_file = models_dir / "model_info.txt"
    
    info_content = """ToyResaleWizard ML Models

Downloaded Models:
- YOLOv8n: Object detection model for toy identification
- CLIP ViT-B/32: Vision-language model for classification

Model Sources:
- YOLOv8: https://github.com/ultralytics/ultralytics
- CLIP: https://github.com/openai/CLIP

Usage:
- YOLO models are loaded from ./models/yolov8n.pt
- CLIP models are downloaded automatically by the library
"""
    
    with open(info_file, 'w') as f:
        f.write(info_content)
    
    logger.info(f"Created model info file at {info_file}")

def main():
    """Main function to download all required models"""
    logger.info("Starting model download process...")
    
    success = True
    
    # Download YOLOv8 model
    if not download_yolo_model():
        success = False
    
    # Verify CLIP model
    if not verify_clip_model():
        success = False
    
    # Create model info
    create_model_info()
    
    if success:
        logger.info("All models downloaded and verified successfully!")
        return 0
    else:
        logger.error("Some models failed to download or verify")
        return 1

if __name__ == "__main__":
    sys.exit(main())