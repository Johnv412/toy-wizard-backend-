"""
SAFE FIRST VERSION: MLOrchestrator with exact MLService compatibility
Pure delegation pattern with new parallel processing architecture
"""

import asyncio
import logging
import numpy as np
from PIL import Image
import io
import json
import gc
import psutil
import threading
from typing import Dict, List, Tuple, Optional
import torch
from ultralytics import YOLO
import clip
import cv2
from datetime import datetime
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings

logger = logging.getLogger(__name__)

class MLOrchestrator:
    """
    SAFE FIRST VERSION: MLOrchestrator with exact MLService method signatures
    
    Key Features:
    - Exact method compatibility with MLService
    - NEW: Parallel processing via ThreadPoolExecutor 
    - NEW: GPU warmup in __init__
    - ZERO business logic changes (pure delegation)
    """
    
    def __init__(self):
        """Initialize MLOrchestrator with exact MLService compatibility"""
        logger.info("Initializing SAFE MLOrchestrator...")
        
        # EXACT compatibility with MLService
        self.yolo_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Memory management (exact MLService compatibility)
        self._model_lock = threading.RLock()
        self._request_counter = 0
        self._max_requests_before_cleanup = 50
        self._last_cleanup = datetime.utcnow()
        
        # Memory monitoring (exact MLService compatibility)
        self._memory_threshold_mb = 2048  # 2GB threshold
        self._gpu_memory_threshold_mb = 1024  # 1GB GPU threshold
        
        # NEW: Parallel processing architecture
        self._thread_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="MLOrch")
        
        # NEW: GPU warmup in __init__
        self._warmup_gpu()
        
        # Toy categories mapping (exact compatibility)
        self.toy_categories = {
            'action_figure': ['action figure', 'superhero', 'robot', 'transformer'],
            'doll': ['doll', 'barbie', 'baby doll', 'fashion doll'],
            'vehicle': ['car', 'truck', 'train', 'airplane', 'boat'],
            'building_set': ['lego', 'blocks', 'construction', 'building'],
            'plush': ['teddy bear', 'stuffed animal', 'plush toy'],
            'board_game': ['board game', 'card game', 'puzzle'],
            'educational': ['learning toy', 'educational', 'science kit'],
            'outdoor': ['ball', 'bike', 'scooter', 'outdoor toy'],
            'electronic': ['electronic toy', 'robot', 'remote control'],
            'collectible': ['collectible', 'trading card', 'figurine']
        }
        
        # Brand recognition patterns (exact compatibility)
        self.brand_patterns = {
            'LEGO': ['lego', 'brick', 'minifigure'],
            'Barbie': ['barbie', 'mattel'],
            'Hot Wheels': ['hot wheels', 'die cast'],
            'Nerf': ['nerf', 'blaster'],
            'Pokemon': ['pokemon', 'pikachu'],
            'Star Wars': ['star wars', 'lightsaber'],
            'Marvel': ['marvel', 'spider-man', 'iron man'],
            'DC': ['batman', 'superman', 'justice league'],
            'Transformers': ['transformers', 'optimus prime'],
            'My Little Pony': ['my little pony', 'mlp']
        }
        
        logger.info(f"SAFE MLOrchestrator initialized on device: {self.device}")
    
    def _warmup_gpu(self):
        """NEW: GPU warmup in __init__ using dummy tensor"""
        try:
            logger.info("Warming up GPU with dummy tensor...")
            dummy = torch.rand(1, 3, 640, 640).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Force GPU computation
            if torch.cuda.is_available():
                dummy_result = dummy * 2.0
                torch.cuda.synchronize()
                del dummy_result
            
            del dummy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("GPU warmup completed successfully")
            
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}")
    
    async def initialize_models(self):
        """EXACT method signature from MLService - initialize ML models"""
        try:
            logger.info("Initializing ML models via MLOrchestrator...")
            
            # Load YOLO model for object detection
            try:
                self.yolo_model = YOLO(settings.YOLO_MODEL_PATH)
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}. Using fallback detection.")
                self.yolo_model = None
            
            # Load CLIP model for image-text matching
            try:
                self.clip_model, self.clip_preprocess = clip.load(
                    settings.CLIP_MODEL_NAME, 
                    device=self.device
                )
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load CLIP model: {e}. Using fallback classification.")
                self.clip_model = None
            
            logger.info("ML models initialization completed via MLOrchestrator")
            
        except Exception as e:
            logger.error(f"Error initializing ML models via MLOrchestrator: {e}")
            raise
    
    async def analyze_toy_image(self, image_data: bytes) -> Dict:
        """EXACT method signature from MLService - analyze toy image with NEW parallel processing"""
        request_id = self._request_counter
        self._request_counter += 1
        
        # Check memory before processing (exact compatibility)
        if not await self._check_memory_availability():
            await self._force_cleanup()
        
        try:
            logger.info(f"Starting SAFE parallel analysis request {request_id}")
            
            # Memory-conscious image preprocessing with optimization (exact compatibility)
            async with self._optimized_image_context(image_data) as processed_images:
                if processed_images is None:
                    return self._get_fallback_analysis()
                
                # NEW: Parallel execution using ThreadPoolExecutor instead of asyncio
                async with self._memory_context(f"parallel_analysis_{request_id}"):
                    start_time = asyncio.get_event_loop().time()
                    
                    # NEW: Submit tasks to ThreadPoolExecutor (max_workers=3)
                    loop = asyncio.get_event_loop()
                    
                    detection_future = loop.run_in_executor(
                        self._thread_pool,
                        self._detect_objects_sync,
                        processed_images.yolo_ready
                    )
                    classification_future = loop.run_in_executor(
                        self._thread_pool,
                        self._classify_toy_sync,
                        processed_images.clip_ready
                    )
                    condition_future = loop.run_in_executor(
                        self._thread_pool,
                        self._assess_condition_sync,
                        processed_images.cv_ready
                    )
                    
                    # Wait for all tasks to complete (same as original)
                    detection_result, classification_result, condition_result = \
                        await asyncio.gather(
                            detection_future,
                            classification_future, 
                            condition_future,
                            return_exceptions=True
                        )
                    
                    processing_time = asyncio.get_event_loop().time() - start_time
                    logger.info(f"SAFE parallel ML processing completed in {processing_time:.2f}s")
                    
                    # Handle any exceptions from parallel tasks (exact compatibility)
                    if isinstance(detection_result, Exception):
                        logger.warning(f"Detection failed: {detection_result}")
                        detection_result = {'objects': [], 'confidence': 0.0}
                    
                    if isinstance(classification_result, Exception):
                        logger.warning(f"Classification failed: {classification_result}")
                        classification_result = self._fallback_classification()
                        
                    if isinstance(condition_result, Exception):
                        logger.warning(f"Condition assessment failed: {condition_result}")
                        condition_result = {'score': 7.0, 'condition_text': 'Good'}
                
                # Immediate cleanup after parallel processing (exact compatibility)
                await self._cleanup_intermediate()
                
                # Combine results (EXACT same business logic)
                analysis = {
                    'toy_name': classification_result.get('name', 'Unknown Toy'),
                    'category': classification_result.get('category', 'Other'),
                    'brand': classification_result.get('brand'),
                    'condition_score': condition_result.get('score', 5.0),
                    'confidence': classification_result.get('confidence', 0.5),
                    'rarity_score': await self._calculate_rarity(classification_result),
                    'estimated_price': await self._estimate_price(classification_result, condition_result),
                    'detection_data': detection_result,
                    'condition_details': condition_result,
                    'timestamp': datetime.utcnow().isoformat(),
                    'request_id': request_id,
                    'processing_time': processing_time
                }
                
                # Periodic cleanup (exact compatibility)
                if self._request_counter % self._max_requests_before_cleanup == 0:
                    await self._periodic_cleanup()
                
                logger.info(f"Completed SAFE analysis request {request_id} in {processing_time:.2f}s")
                return analysis
                
        except Exception as e:
            logger.error(f"Error in SAFE analysis request {request_id}: {e}")
            await self._emergency_cleanup()
            return self._get_fallback_analysis()
        finally:
            # Ensure cleanup always happens (exact compatibility)
            await self._cleanup_intermediate()
    
    # NEW: Synchronous wrapper methods for ThreadPoolExecutor
    def _detect_objects_sync(self, image) -> Dict:
        """Synchronous wrapper for object detection"""
        try:
            if self.yolo_model is None:
                return {'objects': [], 'confidence': 0.0}
            
            # Convert to format expected by YOLO
            if isinstance(image, np.ndarray):
                results = self.yolo_model(image)
                
                # Extract detection results
                detected_objects = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            detected_objects.append({
                                'class': result.names[int(box.cls[0])],
                                'confidence': float(box.conf[0]),
                                'bbox': box.xyxy[0].tolist()
                            })
                
                return {
                    'objects': detected_objects,
                    'confidence': max([obj['confidence'] for obj in detected_objects]) if detected_objects else 0.0
                }
            else:
                return {'objects': [], 'confidence': 0.0}
                
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return {'objects': [], 'confidence': 0.0}
    
    def _classify_toy_sync(self, image) -> Dict:
        """Synchronous wrapper for toy classification"""
        try:
            if self.clip_model is None or self.clip_preprocess is None:
                return self._fallback_classification()
            
            # Prepare image for CLIP
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
                
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Create text prompts for toy categories
            category_prompts = [f"a {keyword}" for category, keywords in self.toy_categories.items() for keyword in keywords]
            text_inputs = clip.tokenize(category_prompts).to(self.device)
            
            # Compute features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # Compute similarities
                similarities = torch.cosine_similarity(image_features, text_features, dim=-1)
                best_match_idx = similarities.argmax().item()
                confidence = float(similarities[best_match_idx])
            
            # Map back to category
            prompt_to_category = {}
            for category, keywords in self.toy_categories.items():
                for keyword in keywords:
                    prompt_to_category[f"a {keyword}"] = category
            
            best_prompt = category_prompts[best_match_idx]
            category = prompt_to_category.get(best_prompt, 'other')
            
            # Extract toy name from best matching keyword
            toy_name = best_prompt.replace("a ", "").title()
            
            # Brand detection
            brand = self._detect_brand_sync(toy_name.lower())
            
            return {
                'name': toy_name,
                'category': category,
                'brand': brand,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._fallback_classification()
    
    def _assess_condition_sync(self, image) -> Dict:
        """Synchronous wrapper for condition assessment"""
        try:
            if isinstance(image, np.ndarray):
                # Simple condition assessment based on image quality metrics
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                
                # Calculate sharpness (Laplacian variance)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Calculate brightness
                brightness = np.mean(gray)
                
                # Simple heuristic scoring
                if sharpness > 1000 and 50 < brightness < 200:
                    condition_score = 8.5
                    condition_text = "Excellent"
                elif sharpness > 500:
                    condition_score = 7.0
                    condition_text = "Good"
                else:
                    condition_score = 5.5
                    condition_text = "Fair"
                
                return {
                    'score': condition_score,
                    'condition_text': condition_text,
                    'metrics': {
                        'sharpness': sharpness,
                        'brightness': brightness
                    }
                }
            else:
                return {'score': 7.0, 'condition_text': 'Good'}
                
        except Exception as e:
            logger.error(f"Condition assessment error: {e}")
            return {'score': 7.0, 'condition_text': 'Good'}
    
    def _detect_brand_sync(self, toy_name: str) -> Optional[str]:
        """Synchronous brand detection"""
        toy_lower = toy_name.lower()
        for brand, keywords in self.brand_patterns.items():
            if any(keyword in toy_lower for keyword in keywords):
                return brand
        return None
    
    # ALL OTHER METHODS: Exact copies from MLService for compatibility
    # (These maintain exact same signatures and behavior)
    
    async def _check_memory_availability(self) -> bool:
        """Check if sufficient memory is available for processing"""
        try:
            # Check system memory
            memory = psutil.virtual_memory()
            available_mb = memory.available / 1024 / 1024
            
            if available_mb < self._memory_threshold_mb:
                logger.warning(f"Low system memory: {available_mb:.0f}MB available")
                return False
            
            # Check GPU memory if available
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_allocated = torch.cuda.memory_allocated(0)
                    gpu_available_mb = (gpu_memory - gpu_allocated) / 1024 / 1024
                    
                    if gpu_available_mb < self._gpu_memory_threshold_mb:
                        logger.warning(f"Low GPU memory: {gpu_available_mb:.0f}MB available")
                        return False
                except Exception as e:
                    logger.warning(f"GPU memory check failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Memory availability check failed: {e}")
            return True  # Assume OK if check fails
    
    @asynccontextmanager
    async def _optimized_image_context(self, image_data: bytes):
        """Optimized context manager for image preprocessing with multiple formats"""
        class OptimizedImageContext:
            def __init__(self):
                self.base_image = None
                self.yolo_ready = None
                self.clip_ready = None
                self.cv_ready = None
        
        context = OptimizedImageContext()
        
        try:
            # Load base image
            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
            context.base_image = pil_image
            
            # Prepare different formats for different models
            context.yolo_ready = np.array(pil_image)
            context.clip_ready = pil_image
            context.cv_ready = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            yield context
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            yield None
        finally:
            # Cleanup
            if context.base_image:
                context.base_image.close()
            del context
    
    @asynccontextmanager
    async def _memory_context(self, context_name: str):
        """Memory management context"""
        try:
            yield
        finally:
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def _force_cleanup(self):
        """Force cleanup of memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    async def _cleanup_intermediate(self):
        """Cleanup intermediate processing artifacts"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def _periodic_cleanup(self):
        """Periodic cleanup based on request counter"""
        logger.info("Performing periodic cleanup...")
        await self._force_cleanup()
        self._last_cleanup = datetime.utcnow()
    
    async def _emergency_cleanup(self):
        """Emergency cleanup when errors occur"""
        logger.warning("Performing emergency cleanup...")
        await self._force_cleanup()
    
    async def _calculate_rarity(self, classification_result: Dict) -> float:
        """Calculate rarity score based on toy classification"""
        base_rarity = 5.0
        
        category = classification_result.get('category', 'other')
        brand = classification_result.get('brand')
        
        # Adjust rarity based on category
        category_multipliers = {
            'collectible': 2.0,
            'action_figure': 1.2,
            'vehicle': 1.1,
            'building_set': 1.3,
            'other': 0.8
        }
        
        rarity = base_rarity * category_multipliers.get(category, 1.0)
        
        # Brand adjustments
        if brand in ['LEGO', 'Star Wars', 'Marvel']:
            rarity *= 1.2
        
        return min(rarity, 10.0)
    
    async def _estimate_price(self, classification_result: Dict, condition_result: Dict) -> float:
        """Estimate price based on classification and condition"""
        base_prices = {
            'action_figure': 15.0,
            'doll': 20.0,
            'vehicle': 12.0,
            'building_set': 25.0,
            'plush': 8.0,
            'board_game': 18.0,
            'educational': 22.0,
            'outdoor': 30.0,
            'electronic': 45.0,
            'collectible': 50.0,
            'other': 15.0
        }
        
        category = classification_result.get('category', 'other')
        condition_score = condition_result.get('score', 7.0)
        confidence = classification_result.get('confidence', 0.5)
        brand = classification_result.get('brand')
        
        base_price = base_prices.get(category, 15.0)
        
        # Condition adjustment
        condition_multiplier = condition_score / 10.0
        
        # Confidence adjustment
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        # Brand premium
        brand_multiplier = 1.0
        if brand in ['LEGO', 'Star Wars', 'Marvel', 'Pokemon']:
            brand_multiplier = 1.5
        elif brand in ['Barbie', 'Hot Wheels']:
            brand_multiplier = 1.2
        
        estimated_price = base_price * condition_multiplier * confidence_multiplier * brand_multiplier
        
        return round(estimated_price, 2)
    
    def _fallback_classification(self) -> Dict:
        """Fallback classification when CLIP is unavailable"""
        return {
            'name': 'Toy',
            'category': 'other',
            'brand': None,
            'confidence': 0.5,
            'note': 'Limited classification available'
        }
    
    def _get_fallback_analysis(self) -> Dict:
        """Fallback analysis when ML models fail"""
        return {
            'toy_name': 'Unknown Toy',
            'category': 'other',
            'brand': None,
            'condition_score': 7.0,
            'confidence': 0.5,
            'rarity_score': 5.0,
            'estimated_price': 20.0,
            'note': 'Analysis performed with limited AI capabilities',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def cleanup(self):
        """Clean up ML models and free memory"""
        try:
            logger.info("Cleaning up SAFE MLOrchestrator...")
            
            # Clear model references
            self.yolo_model = None
            self.clip_model = None
            self.clip_preprocess = None
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True)
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("SAFE MLOrchestrator cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during SAFE MLOrchestrator cleanup: {e}")


# Global instance for singleton pattern
_ml_orchestrator_instance = None

async def get_ml_orchestrator() -> MLOrchestrator:
    """Get the global MLOrchestrator instance (Singleton pattern)"""
    global _ml_orchestrator_instance
    if _ml_orchestrator_instance is None:
        _ml_orchestrator_instance = MLOrchestrator()
        await _ml_orchestrator_instance.initialize_models()
    return _ml_orchestrator_instance
