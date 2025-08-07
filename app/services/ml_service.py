"""
Machine Learning Service for toy identification and analysis with MLOrchestrator pattern
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

from app.core.config import settings

logger = logging.getLogger(__name__)

class MLOrchestrator:
    """
    World-class ML Pipeline Orchestrator for toy identification and analysis
    
    Implements enterprise-grade patterns:
    - Single model loading with shared GPU memory
    - Pipeline orchestration for parallel processing
    - Advanced memory management and cleanup
    - Performance monitoring and optimization
    """
    
    def __init__(self):
        """Initialize MLOrchestrator with optimized architecture"""
        logger.info("Initializing MLOrchestrator with world-class architecture...")
        
        # Core model registry - all models stored in single dictionary
        self.models = {}
        
        # Device configuration with GPU optimization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_gpu = torch.cuda.is_available()
        
        # Performance and memory management
        self._model_lock = threading.RLock()
        self._request_counter = 0
        self._max_requests_before_cleanup = 25  # More aggressive cleanup
        self._last_cleanup = datetime.utcnow()
        
        # Enhanced memory monitoring
        self._memory_threshold_mb = 1536  # Reduced threshold for better performance
        self._gpu_memory_threshold_mb = 768  # Conservative GPU memory
        self._warmup_completed = False
        
        # Load models and warm up GPU
        asyncio.create_task(self._initialize_all_models())
        
        logger.info(f"MLOrchestrator initialized on device: {self.device}")
    
    async def _initialize_all_models(self):
        """Load all ML models into the models dictionary with optimization"""
        try:
            logger.info("Loading models into MLOrchestrator registry...")
            
            # Load YOLOv8 for object detection
            await self._load_yolo_model()
            
            # Load CLIP for image-text matching
            await self._load_clip_model()
            
            # Load Detectron2 for advanced segmentation (placeholder for now)
            await self._load_detectron2_model()
            
            # Warm up GPU memory and models
            await self._warmup_gpu()
            
            logger.info("All models successfully loaded into MLOrchestrator")
            
        except Exception as e:
            logger.error(f"Error loading models into MLOrchestrator: {e}")
            raise
    
    async def _load_yolo_model(self):
        """Load YOLOv8 model with GPU optimization"""
        try:
            logger.info("Loading YOLOv8 model...")
            yolo_model = YOLO(settings.YOLO_MODEL_PATH)
            
            if self.use_gpu:
                yolo_model.to(self.device)
            
            self.models['yolo'] = {
                'model': yolo_model,
                'type': 'object_detection',
                'loaded_at': datetime.utcnow(),
                'memory_usage': self._get_model_memory_usage(yolo_model)
            }
            
            logger.info("YOLOv8 model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load YOLOv8 model: {e}")
            self.models['yolo'] = None
    
    async def _load_clip_model(self):
        """Load CLIP model with GPU optimization"""
        try:
            logger.info("Loading CLIP model...")
            clip_model, clip_preprocess = clip.load(
                settings.CLIP_MODEL_NAME,
                device=self.device
            )
            
            self.models['clip'] = {
                'model': clip_model,
                'preprocess': clip_preprocess,
                'type': 'image_text_matching',
                'loaded_at': datetime.utcnow(),
                'memory_usage': self._get_model_memory_usage(clip_model)
            }
            
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}")
            self.models['clip'] = None
    
    async def _load_detectron2_model(self):
        """Load Detectron2 model for advanced analysis (placeholder)"""
        try:
            # Placeholder for Detectron2 implementation
            # This will be expanded in future iterations
            logger.info("Detectron2 model loading (placeholder)")
            
            self.models['detectron2'] = {
                'model': None,  # Placeholder
                'type': 'instance_segmentation',
                'loaded_at': datetime.utcnow(),
                'memory_usage': 0,
                'status': 'placeholder'
            }
            
        except Exception as e:
            logger.warning(f"Detectron2 model placeholder: {e}")
    
    async def _warmup_gpu(self):
        """
        Warm up GPU memory and models for optimal performance
        This is a critical optimization that pre-allocates GPU memory
        """
        if not self.use_gpu or self._warmup_completed:
            return
        
        try:
            logger.info("Warming up GPU memory and models...")
            
            # Create dummy tensors to warm up GPU
            if self.models.get('yolo') and self.models['yolo']['model']:
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
                with torch.no_grad():
                    _ = self.models['yolo']['model'](dummy_input)
            
            if self.models.get('clip') and self.models['clip']['model']:
                dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_text = clip.tokenize(["warm up text"]).to(self.device)
                with torch.no_grad():
                    _ = self.models['clip']['model'](dummy_image, dummy_text)
            
            # Clear dummy data
            if self.use_gpu:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self._warmup_completed = True
            logger.info("GPU warmup completed successfully")
            
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}")
    
    def _get_model_memory_usage(self, model) -> float:
        """Calculate approximate memory usage of a model in MB"""
        try:
            if hasattr(model, 'parameters'):
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                return param_size / (1024 * 1024)  # Convert to MB
            return 0.0
        except Exception:
            return 0.0
    
    def get_model_status(self) -> Dict:
        """Get status of all loaded models"""
        status = {}
        for model_name, model_info in self.models.items():
            if model_info:
                status[model_name] = {
                    'loaded': model_info.get('model') is not None,
                    'type': model_info.get('type'),
                    'memory_mb': model_info.get('memory_usage', 0),
                    'loaded_at': model_info.get('loaded_at').isoformat() if model_info.get('loaded_at') else None
                }
            else:
                status[model_name] = {'loaded': False, 'error': 'Failed to load'}
        
        return status
    
    async def cleanup(self):
        """Clean up MLOrchestrator models and free memory"""
        try:
            logger.info("Cleaning up MLOrchestrator...")
            
            # Clear all model references
            for model_name in list(self.models.keys()):
                if self.models[model_name]:
                    self.models[model_name] = None
            
            self.models.clear()
            self._warmup_completed = False
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("MLOrchestrator cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during MLOrchestrator cleanup: {e}")
    
    # Compatibility properties for existing code
    @property
    def yolo_model(self):
        """Compatibility property for existing code"""
        yolo_info = self.models.get('yolo')
        return yolo_info['model'] if yolo_info else None
    
    @property
    def clip_model(self):
        """Compatibility property for existing code"""
        clip_info = self.models.get('clip')
        return clip_info['model'] if clip_info else None
    
    @property
    def clip_preprocess(self):
        """Compatibility property for existing code"""
        clip_info = self.models.get('clip')
        return clip_info['preprocess'] if clip_info else None

# Create global instance of MLOrchestrator
_ml_orchestrator_instance = None

async def get_ml_orchestrator() -> MLOrchestrator:
    """Get the global MLOrchestrator instance (Singleton pattern)"""
    global _ml_orchestrator_instance
    if _ml_orchestrator_instance is None:
        _ml_orchestrator_instance = MLOrchestrator()
    return _ml_orchestrator_instance


# MLService class for backwards compatibility
class MLService:
    """Legacy MLService class - delegates to MLOrchestrator for backwards compatibility"""
    
    def __init__(self):
        # Get reference to global MLOrchestrator
        self._orchestrator = None
        
        # Device info for compatibility
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Memory management (for compatibility)
        self._model_lock = threading.RLock()
        self._request_counter = 0
        self._max_requests_before_cleanup = 50
        self._last_cleanup = datetime.utcnow()
        self._memory_threshold_mb = 2048
        self._gpu_memory_threshold_mb = 1024
        
        # Toy categories mapping
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
        
        # Brand recognition patterns
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
    
    async def _get_orchestrator(self) -> MLOrchestrator:
        """Get the global MLOrchestrator instance"""
        if self._orchestrator is None:
            self._orchestrator = await get_ml_orchestrator()
        return self._orchestrator
    
    # Compatibility properties - delegate to MLOrchestrator
    @property
    def yolo_model(self):
        """Get YOLO model from orchestrator (synchronous access)"""
        if self._orchestrator is None:
            return None
        return self._orchestrator.yolo_model
    
    @property
    def clip_model(self):
        """Get CLIP model from orchestrator (synchronous access)"""
        if self._orchestrator is None:
            return None
        return self._orchestrator.clip_model
    
    @property
    def clip_preprocess(self):
        """Get CLIP preprocessor from orchestrator (synchronous access)"""
        if self._orchestrator is None:
            return None
        return self._orchestrator.clip_preprocess
    
    async def initialize_models(self):
        """Initialize ML models - delegates to MLOrchestrator"""
        try:
            logger.info("Initializing ML models via MLOrchestrator...")
            
            # Get orchestrator instance (this triggers model loading)
            self._orchestrator = await get_ml_orchestrator()
            
            # Wait for models to be loaded
            while not self._orchestrator._warmup_completed:
                await asyncio.sleep(0.1)
            
            logger.info("ML models initialization completed via MLOrchestrator")
            
        except Exception as e:
            logger.error(f"Error initializing ML models via MLOrchestrator: {e}")
            raise
    
    async def analyze_toy_image(self, image_data: bytes) -> Dict:
        """Analyze toy image with optimized parallel processing"""
        request_id = self._request_counter
        self._request_counter += 1
        
        # Check memory before processing
        if not await self._check_memory_availability():
            await self._force_cleanup()
        
        try:
            logger.info(f"Starting optimized analysis request {request_id}")
            
            # Memory-conscious image preprocessing with optimization
            async with self._optimized_image_context(image_data) as processed_images:
                if processed_images is None:
                    return self._get_fallback_analysis()
                
                # OPTIMIZATION: Run all ML models in parallel instead of sequential
                async with self._memory_context(f"parallel_analysis_{request_id}"):
                    start_time = asyncio.get_event_loop().time()
                    
                    # Parallel execution of all ML tasks
                    detection_task = asyncio.create_task(
                        self._detect_objects_optimized(processed_images.yolo_ready)
                    )
                    classification_task = asyncio.create_task(
                        self._classify_toy_optimized(processed_images.clip_ready)
                    )
                    condition_task = asyncio.create_task(
                        self._assess_condition_optimized(processed_images.cv_ready)
                    )
                    
                    # Wait for all tasks to complete
                    detection_result, classification_result, condition_result = \
                        await asyncio.gather(
                            detection_task,
                            classification_task, 
                            condition_task,
                            return_exceptions=True
                        )
                    
                    processing_time = asyncio.get_event_loop().time() - start_time
                    logger.info(f"Parallel ML processing completed in {processing_time:.2f}s")
                    
                    # Handle any exceptions from parallel tasks
                    if isinstance(detection_result, Exception):
                        logger.warning(f"Detection failed: {detection_result}")
                        detection_result = {'objects': [], 'confidence': 0.0}
                    
                    if isinstance(classification_result, Exception):
                        logger.warning(f"Classification failed: {classification_result}")
                        classification_result = self._fallback_classification()
                        
                    if isinstance(condition_result, Exception):
                        logger.warning(f"Condition assessment failed: {condition_result}")
                        condition_result = {'score': 7.0, 'condition_text': 'Good'}
                
                # Immediate cleanup after parallel processing
                await self._cleanup_intermediate()
                
                # Combine results
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
                
                # Periodic cleanup
                if self._request_counter % self._max_requests_before_cleanup == 0:
                    await self._periodic_cleanup()
                
                logger.info(f"Completed optimized analysis request {request_id} in {processing_time:.2f}s")
                return analysis
                
        except Exception as e:
            logger.error(f"Error in optimized analysis request {request_id}: {e}")
            await self._emergency_cleanup()
            return self._get_fallback_analysis()
        finally:
            # Ensure cleanup always happens
            await self._cleanup_intermediate()

    @asynccontextmanager
    async def _optimized_image_context(self, image_data: bytes):
        """Optimized context manager for image preprocessing with multiple formats"""
        class OptimizedImageContext:
            def __init__(self):
                self.base_image = None
                self.yolo_ready = None
                self.clip_ready = None
                self.cv_ready = None
            
            def __enter__(self):
                try:
                    # Single image load and conversion
                    self.base_image = Image.open(io.BytesIO(image_data))
                    
                    # Ensure RGB format once
                    if self.base_image.mode != 'RGB':
                        old_image = self.base_image
                        self.base_image = self.base_image.convert('RGB')
                        old_image.close()
                    
                    # Single resize operation
                    max_size = (1024, 1024)
                    if self.base_image.size[0] > max_size[0] or self.base_image.size[1] > max_size[1]:
                        old_image = self.base_image
                        self.base_image.thumbnail(max_size, Image.Resampling.LANCZOS)
                        old_image.close()
                    
                    # Prepare optimized formats for each model (done once)
                    self.yolo_ready = np.array(self.base_image)  # YOLO needs numpy array
                    self.clip_ready = self.base_image            # CLIP needs PIL Image
                    self.cv_ready = cv2.cvtColor(self.yolo_ready, cv2.COLOR_RGB2BGR)  # OpenCV format
                    
                    return self
                except Exception as e:
                    logger.error(f"Error in optimized image preprocessing: {e}")
                    return None
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Cleanup all image references
                if self.base_image:
                    try:
                        self.base_image.close()
                    except:
                        pass
                
                # Clear numpy arrays
                if self.yolo_ready is not None:
                    del self.yolo_ready
                if self.cv_ready is not None:
                    del self.cv_ready
                
                # Force garbage collection
                gc.collect()
        
        context = OptimizedImageContext()
        try:
            yield context.__enter__()
        finally:
            context.__exit__(None, None, None)

    async def _detect_objects_optimized(self, yolo_image: np.ndarray) -> Dict:
        """Optimized object detection with reduced memory footprint"""
        try:
            if self.yolo_model is None:
                return {'objects': [], 'confidence': 0.0}
            
            # Single inference call with memory management
            with torch.no_grad():
                results = self.yolo_model(yolo_image, verbose=False)
                
                detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            detection = {
                                'class': result.names[int(box.cls)],
                                'confidence': float(box.conf),
                                'bbox': box.xyxy.tolist()[0]
                            }
                            detections.append(detection)
                
                # Immediate cleanup
                del results
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return {
                    'objects': detections,
                    'count': len(detections),
                    'primary_object': detections[0] if detections else None
                }
            
        except Exception as e:
            logger.error(f"Error in optimized object detection: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {'objects': [], 'confidence': 0.0}

    async def _classify_toy_optimized(self, clip_image: Image.Image) -> Dict:
        """Optimized toy classification with efficient text processing"""
        try:
            if self.clip_model is None:
                return self._fallback_classification()
            
            # Pre-computed text embeddings (cache these in production)
            category_prompts = [f"a {keyword}" for category, keywords in self.toy_categories.items() for keyword in keywords]
            brand_prompts = [f"a {brand} {keyword}" for brand, keywords in self.brand_patterns.items() for keyword in keywords]
            
            # Single preprocessing and inference
            with torch.no_grad():
                # Image preprocessing
                image_input = self.clip_preprocess(clip_image).unsqueeze(0).to(self.device)
                
                # Text tokenization
                category_text = clip.tokenize(category_prompts[:77]).to(self.device)  # Limit tokens
                brand_text = clip.tokenize(brand_prompts[:77]).to(self.device)
                
                # Parallel encoding
                image_features = self.clip_model.encode_image(image_input)
                category_features = self.clip_model.encode_text(category_text)
                brand_features = self.clip_model.encode_text(brand_text)
                
                # Fast similarity computation
                category_similarities = (100.0 * image_features @ category_features.T).softmax(dim=-1)
                brand_similarities = (100.0 * image_features @ brand_features.T).softmax(dim=-1)
                
                # Extract results
                best_category_idx = category_similarities.argmax().item()
                best_brand_idx = brand_similarities.argmax().item()
                
                category_confidence = category_similarities[0, best_category_idx].item()
                brand_confidence = brand_similarities[0, best_brand_idx].item()
                
                # Immediate cleanup
                del image_input, image_features, category_features, brand_features
                del category_similarities, brand_similarities
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Map results efficiently
            category_prompt = category_prompts[best_category_idx]
            brand_prompt = brand_prompts[best_brand_idx]
            
            category = next((cat for cat, keywords in self.toy_categories.items() 
                           if any(keyword in category_prompt for keyword in keywords)), 'other')
            
            brand = next((b for b, keywords in self.brand_patterns.items()
                         if any(keyword in brand_prompt for keyword in keywords)), None)
            
            toy_name = self._generate_toy_name(category, brand, category_confidence)
            
            return {
                'name': toy_name,
                'category': category,
                'brand': brand,
                'confidence': max(category_confidence, brand_confidence),
                'category_confidence': category_confidence,
                'brand_confidence': brand_confidence
            }
            
        except Exception as e:
            logger.error(f"Error in optimized toy classification: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._fallback_classification()

    async def _assess_condition_optimized(self, cv_image: np.ndarray) -> Dict:
        """Optimized condition assessment with efficient image analysis"""
        try:
            # Fast condition analysis using vectorized operations
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Parallel metric calculations
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
            brightness = np.mean(gray) / 255.0 * 10.0
            
            # Fast color analysis
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            color_vibrancy = np.std(hsv[:, :, 1]) / 255.0 * 10.0
            
            # Simple defect detection
            edges = cv2.Canny(gray, 50, 150)
            defect_score = 10.0 - min(np.count_nonzero(edges) / edges.size * 20, 10.0)
            
            # Quick condition calculation
            condition_score = min(10.0, (sharpness + brightness + color_vibrancy + defect_score) / 4.0)
            
            return {
                'score': max(1.0, condition_score),
                'sharpness': min(10.0, sharpness),
                'brightness': min(10.0, brightness),
                'color_vibrancy': min(10.0, color_vibrancy),
                'defect_score': defect_score,
                'condition_text': self._get_condition_text(condition_score)
            }
            
        except Exception as e:
            logger.error(f"Error in optimized condition assessment: {e}")
            return {
                'score': 7.0,
                'condition_text': 'Good',
                'note': 'Optimized condition assessment unavailable'
            }

    @asynccontextmanager
    async def _memory_context(self, operation_name: str):
        """Context manager for memory-aware operations"""
        initial_memory = await self._get_memory_usage()
        try:
            logger.debug(f"Starting {operation_name}, memory: {initial_memory}MB")
            yield
        finally:
            final_memory = await self._get_memory_usage()
            logger.debug(f"Finished {operation_name}, memory: {final_memory}MB")
            await self._cleanup_intermediate()

    def _memory_managed_image(self, image_data: bytes):
        """Context manager for safe image handling"""
        class ImageContext:
            def __enter__(self):
                try:
                    # Convert bytes to PIL Image with size limits
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Ensure RGB format
                    if image.mode != 'RGB':
                        old_image = image
                        image = image.convert('RGB')
                        old_image.close()
                    
                    # Resize if too large to prevent memory issues
                    max_size = (1024, 1024)
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        old_image = image
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                        old_image.close()
                    
                    return image
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    return None
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Image cleanup handled by caller
                pass
        
        return ImageContext()

    async def _check_memory_availability(self) -> bool:
        """Check if sufficient memory is available"""
        try:
            # Check system memory
            memory = psutil.virtual_memory()
            if memory.percent > 85:  # 85% threshold
                logger.warning(f"High system memory usage: {memory.percent}%")
                return False
            
            # Check GPU memory if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                if gpu_memory > self._gpu_memory_threshold_mb:
                    logger.warning(f"High GPU memory usage: {gpu_memory:.1f}MB")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking memory: {e}")
            return True  # Assume OK if can't check

    async def _cleanup_intermediate(self):
        """Clean up intermediate processing artifacts"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clean GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            logger.error(f"Error during intermediate cleanup: {e}")

    async def _force_cleanup(self):
        """Force aggressive cleanup when memory is low"""
        try:
            logger.warning("Forcing aggressive memory cleanup")
            
            # Multiple GC passes
            for _ in range(3):
                gc.collect()
            
            # GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # Log memory status
            await self._log_memory_status()
            
        except Exception as e:
            logger.error(f"Error during force cleanup: {e}")

    async def _emergency_cleanup(self):
        """Emergency cleanup on errors"""
        try:
            logger.error("Performing emergency cleanup")
            
            # Clear all possible references
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            logger.error(f"Error during emergency cleanup: {e}")

    async def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    async def _log_memory_status(self):
        """Log current memory status"""
        try:
            memory = psutil.virtual_memory()
            process_memory = await self._get_memory_usage()
            
            logger.info(f"Memory status - System: {memory.percent:.1f}%, Process: {process_memory:.1f}MB")
            
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated() / 1024**2
                gpu_reserved = torch.cuda.memory_reserved() / 1024**2
                logger.info(f"GPU memory - Allocated: {gpu_allocated:.1f}MB, Reserved: {gpu_reserved:.1f}MB")
                
        except Exception as e:
            logger.error(f"Error logging memory status: {e}")

    async def _periodic_cleanup(self):
        """Periodic maintenance cleanup"""
        try:
            logger.info("Performing periodic cleanup")
            await self._force_cleanup()
            self._last_cleanup = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}")
    
    async def _detect_objects(self, image: Image.Image) -> Dict:
        """Detect objects in the image using YOLO"""
        try:
            if self.yolo_model is None:
                return {'objects': [], 'confidence': 0.0}
            
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Run YOLO detection
            results = self.yolo_model(img_array)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = {
                            'class': result.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
                        }
                        detections.append(detection)
            
            return {
                'objects': detections,
                'count': len(detections),
                'primary_object': detections[0] if detections else None
            }
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return {'objects': [], 'confidence': 0.0}
    
    async def _classify_toy(self, image: Image.Image) -> Dict:
        """Classify toy using CLIP model"""
        try:
            if self.clip_model is None:
                return self._fallback_classification()
            
            # Prepare text prompts for classification
            category_prompts = []
            for category, keywords in self.toy_categories.items():
                for keyword in keywords:
                    category_prompts.append(f"a {keyword}")
            
            brand_prompts = []
            for brand, keywords in self.brand_patterns.items():
                for keyword in keywords:
                    brand_prompts.append(f"a {brand} {keyword}")
            
            # Preprocess image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text prompts
            category_text = clip.tokenize(category_prompts).to(self.device)
            brand_text = clip.tokenize(brand_prompts).to(self.device)
            
            # Calculate similarities
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                category_features = self.clip_model.encode_text(category_text)
                brand_features = self.clip_model.encode_text(brand_text)
                
                # Calculate similarities
                category_similarities = (100.0 * image_features @ category_features.T).softmax(dim=-1)
                brand_similarities = (100.0 * image_features @ brand_features.T).softmax(dim=-1)
                
                # Clean up GPU tensors immediately
                del image_features, category_features, brand_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Get best matches
            best_category_idx = category_similarities.argmax().item()
            best_brand_idx = brand_similarities.argmax().item()
            
            category_confidence = category_similarities[0, best_category_idx].item()
            brand_confidence = brand_similarities[0, best_brand_idx].item()
            
            # Map back to categories and brands
            category_prompt = category_prompts[best_category_idx]
            brand_prompt = brand_prompts[best_brand_idx]
            
            # Extract category
            category = None
            for cat, keywords in self.toy_categories.items():
                if any(keyword in category_prompt for keyword in keywords):
                    category = cat
                    break
            
            # Extract brand
            brand = None
            for b, keywords in self.brand_patterns.items():
                if any(keyword in brand_prompt for keyword in keywords):
                    brand = b
                    break
            
            # Generate toy name
            toy_name = self._generate_toy_name(category, brand, category_confidence)
            
            return {
                'name': toy_name,
                'category': category or 'other',
                'brand': brand,
                'confidence': max(category_confidence, brand_confidence),
                'category_confidence': category_confidence,
                'brand_confidence': brand_confidence
            }
            
        except Exception as e:
            logger.error(f"Error in toy classification: {e}")
            return self._fallback_classification()
    
    async def _assess_condition(self, image: Image.Image) -> Dict:
        """Assess toy condition based on visual analysis"""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Analyze image quality factors
            sharpness = self._calculate_sharpness(cv_image)
            brightness = self._calculate_brightness(cv_image)
            color_vibrancy = self._calculate_color_vibrancy(cv_image)
            defect_score = await self._detect_defects(cv_image)
            
            # Calculate condition score (1-10 scale)
            condition_score = self._calculate_condition_score(
                sharpness, brightness, color_vibrancy, defect_score
            )
            
            return {
                'score': condition_score,
                'sharpness': sharpness,
                'brightness': brightness,
                'color_vibrancy': color_vibrancy,
                'defect_score': defect_score,
                'condition_text': self._get_condition_text(condition_score)
            }
            
        except Exception as e:
            logger.error(f"Error assessing condition: {e}")
            return {
                'score': 7.0,
                'condition_text': 'Good',
                'note': 'Condition assessment unavailable'
            }
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(laplacian_var / 1000.0, 10.0)  # Normalize to 0-10
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) / 255.0 * 10.0  # Scale to 0-10
    
    def _calculate_color_vibrancy(self, image: np.ndarray) -> float:
        """Calculate color vibrancy/saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        return np.mean(saturation) / 255.0 * 10.0  # Scale to 0-10
    
    async def _detect_defects(self, image: np.ndarray) -> float:
        """Detect potential defects (scratches, dents, etc.)"""
        # Simplified defect detection
        # In production, this would use more sophisticated algorithms
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection to find potential defects
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Higher edge density might indicate more defects
        defect_score = min(edge_density * 20, 10.0)
        return 10.0 - defect_score  # Invert so higher is better
    
    def _calculate_condition_score(self, sharpness: float, brightness: float, 
                                 color_vibrancy: float, defect_score: float) -> float:
        """Calculate overall condition score"""
        # Weighted average of different factors
        weights = {
            'sharpness': 0.2,
            'brightness': 0.2,
            'color_vibrancy': 0.3,
            'defect_score': 0.3
        }
        
        score = (
            sharpness * weights['sharpness'] +
            brightness * weights['brightness'] +
            color_vibrancy * weights['color_vibrancy'] +
            defect_score * weights['defect_score']
        )
        
        return max(1.0, min(10.0, score))
    
    def _get_condition_text(self, score: float) -> str:
        """Convert condition score to text description"""
        if score >= 9.0:
            return "Mint/New"
        elif score >= 8.0:
            return "Excellent"
        elif score >= 7.0:
            return "Very Good"
        elif score >= 6.0:
            return "Good"
        elif score >= 5.0:
            return "Fair"
        elif score >= 3.0:
            return "Poor"
        else:
            return "Damaged"
    
    async def _calculate_rarity(self, classification: Dict) -> float:
        """Calculate rarity score based on toy classification"""
        # Simplified rarity calculation
        # In production, this would use market data and production numbers
        base_rarity = 5.0
        
        # Adjust based on brand
        brand = classification.get('brand')
        if brand in ['LEGO', 'Pokemon', 'Star Wars']:
            base_rarity += 2.0
        elif brand in ['Marvel', 'DC', 'Transformers']:
            base_rarity += 1.5
        
        # Adjust based on category
        category = classification.get('category')
        if category in ['collectible', 'action_figure']:
            base_rarity += 1.0
        elif category in ['electronic', 'building_set']:
            base_rarity += 0.5
        
        # Add some randomness for now (would be replaced with real data)
        import random
        random.seed(hash(classification.get('name', '')))
        variance = (random.random() - 0.5) * 2.0
        
        return max(1.0, min(10.0, base_rarity + variance))
    
    async def _estimate_price(self, classification: Dict, condition: Dict) -> float:
        """Estimate toy price based on classification and condition"""
        # Base price estimation
        base_prices = {
            'action_figure': 25.0,
            'doll': 20.0,
            'vehicle': 15.0,
            'building_set': 40.0,
            'plush': 15.0,
            'board_game': 30.0,
            'educational': 25.0,
            'outdoor': 35.0,
            'electronic': 45.0,
            'collectible': 50.0
        }
        
        category = classification.get('category', 'other')
        base_price = base_prices.get(category, 20.0)
        
        # Brand multiplier
        brand_multipliers = {
            'LEGO': 2.0,
            'Pokemon': 1.8,
            'Star Wars': 1.7,
            'Marvel': 1.5,
            'DC': 1.5,
            'Transformers': 1.4,
            'Barbie': 1.3,
            'Hot Wheels': 1.2,
            'Nerf': 1.1
        }
        
        brand = classification.get('brand')
        brand_multiplier = brand_multipliers.get(brand, 1.0)
        
        # Condition multiplier
        condition_score = condition.get('score', 7.0)
        condition_multiplier = max(0.3, condition_score / 10.0)
        
        # Confidence multiplier
        confidence = classification.get('confidence', 0.5)
        confidence_multiplier = max(0.8, confidence)
        
        estimated_price = base_price * brand_multiplier * condition_multiplier * confidence_multiplier
        
        return round(estimated_price, 2)
    
    def _generate_toy_name(self, category: str, brand: str, confidence: float) -> str:
        """Generate a descriptive toy name"""
        if brand and confidence > 0.7:
            if category:
                return f"{brand} {category.replace('_', ' ').title()}"
            else:
                return f"{brand} Toy"
        elif category:
            return f"{category.replace('_', ' ').title()}"
        else:
            return "Unknown Toy"
    
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
            logger.info("Cleaning up ML service...")
            
            # Clear model references
            self.yolo_model = None
            self.clip_model = None
            self.clip_preprocess = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("ML service cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during ML service cleanup: {e}")