"""
Security middleware for ToyResaleWizard API
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for rate limiting and request monitoring
    """
    
    def __init__(self, app, rate_limit: int = 100):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.requests = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Log request
        start_time = time.time()
        client_ip = request.client.host
        
        # Basic rate limiting (in-memory, for development)
        current_time = int(time.time() / 60)  # Per minute
        if client_ip not in self.requests:
            self.requests[client_ip] = {}
        
        if current_time not in self.requests[client_ip]:
            self.requests[client_ip][current_time] = 0
        
        self.requests[client_ip][current_time] += 1
        
        if self.requests[client_ip][current_time] > self.rate_limit:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        
        # Process request
        response = await call_next(request)
        
        # Log response time
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
        
        return response

def add_security_headers(response: Response) -> Response:
    """
    Add security headers to response
    """
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response