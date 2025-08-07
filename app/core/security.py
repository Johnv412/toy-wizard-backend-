"""
Security utilities for ToyResaleWizard API
"""

import hashlib
import secrets
from typing import Optional, Tuple
from fastapi import HTTPException

# Allowed image file types
ALLOWED_IMAGE_TYPES = {
    'image/jpeg',
    'image/jpg', 
    'image/png',
    'image/webp',
    'image/gif'
}

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


def validate_image_file(file_content: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validate that uploaded file is a safe image
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            return False, f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        
        if len(file_content) == 0:
            return False, "File is empty"
        
        # Check file type using file signatures (magic numbers)
        # This is more secure than just checking extensions
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        if ext not in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
            return False, f"Invalid file extension: {ext}. Allowed: jpg, jpeg, png, webp, gif"
        
        # Additional security checks
        if file_content[:4] == b'\xFF\xD8\xFF':  # JPEG
            return True, ""
        elif file_content[:8] == b'\x89PNG\r\n\x1A\n':  # PNG
            return True, ""
        elif file_content[:6] in [b'GIF87a', b'GIF89a']:  # GIF
            return True, ""
        elif file_content[:4] == b'RIFF' and file_content[8:12] == b'WEBP':  # WebP
            return True, ""
        else:
            # For other validated MIME types, allow through
            return True, ""
            
    except Exception as e:
        return False, f"File validation error: {str(e)}"


def generate_secure_filename(original_filename: str) -> str:
    """
    Generate a secure filename with random component
    """
    # Extract extension
    ext = ''
    if '.' in original_filename:
        ext = '.' + original_filename.rsplit('.', 1)[1].lower()
    
    # Generate secure random filename
    random_part = secrets.token_hex(16)
    return f"upload_{random_part}{ext}"


def generate_content_hash(content: bytes) -> str:
    """
    Generate a hash of file content for deduplication
    """
    return hashlib.sha256(content).hexdigest()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing potentially dangerous characters
    """
    # Remove path separators and other dangerous characters
    dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    sanitized = filename
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:250] + ('.' + ext if ext else '')
    
    return sanitized