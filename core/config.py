"""
Configuration management for the intelligent data processor.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # MongoDB
    mongo_url: str = Field(
        default="mongodb://localhost:27017",
        env="MONGO_URL"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # Application
    debug: bool = Field(default=True, env="DEBUG")
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    max_file_size: str = Field(default="50MB", env="MAX_FILE_SIZE")
    upload_dir: Path = Field(default=Path("./uploads"), env="UPLOAD_DIR")
    
    # AI/ML
    spacy_model: str = Field(default="en_core_web_sm", env="SPACY_MODEL")
    use_gpu: bool = Field(default=False, env="USE_GPU")
    confidence_threshold: float = Field(default=0.7, env="CONFIDENCE_THRESHOLD")
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    use_openai_classification: bool = Field(default=True, env="USE_OPENAI_CLASSIFICATION")
    
    # OCR
    tesseract_path: Optional[str] = Field(default=None, env="TESSERACT_PATH")
    ocr_languages: str = Field(default="eng", env="OCR_LANGUAGES")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Path = Field(default=Path("./logs/app.log"), env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert max file size string to bytes."""
        size_str = self.max_file_size.upper()
        if size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)


# Global settings instance
settings = Settings()
