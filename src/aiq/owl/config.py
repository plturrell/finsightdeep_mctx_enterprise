"""
Configuration Module

This module manages application configuration settings,
providing a centralized way to access and modify settings.
"""

import os
import logging
from pydantic_settings import BaseSettings
from functools import lru_cache

log = logging.getLogger("mctx")


class Settings(BaseSettings):
    """
    Application configuration settings.
    
    This class manages all configuration settings for the MCTX API,
    including environment-specific settings and defaults.
    """
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MCTX API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    # Performance Settings
    DEFAULT_BATCH_SIZE: int = 32
    MAX_BATCH_SIZE: int = 128
    DEFAULT_NUM_SIMULATIONS: int = 32
    MAX_NUM_SIMULATIONS: int = 1000
    
    # Security Settings
    API_KEY_REQUIRED: bool = False
    API_KEY_NAME: str = "X-API-Key"
    API_KEY: str = ""
    
    # JAX Settings
    JAX_PLATFORM: str = "cpu"  # Options: cpu, gpu, tpu
    JAX_ENABLE_X64: bool = False
    JAX_DEBUG: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    This function returns the application settings, caching them
    for improved performance.
    
    Returns:
        Settings: Application configuration settings
    """
    return Settings()


# Configure logging based on settings
def setup_logging() -> logging.Logger:
    """
    Set up application logging based on configuration.
    
    Returns:
        Logger: Configured logger instance
    """
    settings = get_settings()
    log_level = getattr(logging, settings.LOG_LEVEL.upper())
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    # Configure MCTX logger
    logger = logging.getLogger("mctx")
    logger.setLevel(log_level)
    
    # Add console handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Configure JAX based on settings
    os.environ["JAX_PLATFORM_NAME"] = settings.JAX_PLATFORM
    os.environ["JAX_ENABLE_X64"] = "1" if settings.JAX_ENABLE_X64 else "0"
    os.environ["JAX_DEBUG_NANS"] = "1" if settings.JAX_DEBUG else "0"
    
    return logger