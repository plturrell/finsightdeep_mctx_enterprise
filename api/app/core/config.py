import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from functools import lru_cache


class Settings(BaseSettings):
    """API settings."""
    
    # Core settings
    debug: bool = False
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # API settings
    api_secret_key: str = "default_development_key_replace_in_production"
    api_key_required: bool = False
    api_key: Optional[str] = "test_api_key"
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    
    # HANA settings
    hana_enabled: bool = True
    hana_host: Optional[str] = None
    hana_port: int = 443
    hana_user: Optional[str] = None
    hana_password: Optional[str] = None
    hana_schema: str = "MCTX"
    
    # Performance settings
    max_batch_size: int = 64
    max_num_simulations: int = 500
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from environment variable."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_prefix = ""


@lru_cache()
def get_settings():
    """Get cached settings instance."""
    return Settings(
        # Core settings from environment
        debug=os.environ.get("DEBUG", "False").lower() in ("true", "1", "yes"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        
        # API settings
        api_secret_key=os.environ.get("API_SECRET_KEY", "default_development_key_replace_in_production"),
        api_key_required=os.environ.get("API_KEY_REQUIRED", "False").lower() in ("true", "1", "yes"),
        api_key=os.environ.get("API_KEY", "test_api_key"),
        
        # CORS settings from environment
        cors_origins=os.environ.get("CORS_ORIGINS", "*"),
        
        # HANA settings
        hana_enabled=os.environ.get("HANA_ENABLED", "True").lower() in ("true", "1", "yes"),
        hana_host=os.environ.get("HANA_HOST"),
        hana_port=int(os.environ.get("HANA_PORT", "443")),
        hana_user=os.environ.get("HANA_USER"),
        hana_password=os.environ.get("HANA_PASSWORD"),
        hana_schema=os.environ.get("HANA_SCHEMA", "MCTX"),
        
        # Performance settings
        max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "64")),
        max_num_simulations=int(os.environ.get("MAX_NUM_SIMULATIONS", "500")),
    )