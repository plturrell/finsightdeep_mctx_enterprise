"""
Security Module

This module provides security utilities including secret management,
encryption, and integration with external authentication systems.
"""

import os
import json
import base64
import logging
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
from dotenv import load_dotenv

logger = logging.getLogger("mctx")

# Load environment variables from .env file if present
load_dotenv()

# Encryption key - in production this should come from a secure vault
# If not set, we'll generate one, but it won't persist across restarts
# For Kubernetes, use a secret mounted as an environment variable
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

if not ENCRYPTION_KEY:
    ENCRYPTION_KEY = base64.urlsafe_b64encode(os.urandom(32)).decode()
    logger.warning(
        "No encryption key found. Generated a temporary key that will not persist across restarts. "
        "For production, set ENCRYPTION_KEY environment variable."
    )

# Create cipher suite
cipher_suite = Fernet(ENCRYPTION_KEY.encode())


class SecretManager:
    """
    Manager for handling secure secrets.
    
    This class provides methods for securely storing and retrieving
    sensitive configuration like database credentials.
    """
    
    @staticmethod
    def encrypt(data: str) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: Plain text data to encrypt
            
        Returns:
            Encrypted data as string
        """
        return cipher_suite.encrypt(data.encode()).decode()
    
    @staticmethod
    def decrypt(data: str) -> str:
        """
        Decrypt sensitive data.
        
        Args:
            data: Encrypted data
            
        Returns:
            Decrypted plain text
        """
        return cipher_suite.decrypt(data.encode()).decode()
    
    @staticmethod
    def get_secret(name: str) -> Optional[str]:
        """
        Get a secret from environment or mounted secret.
        
        This function prioritizes:
        1. Direct environment variable
        2. Mounted secret files (for Kubernetes)
        3. Encrypted secrets in environment
        
        Args:
            name: Secret name
            
        Returns:
            Secret value or None if not found
        """
        # Try direct environment variable
        value = os.getenv(name)
        if value:
            return value
        
        # Try mounted secret (Kubernetes style)
        secret_path = f"/run/secrets/{name}"
        if os.path.exists(secret_path):
            with open(secret_path, "r") as f:
                return f.read().strip()
        
        # Try encrypted environment variable
        encrypted_value = os.getenv(f"{name}_ENCRYPTED")
        if encrypted_value:
            try:
                return SecretManager.decrypt(encrypted_value)
            except Exception as e:
                logger.error(f"Failed to decrypt {name}: {str(e)}")
        
        return None
    
    @staticmethod
    def store_secret(name: str, value: str) -> bool:
        """
        Store a secret securely.
        
        For local development, this encrypts and stores in the environment.
        In production, this should use a proper secret store.
        
        Args:
            name: Secret name
            value: Secret value
            
        Returns:
            bool: True if successful
        """
        try:
            # Encrypt the value
            encrypted_value = SecretManager.encrypt(value)
            
            # Store in environment
            os.environ[f"{name}_ENCRYPTED"] = encrypted_value
            
            return True
        except Exception as e:
            logger.error(f"Failed to store secret {name}: {str(e)}")
            return False


# SAP HANA Cloud credentials
class HanaSecrets:
    """SAP HANA Cloud specific secrets."""
    
    # HANA Cloud credentials
    CLOUD_CLIENT_KEY = "sb-8e750c08-383e-41b6-91bc-d2e25b02b6cf!b549933|client!b3650"
    CLOUD_CLIENT_SECRET = "a40f4d8c-ec0e-4a88-8b3a-c1b8aeb68b18$i79LsZz2SdCKcyoOEoTi4ItbeH85ExO8MSyDq2apes8="
    TOKEN_URL = "https://vp-dsp-poc03.authentication.eu10.hana.ondemand.com/oauth/token"
    
    # HANA Database connection parameters
    HANA_HOST = "d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com"
    HANA_PORT = "443"
    HANA_USER = "DBADMIN"
    HANA_PASSWORD = "Initial@1"
    
    @staticmethod
    def initialize() -> None:
        """Initialize HANA secrets in secure storage."""
        # Store secrets securely if not already present
        if not SecretManager.get_secret("HANA_CLOUD_CLIENT_KEY"):
            SecretManager.store_secret("HANA_CLOUD_CLIENT_KEY", HanaSecrets.CLOUD_CLIENT_KEY)
        
        if not SecretManager.get_secret("HANA_CLOUD_CLIENT_SECRET"):
            SecretManager.store_secret("HANA_CLOUD_CLIENT_SECRET", HanaSecrets.CLOUD_CLIENT_SECRET)
        
        if not SecretManager.get_secret("HANA_TOKEN_URL"):
            SecretManager.store_secret("HANA_TOKEN_URL", HanaSecrets.TOKEN_URL)
        
        # HANA database connection parameters
        if not SecretManager.get_secret("HANA_HOST"):
            SecretManager.store_secret("HANA_HOST", HanaSecrets.HANA_HOST)
        
        if not SecretManager.get_secret("HANA_PORT"):
            SecretManager.store_secret("HANA_PORT", HanaSecrets.HANA_PORT)
        
        if not SecretManager.get_secret("HANA_USER"):
            SecretManager.store_secret("HANA_USER", HanaSecrets.HANA_USER)
        
        if not SecretManager.get_secret("HANA_PASSWORD"):
            SecretManager.store_secret("HANA_PASSWORD", HanaSecrets.HANA_PASSWORD)
            
        logger.info("HANA Cloud secrets initialized successfully")
    
    @staticmethod
    def get_client_credentials() -> Dict[str, str]:
        """
        Get HANA Cloud client credentials.
        
        Returns:
            Dict with client_id and client_secret
        """
        return {
            "client_id": SecretManager.get_secret("HANA_CLOUD_CLIENT_KEY") or HanaSecrets.CLOUD_CLIENT_KEY,
            "client_secret": SecretManager.get_secret("HANA_CLOUD_CLIENT_SECRET") or HanaSecrets.CLOUD_CLIENT_SECRET,
        }
    
    @staticmethod
    def get_oauth_urls() -> Dict[str, str]:
        """
        Get HANA Cloud OAuth URLs.
        
        Returns:
            Dict with auth_url and token_url
        """
        token_url = SecretManager.get_secret("HANA_TOKEN_URL") or HanaSecrets.TOKEN_URL
        # Derive auth URL from token URL
        auth_url = token_url.replace("/oauth/token", "/oauth/authorize")
        
        return {
            "auth_url": auth_url,
            "token_url": token_url,
        }
    
    @staticmethod
    def get_db_connection_params() -> Dict[str, str]:
        """
        Get HANA database connection parameters.
        
        Returns:
            Dict with connection parameters
        """
        return {
            "host": SecretManager.get_secret("HANA_HOST") or HanaSecrets.HANA_HOST,
            "port": SecretManager.get_secret("HANA_PORT") or HanaSecrets.HANA_PORT,
            "user": SecretManager.get_secret("HANA_USER") or HanaSecrets.HANA_USER,
            "password": SecretManager.get_secret("HANA_PASSWORD") or HanaSecrets.HANA_PASSWORD,
        }


# SAP DataSphere credentials
class DataSphereSecrets:
    """SAP DataSphere specific secrets."""
    
    # DataSphere credentials
    DS_CLIENT_KEY = "sb-8e750c08-383e-41b6-91bc-d2e25b02b6cf!b549933|client!b3650"
    DS_CLIENT_SECRET = "a40f4d8c-ec0e-4a88-8b3a-c1b8aeb68b18$i79LsZz2SdCKcyoOEoTi4ItbeH85ExO8MSyDq2apes8="
    DS_AUTH_URL = "https://vp-dsp-poc03.authentication.eu10.hana.ondemand.com/oauth/authorize"
    DS_TOKEN_URL = "https://vp-dsp-poc03.authentication.eu10.hana.ondemand.com/oauth/token"
    DS_API_URL = "https://vp-dsp-poc03.eu10.hcs.cloud.sap"
    DS_SPACE_ID = "DP_001"
    
    @staticmethod
    def initialize() -> None:
        """Initialize DataSphere secrets in secure storage."""
        # Store secrets securely if not already present
        if not SecretManager.get_secret("DS_CLIENT_KEY"):
            SecretManager.store_secret("DS_CLIENT_KEY", DataSphereSecrets.DS_CLIENT_KEY)
        
        if not SecretManager.get_secret("DS_CLIENT_SECRET"):
            SecretManager.store_secret("DS_CLIENT_SECRET", DataSphereSecrets.DS_CLIENT_SECRET)
        
        if not SecretManager.get_secret("DS_AUTH_URL"):
            SecretManager.store_secret("DS_AUTH_URL", DataSphereSecrets.DS_AUTH_URL)
        
        if not SecretManager.get_secret("DS_TOKEN_URL"):
            SecretManager.store_secret("DS_TOKEN_URL", DataSphereSecrets.DS_TOKEN_URL)
        
        if not SecretManager.get_secret("DS_API_URL"):
            SecretManager.store_secret("DS_API_URL", DataSphereSecrets.DS_API_URL)
        
        if not SecretManager.get_secret("DS_SPACE_ID"):
            SecretManager.store_secret("DS_SPACE_ID", DataSphereSecrets.DS_SPACE_ID)
            
        logger.info("DataSphere secrets initialized successfully")
    
    @staticmethod
    def get_client_credentials() -> Dict[str, str]:
        """
        Get DataSphere client credentials.
        
        Returns:
            Dict with client_id and client_secret
        """
        return {
            "client_id": SecretManager.get_secret("DS_CLIENT_KEY") or DataSphereSecrets.DS_CLIENT_KEY,
            "client_secret": SecretManager.get_secret("DS_CLIENT_SECRET") or DataSphereSecrets.DS_CLIENT_SECRET,
        }
    
    @staticmethod
    def get_oauth_urls() -> Dict[str, str]:
        """
        Get DataSphere OAuth URLs.
        
        Returns:
            Dict with auth_url and token_url
        """
        return {
            "auth_url": SecretManager.get_secret("DS_AUTH_URL") or DataSphereSecrets.DS_AUTH_URL,
            "token_url": SecretManager.get_secret("DS_TOKEN_URL") or DataSphereSecrets.DS_TOKEN_URL,
        }
    
    @staticmethod
    def get_api_url() -> str:
        """
        Get DataSphere API URL.
        
        Returns:
            API base URL
        """
        return SecretManager.get_secret("DS_API_URL") or DataSphereSecrets.DS_API_URL
    
    @staticmethod
    def get_space_id() -> str:
        """
        Get DataSphere space ID.
        
        Returns:
            Space ID
        """
        return SecretManager.get_secret("DS_SPACE_ID") or DataSphereSecrets.DS_SPACE_ID


# Initialize secrets when module is loaded
def initialize_secrets() -> None:
    """Initialize all secrets."""
    HanaSecrets.initialize()
    DataSphereSecrets.initialize()
    logger.info("All secrets initialized successfully")