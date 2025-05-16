"""
API Vault - Secure API Key Management

This module provides secure storage and access to exchange API keys
using encryption and environment variables.
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class APIVault:
    """
    Secure storage and management of API keys for exchanges.
    
    This class handles encryption, decryption, and secure access to API keys
    for various exchanges. It uses Fernet symmetric encryption and stores
    keys securely, never exposing them in plaintext in the codebase.
    """
    
    def __init__(self, key_path: Optional[str] = None, env_prefix: str = "EXCHANGE_"):
        """
        Initialize the API Vault.
        
        Parameters:
        - key_path: Path to the encryption key file. If None, will use environment variable.
        - env_prefix: Prefix for environment variables containing API keys.
        """
        self.env_prefix = env_prefix
        load_dotenv()
        
        key = os.getenv("API_VAULT_KEY")
        if not key and key_path:
            try:
                with open(key_path, "rb") as f:
                    key = f.read()
            except FileNotFoundError:
                logger.warning(f"Key file not found at {key_path}. Generating new key.")
                key = Fernet.generate_key()
                self._save_key(key, key_path)
        elif not key:
            logger.warning("No encryption key found. Generating new key.")
            key = Fernet.generate_key()
            if key_path:
                self._save_key(key, key_path)
        
        self.cipher = Fernet(key)
        self.vault_path = os.getenv("API_VAULT_PATH", "secure/api_keys.vault")
        self.keys = self._load_keys()
    
    def _save_key(self, key: bytes, key_path: str) -> None:
        """
        Save encryption key to file.
        
        Parameters:
        - key: Encryption key bytes
        - key_path: Path to save the key
        """
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, "wb") as f:
            f.write(key)
        logger.info(f"Encryption key saved to {key_path}")
    
    def _load_keys(self) -> Dict[str, Dict[str, str]]:
        """
        Load encrypted API keys from vault file.
        
        Returns:
        - Dictionary of exchange API keys
        """
        if not os.path.exists(self.vault_path):
            logger.info(f"Vault file not found at {self.vault_path}. Creating new vault.")
            return {}
        
        try:
            with open(self.vault_path, "rb") as f:
                encrypted_data = f.read()
            
            if not encrypted_data:
                return {}
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data)
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            return {}
    
    def _save_keys(self) -> None:
        """Save encrypted API keys to vault file."""
        try:
            encrypted_data = self.cipher.encrypt(json.dumps(self.keys).encode())
            os.makedirs(os.path.dirname(self.vault_path), exist_ok=True)
            with open(self.vault_path, "wb") as f:
                f.write(encrypted_data)
            logger.info(f"API keys saved to {self.vault_path}")
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
    
    def get_credentials(self, exchange: str) -> Dict[str, str]:
        """
        Get API credentials for an exchange.
        
        Parameters:
        - exchange: Name of the exchange
        
        Returns:
        - Dictionary with API credentials
        """
        api_key = os.getenv(f"{self.env_prefix}{exchange.upper()}_API_KEY")
        secret = os.getenv(f"{self.env_prefix}{exchange.upper()}_SECRET")
        
        if api_key and secret:
            return {"apiKey": api_key, "secret": secret}
        
        if exchange in self.keys:
            return self.keys[exchange]
        
        logger.warning(f"No API credentials found for {exchange}")
        return {}
    
    def set_credentials(self, exchange: str, api_key: str, secret: str, 
                        additional_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Set API credentials for an exchange.
        
        Parameters:
        - exchange: Name of the exchange
        - api_key: API key
        - secret: API secret
        - additional_params: Additional parameters for the exchange
        """
        creds = {"apiKey": api_key, "secret": secret}
        if additional_params:
            creds.update(additional_params)
        
        self.keys[exchange] = creds
        self._save_keys()
        logger.info(f"API credentials set for {exchange}")
    
    def remove_credentials(self, exchange: str) -> bool:
        """
        Remove API credentials for an exchange.
        
        Parameters:
        - exchange: Name of the exchange
        
        Returns:
        - True if credentials were removed, False otherwise
        """
        if exchange in self.keys:
            del self.keys[exchange]
            self._save_keys()
            logger.info(f"API credentials removed for {exchange}")
            return True
        return False
    
    def list_exchanges(self) -> list:
        """
        List all exchanges with stored credentials.
        
        Returns:
        - List of exchange names
        """
        env_exchanges = [
            key.replace(self.env_prefix, "").lower().split("_")[0]
            for key in os.environ
            if key.startswith(self.env_prefix) and key.endswith("_API_KEY")
        ]
        
        vault_exchanges = list(self.keys.keys())
        
        return list(set(env_exchanges + vault_exchanges))
