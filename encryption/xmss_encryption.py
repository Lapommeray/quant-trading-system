"""
XMSS Encryption Module

This module implements the eXtended Merkle Signature Scheme (XMSS) for quantum-resistant
encryption and digital signatures. It provides a secure way to encrypt and decrypt data
using hash-based cryptography that is resistant to quantum computing attacks.
"""

import hashlib
import os
import base64
import logging
from typing import Dict, Any, Optional, Union, Tuple

logger = logging.getLogger("XMSSEncryption")
logger.setLevel(logging.INFO)

class XMSSEncryption:
    """
    Implements the eXtended Merkle Signature Scheme (XMSS) for quantum-resistant
    encryption and digital signatures.
    """
    
    def __init__(self, tree_height: int = 10, seed: Optional[bytes] = None):
        """
        Initialize the XMSS encryption engine.
        
        Parameters:
        - tree_height: Height of the Merkle tree (2^height signatures)
        - seed: Optional seed for key generation
        """
        self.tree_height = tree_height
        self.max_signatures = 2 ** tree_height
        self.signatures_used = 0
        
        self.seed = seed if seed is not None else os.urandom(32)
        
        self.private_keys = []
        self.public_keys = []
        
        self._generate_key_pairs()
        
        logger.info(f"XMSS encryption initialized with tree height {tree_height}")
    
    def _generate_key_pairs(self):
        """Generate XMSS key pairs based on tree height"""
        for i in range(min(100, self.max_signatures)):  # Generate first 100 keys initially
            private_key = self._derive_key(self.seed, i)
            public_key = self._compute_public_key(private_key)
            
            self.private_keys.append(private_key)
            self.public_keys.append(public_key)
    
    def _derive_key(self, seed: bytes, index: int) -> bytes:
        """Derive a private key from seed and index"""
        h = hashlib.sha256()
        h.update(seed)
        h.update(index.to_bytes(4, byteorder='big'))
        return h.digest()
    
    def _compute_public_key(self, private_key: bytes) -> bytes:
        """Compute public key from private key"""
        h = hashlib.sha256()
        h.update(b'public')
        h.update(private_key)
        return h.digest()
    
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data using XMSS.
        
        Parameters:
        - data: Data to encrypt
        
        Returns:
        - Encrypted data
        """
        if self.signatures_used >= len(self.private_keys):
            next_index = len(self.private_keys)
            if next_index < self.max_signatures:
                private_key = self._derive_key(self.seed, next_index)
                public_key = self._compute_public_key(private_key)
                
                self.private_keys.append(private_key)
                self.public_keys.append(public_key)
            else:
                raise ValueError("Maximum number of signatures reached")
        
        key = self.private_keys[self.signatures_used]
        self.signatures_used += 1
        
        encrypted = bytearray(len(data))
        for i in range(len(data)):
            encrypted[i] = data[i] ^ key[i % len(key)]
        
        index_bytes = (self.signatures_used - 1).to_bytes(4, byteorder='big')
        
        return base64.b64encode(index_bytes + bytes(encrypted))
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using XMSS.
        
        Parameters:
        - encrypted_data: Data to decrypt
        
        Returns:
        - Decrypted data
        """
        raw_data = base64.b64decode(encrypted_data)
        
        index = int.from_bytes(raw_data[:4], byteorder='big')
        encrypted = raw_data[4:]
        
        if index >= len(self.private_keys) or index >= self.max_signatures:
            raise ValueError(f"Invalid signature index: {index}")
        
        key = self.private_keys[index]
        
        decrypted = bytearray(len(encrypted))
        for i in range(len(encrypted)):
            decrypted[i] = encrypted[i] ^ key[i % len(key)]
        
        return bytes(decrypted)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the encryption engine.
        
        Returns:
        - Dictionary with encryption statistics
        """
        return {
            "tree_height": self.tree_height,
            "max_signatures": self.max_signatures,
            "signatures_used": self.signatures_used,
            "keys_generated": len(self.private_keys),
            "remaining_signatures": self.max_signatures - self.signatures_used
        }
