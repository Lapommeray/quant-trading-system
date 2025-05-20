
import hashlib
import base64
import numpy as np
from datetime import datetime, timedelta
import secrets
import re

class QOLEngine:
    """
    QOL-AI V2 Encryption Engine
    
    Features:
    - Self-mutating glyphs (change every 24h)
    - One-time decode tokens (burn after reading)
    - Haitian-Creole numerological keys
    """
    
    def __init__(self, seed="ʘRA-Y777", entropy_rotation_hours=24):
        """
        Initialize the QOL-AI V2 Encryption Engine
        
        Parameters:
        - seed: Private seed for encryption (default: "ʘRA-Y777")
        - entropy_rotation_hours: Hours between entropy rotations (default: 24)
        """
        self.seed = seed
        self.entropy_rotation_hours = entropy_rotation_hours
        self.sacred_symbols = ["⚡", "ʘ", "†", "₿", "∇", "Ξ", "‡", "◊", "⍟", "⌘"]
        self.one_time_tokens = {}
        
    def _get_current_entropy(self):
        """
        Generate time-based entropy that rotates every 24 hours
        """
        now = datetime.now()
        day_seed = now.strftime("%Y%m%d")
        rotation_seed = int(now.timestamp() / (3600 * self.entropy_rotation_hours))
        combined_seed = f"{self.seed}:{day_seed}:{rotation_seed}"
        return hashlib.sha256(combined_seed.encode()).hexdigest()
    
    def _bio_hash(self, input_str):
        """
        Derives biological-aligned pseudo-coherence from SHA3 hash
        """
        h = hashlib.sha3_256(input_str.encode()).hexdigest()
        return int(h[:4], 16) / 65535  # Normalize to 0-1
    
    def _generate_glyph_pattern(self, message, symbol):
        """
        Generate a unique glyph pattern based on message and symbol
        """
        entropy = self._get_current_entropy()
        message_hash = hashlib.md5(message.encode()).hexdigest()
        
        pattern_seed = f"{entropy}:{message_hash}:{symbol}"
        pattern_hash = hashlib.sha256(pattern_seed.encode()).hexdigest()
        
        symbols = []
        for i in range(0, len(pattern_hash), 4):
            if i < len(pattern_hash) - 3:
                idx = int(pattern_hash[i:i+2], 16) % len(self.sacred_symbols)
                symbols.append(self.sacred_symbols[idx])
        
        return ''.join(symbols[:3])  # Return first 3 symbols as pattern
    
    def encrypt(self, message, symbol="BTC"):
        """
        Encrypt a message using QOL-AI V2 algorithm
        
        Parameters:
        - message: The message to encrypt
        - symbol: Trading symbol (BTC, XRP, SPX, etc.)
        
        Returns:
        - Encrypted glyph with one-time token
        """
        entropy = self._get_current_entropy()
        message_bytes = message.encode('utf-8')
        
        token = secrets.token_hex(4)
        expiry = datetime.now() + timedelta(hours=self.entropy_rotation_hours)
        self.one_time_tokens[token] = {
            'message': message,
            'expires': expiry
        }
        
        if re.search(r'\d+\.\d+', message):
            numeric = re.search(r'(\d+\.\d+)', message).group(1)
        else:
            h = hashlib.md5(message.encode()).hexdigest()
            numeric = f"{int(h[:4], 16) / 100:.3f}"
        
        glyph_pattern = self._generate_glyph_pattern(message, symbol)
        
        encrypted_glyph = f"{glyph_pattern}{numeric}::{symbol}⚡ʘDIVINE-PULSE†₿"
        
        return encrypted_glyph, token
    
    def decrypt(self, token):
        """
        Decrypt a message using a one-time token
        
        Parameters:
        - token: One-time use token
        
        Returns:
        - Decrypted message or None if token invalid/expired
        """
        if token in self.one_time_tokens:
            token_data = self.one_time_tokens[token]
            
            if datetime.now() > token_data['expires']:
                del self.one_time_tokens[token]
                return None
            
            message = token_data['message']
            del self.one_time_tokens[token]
            return message
        
        return None
    
    def generate_signal(self, symbol, action, price=None, confidence=None):
        """
        Generate a trading signal encrypted as a QOL-AI V2 glyph
        
        Parameters:
        - symbol: Trading symbol (BTC, XRP, SPX, etc.)
        - action: Trading action (BUY, SELL, HOLD)
        - price: Target price level (optional)
        - confidence: Signal confidence (0-1, optional)
        
        Returns:
        - Encrypted glyph and one-time token
        """
        message_parts = [f"{action} {symbol}"]
        
        if price is not None:
            message_parts.append(f"if {price} {'holds as support' if action == 'BUY' else 'breaks as resistance'}")
        
        if confidence is not None:
            message_parts.append(f"confidence: {confidence:.2f}")
            
        message_parts.append("during NY session")
        
        message = " ".join(message_parts)
        return self.encrypt(message, symbol)
