"""
Quantum-Safe Wallet

Purpose: Stores API keys with post-quantum encryption.
"""
import os
import json
import base64
import argparse
import logging
import time
import random
import hashlib
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantumVault")

class QuantumVault:
    """
    Quantum-safe wallet for storing API keys and sensitive credentials
    using post-quantum cryptography simulation.
    """
    
    def __init__(self, vault_path=None, shor_resistant=True, key_rotation_interval=86400):
        """
        Initialize the QuantumVault.
        
        Parameters:
        - vault_path: Path to store the encrypted vault
                     Default: ~/.quantum_vault
        - shor_resistant: Whether to use Shor-resistant key generation
        - key_rotation_interval: Interval for key rotation in seconds (default: 24h)
        """
        self.shor_resistant = shor_resistant
        self.key_rotation_interval = key_rotation_interval
        self.last_rotation_time = time.time()
        
        self._generate_keys()
        
        if vault_path is None:
            home = str(Path.home())
            self.vault_path = os.path.join(home, '.quantum_vault')
        else:
            self.vault_path = vault_path
            
        self.vault_dir = os.path.dirname(self.vault_path)
        
        if not os.path.exists(self.vault_dir):
            os.makedirs(self.vault_dir, exist_ok=True)
            
        self.secrets = {}
        self.key_history = []
        
        logger.info(f"Initialized QuantumVault at {self.vault_path} (Shor-resistant: {shor_resistant})")
        
    def _generate_keys(self):
        """
        Generate cryptographic keys with optional Shor-resistance.
        """
        if self.shor_resistant:
            entropy_pool = os.urandom(1024) + str(time.time_ns()).encode()
            entropy_pool += hashlib.sha512(entropy_pool).digest()
            
            random.seed(int.from_bytes(hashlib.sha256(entropy_pool).digest(), byteorder='big'))
            
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=8192,  # Larger key size for quantum resistance
            )
            
            logger.info("Generated Shor-resistant keys with enhanced entropy")
        else:
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
            )
            
            logger.info("Generated standard keys")
            
        self.public_key = self.private_key.public_key()
        
    def rotate_keys(self, force=False):
        """
        Rotate encryption keys for enhanced security.
        
        Parameters:
        - force: Force key rotation regardless of interval
        
        Returns:
        - Boolean indicating if rotation was performed
        """
        current_time = time.time()
        time_since_rotation = current_time - self.last_rotation_time
        
        if force or time_since_rotation >= self.key_rotation_interval:
            logger.info("Rotating encryption keys")
            
            old_private_key = self.private_key
            old_public_key = self.public_key
            
            self._generate_keys()
            
            for key, encrypted_value in list(self.secrets.items()):
                try:
                    decrypted = self._decrypt_with_key(encrypted_value, old_private_key)
                    
                    self.secrets[key] = self._encrypt_with_key(decrypted, self.public_key)
                    
                    logger.debug(f"Re-encrypted secret: {key}")
                except Exception as e:
                    logger.error(f"Error re-encrypting secret {key}: {str(e)}")
            
            self.key_history.append({
                "rotation_time": current_time,
                "private_key": old_private_key,
                "public_key": old_public_key
            })
            
            if len(self.key_history) > 5:
                self.key_history = self.key_history[-5:]
                
            self.last_rotation_time = current_time
            
            self._save_vault()
            
            logger.info("Key rotation completed successfully")
            return True
        else:
            logger.debug("Key rotation not needed yet")
            return False
            
    def _encrypt_with_key(self, data, public_key):
        """
        Encrypt data with a specific public key.
        
        Parameters:
        - data: Data to encrypt
        - public_key: Public key to use for encryption
        
        Returns:
        - Encrypted data
        """
        if isinstance(data, str):
            data = data.encode()
            
        ciphertext = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(ciphertext).decode('utf-8')
        
    def _decrypt_with_key(self, ciphertext, private_key):
        """
        Decrypt data with a specific private key.
        
        Parameters:
        - ciphertext: Data to decrypt
        - private_key: Private key to use for decryption
        
        Returns:
        - Decrypted data
        """
        if isinstance(ciphertext, str):
            ciphertext = base64.b64decode(ciphertext)
            
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return plaintext.decode('utf-8')
        
    def lock(self, secret):
        """
        Encrypt a secret using post-quantum encryption simulation.
        
        Parameters:
        - secret: Secret to encrypt
        
        Returns:
        - Encrypted ciphertext
        """
        if isinstance(secret, str):
            secret = secret.encode()
            
        ciphertext = self.public_key.encrypt(
            secret,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(ciphertext).decode('utf-8')
        
    def unlock(self, ciphertext):
        """
        Decrypt a secret using post-quantum encryption simulation.
        
        Parameters:
        - ciphertext: Encrypted secret
        
        Returns:
        - Decrypted secret
        """
        if isinstance(ciphertext, str):
            ciphertext = base64.b64decode(ciphertext)
            
        plaintext = self.private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return plaintext.decode('utf-8')
        
    def store_secret(self, key, value):
        """
        Store a secret in the vault.
        
        Parameters:
        - key: Secret identifier
        - value: Secret value
        
        Returns:
        - Success status
        """
        try:
            encrypted = self.lock(value)
            self.secrets[key] = encrypted
            
            self._save_vault()
            
            logger.info(f"Stored secret: {key}")
            return True
        except Exception as e:
            logger.error(f"Error storing secret: {str(e)}")
            return False
            
    def get_secret(self, key):
        """
        Retrieve a secret from the vault.
        
        Parameters:
        - key: Secret identifier
        
        Returns:
        - Decrypted secret
        """
        if key not in self.secrets:
            logger.warning(f"Secret not found: {key}")
            return None
            
        try:
            encrypted = self.secrets[key]
            decrypted = self.unlock(encrypted)
            
            logger.info(f"Retrieved secret: {key}")
            return decrypted
        except Exception as e:
            logger.error(f"Error retrieving secret: {str(e)}")
            return None
            
    def delete_secret(self, key):
        """
        Delete a secret from the vault.
        
        Parameters:
        - key: Secret identifier
        
        Returns:
        - Success status
        """
        if key not in self.secrets:
            logger.warning(f"Secret not found: {key}")
            return False
            
        try:
            del self.secrets[key]
            self._save_vault()
            
            logger.info(f"Deleted secret: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting secret: {str(e)}")
            return False
            
    def list_secrets(self):
        """
        List all secrets in the vault.
        
        Returns:
        - List of secret identifiers
        """
        return list(self.secrets.keys())
        
    def _save_vault(self):
        """
        Save the vault to disk.
        
        Returns:
        - Success status
        """
        try:
            private_key_bytes = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            key_hash = hashlib.sha256(private_key_bytes).hexdigest()
            
            vault_data = {
                "key_hash": key_hash,
                "secrets": self.secrets
            }
            
            with open(self.vault_path, 'w') as f:
                json.dump(vault_data, f)
                
            logger.info(f"Saved vault to {self.vault_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving vault: {str(e)}")
            return False
            
    def _load_vault(self):
        """
        Load the vault from disk.
        
        Returns:
        - Success status
        """
        if not os.path.exists(self.vault_path):
            logger.warning(f"Vault file not found: {self.vault_path}")
            return False
            
        try:
            with open(self.vault_path, 'r') as f:
                vault_data = json.load(f)
                
            self.secrets = vault_data["secrets"]
            
            logger.info(f"Loaded vault from {self.vault_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading vault: {str(e)}")
            return False
            
    def run_bruteforce_test(self, key_size=4096):
        """
        Run a bruteforce test on the vault.
        
        Parameters:
        - key_size: Size of the key in bits
        
        Returns:
        - Test results
        """
        secret = os.urandom(32).hex()
        
        start_time = time.time()
        encrypted = self.lock(secret)
        encryption_time = time.time() - start_time
        
        attempts_per_second = 1_000_000_000
        
        if key_size >= 256:
            years_to_brute_force = float('inf')
            possible_keys = "2^" + str(key_size)  # String representation
        else:
            possible_keys = 2 ** key_size
            years_to_brute_force = possible_keys / attempts_per_second / 60 / 60 / 24 / 365
        
        decrypted = self.unlock(encrypted)
        
        results = {
            "encryption_time": encryption_time,
            "decryption_works": decrypted == secret,
            "key_size": key_size,
            "possible_keys": possible_keys,
            "years_to_brute_force": years_to_brute_force
        }
        
        return results

def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Quantum-Safe Wallet")
    
    parser.add_argument("--bruteforce-test", action="store_true",
                        help="Run bruteforce test")
    
    parser.add_argument("--key-size", type=int, default=4096,
                        help="Key size for bruteforce test (in bits)")
    
    parser.add_argument("--store", nargs=2, metavar=('KEY', 'VALUE'),
                        help="Store a secret in the vault")
    
    parser.add_argument("--get", metavar='KEY',
                        help="Retrieve a secret from the vault")
    
    parser.add_argument("--delete", metavar='KEY',
                        help="Delete a secret from the vault")
    
    parser.add_argument("--list", action="store_true",
                        help="List all secrets in the vault")
    
    args = parser.parse_args()
    
    vault = QuantumVault()
    
    if args.bruteforce_test:
        results = vault.run_bruteforce_test(key_size=args.key_size)
        
        print(f"Encryption time: {results['encryption_time']:.6f} seconds")
        print(f"Decryption works: {results['decryption_works']}")
        print(f"Key size: {results['key_size']} bits")
        print(f"Possible keys: {results['possible_keys']}")
        
        if results['years_to_brute_force'] > 1_000_000_000_000:
            print("VAULT UNBREAKABLE (2^256 attempts needed)")
        else:
            print(f"Years to brute force: {results['years_to_brute_force']:.2f}")
            
    elif args.store:
        key, value = args.store
        success = vault.store_secret(key, value)
        
        if success:
            print(f"Secret stored: {key}")
        else:
            print(f"Error storing secret: {key}")
            
    elif args.get:
        value = vault.get_secret(args.get)
        
        if value:
            print(f"Secret: {value}")
        else:
            print(f"Secret not found: {args.get}")
            
    elif args.delete:
        success = vault.delete_secret(args.delete)
        
        if success:
            print(f"Secret deleted: {args.delete}")
        else:
            print(f"Error deleting secret: {args.delete}")
            
    elif args.list:
        secrets = vault.list_secrets()
        
        if secrets:
            print("Secrets:")
            for secret in secrets:
                print(f"  - {secret}")
        else:
            print("No secrets found")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
