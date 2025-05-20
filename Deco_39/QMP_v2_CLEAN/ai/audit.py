"""
Blockchain-Verified Audit Trail Module

This module implements the Blockchain-Verified Audit Trail for the QMP Overrider system.
It provides tamper-proof verification of trading events and post-mortem analysis using
Merkle root hashing and Ethereum Smart Contract storage.
"""

import hashlib
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import os

try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

class MerkleTree:
    """
    Merkle Tree implementation for tamper-proof audit trails.
    
    This class implements a Merkle Tree for creating cryptographic proofs
    of trading events and system actions.
    """
    
    def __init__(self, hash_function=hashlib.sha256):
        """
        Initialize the Merkle Tree.
        
        Parameters:
            hash_function: Hash function to use for the Merkle Tree
        """
        self.hash_function = hash_function
        self.leaves = []
        self.levels = None
        self.is_ready = False
    
    def add_leaf(self, data, do_hash=True):
        """
        Add a leaf to the Merkle Tree.
        
        Parameters:
            data: Data to add as a leaf
            do_hash: Whether to hash the data before adding
        """
        if do_hash:
            if isinstance(data, str):
                data = data.encode('utf-8')
            elif isinstance(data, dict) or isinstance(data, list):
                data = json.dumps(data, sort_keys=True).encode('utf-8')
            
            leaf = self.hash_function(data).hexdigest()
        else:
            leaf = data
        
        self.leaves.append(leaf)
        self.is_ready = False
    
    def _calculate_next_level(self):
        """Calculate the next level of the Merkle Tree."""
        solo_leave = None
        n = len(self.levels[0])
        
        if n % 2 == 1:
            solo_leave = self.levels[0][-1]
            n -= 1
        
        new_level = []
        for i in range(0, n, 2):
            data = self.levels[0][i] + self.levels[0][i+1]
            if isinstance(data, str):
                data = data.encode('utf-8')
            new_level.append(self.hash_function(data).hexdigest())
        
        if solo_leave is not None:
            new_level.append(solo_leave)
        
        self.levels = [new_level] + self.levels
    
    def make_tree(self):
        """
        Make the Merkle Tree.
        
        Returns:
            Root hash of the Merkle Tree
        """
        if self.is_ready:
            return self.get_merkle_root()
        
        if len(self.leaves) > 0:
            self.levels = [self.leaves]
            while len(self.levels[0]) > 1:
                self._calculate_next_level()
        else:
            self.levels = [['']]
        
        self.is_ready = True
        return self.get_merkle_root()
    
    def get_merkle_root(self):
        """
        Get the Merkle Root.
        
        Returns:
            Merkle Root hash
        """
        if not self.is_ready or self.levels is None:
            self.make_tree()
        
        if self.levels is not None and len(self.levels) > 0:
            return self.levels[0][0]
        
        return None
    
    def get_proof(self, index):
        """
        Get the Merkle Proof for a leaf.
        
        Parameters:
            index: Index of the leaf
            
        Returns:
            Merkle Proof
        """
        if not self.is_ready or self.levels is None:
            self.make_tree()
        
        if index < 0 or index >= len(self.leaves):
            return None
        
        proof = []
        for level in self.levels[::-1]:
            if index % 2 == 0:
                if index + 1 < len(level):
                    proof.append({'position': 'right', 'data': level[index + 1]})
            else:
                proof.append({'position': 'left', 'data': level[index - 1]})
            
            index = index // 2
        
        return proof
    
    def validate_proof(self, proof, target_hash, merkle_root):
        """
        Validate a Merkle Proof.
        
        Parameters:
            proof: Merkle Proof
            target_hash: Target hash to validate
            merkle_root: Merkle Root to validate against
            
        Returns:
            True if the proof is valid, False otherwise
        """
        if not proof or not target_hash or not merkle_root:
            return False
        
        current = target_hash
        
        for p in proof:
            if p['position'] == 'left':
                data = p['data'] + current
            else:
                data = current + p['data']
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            current = self.hash_function(data).hexdigest()
        
        return current == merkle_root

class BlockchainAudit:
    """
    Blockchain-Verified Audit Trail for the QMP Overrider system.
    
    This class provides tamper-proof verification of trading events and post-mortem analysis
    using Merkle root hashing and Ethereum Smart Contract storage.
    """
    
    def __init__(self, blockchain_enabled=True, infura_url=None, contract_address=None, private_key=None):
        """
        Initialize the Blockchain Audit system.
        
        Parameters:
            blockchain_enabled: Whether to enable blockchain verification
            infura_url: Infura URL for Ethereum connection
            contract_address: Address of the audit contract
            private_key: Private key for transaction signing
        """
        self.logger = logging.getLogger("BlockchainAudit")
        
        self.blockchain_enabled = blockchain_enabled and WEB3_AVAILABLE
        self.infura_url = infura_url
        self.contract_address = contract_address
        self.private_key = private_key
        
        self.log_dir = Path("logs/blockchain")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.merkle_tree = MerkleTree()
        self.events = []
        self.roots = []
        
        if self.blockchain_enabled:
            try:
                if self.infura_url:
                    self.w3 = Web3(Web3.HTTPProvider(self.infura_url))
                    self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                    
                    if self.w3.is_connected():
                        self.logger.info(f"Connected to Ethereum node: {self.infura_url}")
                        
                        abi_path = Path(__file__).parent / "audit_contract_abi.json"
                        if abi_path.exists():
                            with open(abi_path, "r") as f:
                                contract_abi = json.load(f)
                            
                            if self.contract_address:
                                self.contract = self.w3.eth.contract(
                                    address=self.contract_address,
                                    abi=contract_abi
                                )
                                self.logger.info(f"Loaded audit contract at {self.contract_address}")
                            else:
                                self.logger.warning("No contract address provided")
                                self.blockchain_enabled = False
                        else:
                            self.logger.warning(f"Contract ABI not found at {abi_path}")
                            self.blockchain_enabled = False
                    else:
                        self.logger.warning(f"Failed to connect to Ethereum node: {self.infura_url}")
                        self.blockchain_enabled = False
                else:
                    self.logger.warning("No Infura URL provided")
                    self.blockchain_enabled = False
            except Exception as e:
                self.logger.error(f"Error initializing blockchain connection: {e}")
                self.blockchain_enabled = False
        
        self.logger.info(f"Blockchain Audit initialized (blockchain_enabled={self.blockchain_enabled})")
    
    def log_event(self, event_data):
        """
        Log an event to the audit trail.
        
        Parameters:
            event_data: Event data to log
            
        Returns:
            Transaction hash if blockchain is enabled, None otherwise
        """
        if isinstance(event_data, dict) and 'timestamp' not in event_data:
            event_data['timestamp'] = datetime.now().isoformat()
        
        self.events.append(event_data)
        
        self.merkle_tree.add_leaf(event_data)
        
        merkle_root = self.merkle_tree.get_merkle_root()
        
        self._save_event(event_data, merkle_root)
        
        tx_hash = None
        if self.blockchain_enabled:
            try:
                tx_hash = self._submit_to_blockchain(merkle_root)
                self.logger.info(f"Event logged to blockchain: {tx_hash}")
            except Exception as e:
                self.logger.error(f"Error submitting to blockchain: {e}")
        
        return tx_hash
    
    def _save_event(self, event_data, merkle_root):
        """
        Save an event to a local file.
        
        Parameters:
            event_data: Event data to save
            merkle_root: Merkle Root of the event
        """
        try:
            event_record = {
                'data': event_data,
                'merkle_root': merkle_root,
                'timestamp': datetime.now().isoformat()
            }
            
            event_file = self.log_dir / f"event_{int(time.time())}.json"
            with open(event_file, "w") as f:
                json.dump(event_record, f, indent=2)
            
            roots_file = self.log_dir / "merkle_roots.json"
            self.roots.append({
                'root': merkle_root,
                'timestamp': datetime.now().isoformat(),
                'event_count': len(self.events)
            })
            
            with open(roots_file, "w") as f:
                json.dump(self.roots, f, indent=2)
            
            self.logger.debug(f"Event saved to {event_file}")
        except Exception as e:
            self.logger.error(f"Error saving event: {e}")
    
    def _submit_to_blockchain(self, merkle_root):
        """
        Submit a Merkle Root to the blockchain.
        
        Parameters:
            merkle_root: Merkle Root to submit
            
        Returns:
            Transaction hash
        """
        if not self.blockchain_enabled:
            return None
        
        try:
            merkle_root_bytes = bytes.fromhex(merkle_root)
            
            nonce = self.w3.eth.get_transaction_count(self.w3.eth.account.from_key(self.private_key).address)
            
            tx = self.contract.functions.storeRoot(merkle_root_bytes).build_transaction({
                'chainId': 1,  # Ethereum mainnet
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return tx_hash.hex()
        except Exception as e:
            self.logger.error(f"Error submitting to blockchain: {e}")
            return None
    
    def verify_event(self, event_data, merkle_root=None):
        """
        Verify an event against the audit trail.
        
        Parameters:
            event_data: Event data to verify
            merkle_root: Merkle Root to verify against (or None to use latest)
            
        Returns:
            True if the event is verified, False otherwise
        """
        if merkle_root is None:
            if not self.roots:
                return False
            merkle_root = self.roots[-1]['root']
        
        verification_tree = MerkleTree()
        for event in self.events:
            verification_tree.add_leaf(event)
        
        calculated_root = verification_tree.make_tree()
        
        if calculated_root != merkle_root:
            self.logger.warning(f"Merkle Root mismatch: {calculated_root} != {merkle_root}")
            return False
        
        event_index = None
        for i, event in enumerate(self.events):
            if event == event_data:
                event_index = i
                break
        
        if event_index is None:
            self.logger.warning("Event not found in audit trail")
            return False
        
        proof = verification_tree.get_proof(event_index)
        
        leaf_hash = hashlib.sha256(json.dumps(event_data, sort_keys=True).encode('utf-8')).hexdigest()
        
        is_valid = verification_tree.validate_proof(proof, leaf_hash, merkle_root)
        
        if not is_valid:
            self.logger.warning("Event proof validation failed")
        
        return is_valid
    
    def verify_blockchain_root(self, merkle_root):
        """
        Verify a Merkle Root against the blockchain.
        
        Parameters:
            merkle_root: Merkle Root to verify
            
        Returns:
            True if the root is verified, False otherwise
        """
        if not self.blockchain_enabled:
            self.logger.warning("Blockchain verification not enabled")
            return False
        
        try:
            merkle_root_bytes = bytes.fromhex(merkle_root)
            
            is_valid = self.contract.functions.verifyRoot(merkle_root_bytes).call()
            
            if not is_valid:
                self.logger.warning(f"Blockchain verification failed for root: {merkle_root}")
            
            return is_valid
        except Exception as e:
            self.logger.error(f"Error verifying blockchain root: {e}")
            return False
    
    def generate_audit_report(self, start_time=None, end_time=None):
        """
        Generate an audit report for a time period.
        
        Parameters:
            start_time: Start time for the report (or None for all time)
            end_time: End time for the report (or None for now)
            
        Returns:
            Audit report as a string
        """
        if start_time is None:
            start_time = datetime.fromtimestamp(0)
        elif isinstance(start_time, (int, float)):
            start_time = datetime.fromtimestamp(start_time)
        
        if end_time is None:
            end_time = datetime.now()
        elif isinstance(end_time, (int, float)):
            end_time = datetime.fromtimestamp(end_time)
        
        filtered_events = []
        for event in self.events:
            event_time = None
            if isinstance(event, dict) and 'timestamp' in event:
                try:
                    event_time = datetime.fromisoformat(event['timestamp'])
                except (ValueError, TypeError):
                    pass
            
            if event_time is None:
                continue
            
            if start_time <= event_time <= end_time:
                filtered_events.append(event)
        
        report = f"""
BLOCKCHAIN AUDIT REPORT
======================
Period: {start_time.isoformat()} to {end_time.isoformat()}
Events: {len(filtered_events)}
Blockchain Verification: {'Enabled' if self.blockchain_enabled else 'Disabled'}

Event Summary:
"""
        
        for i, event in enumerate(filtered_events):
            report += f"\n{i+1}. {event.get('type', 'Unknown')} at {event.get('timestamp', 'Unknown')}"
            if 'details' in event:
                report += f"\n   Details: {event['details']}"
        
        report += f"\n\nVerification Status: "
        
        verification_tree = MerkleTree()
        for event in filtered_events:
            verification_tree.add_leaf(event)
        
        calculated_root = verification_tree.make_tree()
        
        report += f"\nMerkle Root: {calculated_root}"
        
        if self.blockchain_enabled:
            is_valid = self.verify_blockchain_root(calculated_root)
            report += f"\nBlockchain Verification: {'Valid' if is_valid else 'Invalid'}"
        
        report += f"\n\nReport generated at: {datetime.now().isoformat()}"
        
        return report
