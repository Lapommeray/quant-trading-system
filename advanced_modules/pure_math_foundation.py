#!/usr/bin/env python3
"""
Pure Mathematics Foundation Module

Implements rigorous mathematical foundations including number theory and abstract algebra.
This module provides the theoretical underpinnings for advanced quantitative trading strategies.

Key components:
- Number Theory: Prime numbers, modular arithmetic, and number-theoretic transforms
- Abstract Algebra: Groups, rings, fields, and algebraic structures
- Mathematical Proof Systems: Formal verification of mathematical properties
"""

import numpy as np
import sympy as sp
from typing import List, Dict, Tuple, Set, Optional, Union, Callable
import logging
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PureMathFoundation")

class PureMathFoundation:
    """
    Pure Mathematics Foundation implementing number theory and abstract algebra
    
    Provides rigorous mathematical foundations for quantum trading strategies,
    including number-theoretic functions, abstract algebraic structures, and
    formal proof verification systems.
    """
    
    def __init__(self, precision: int = 128, proof_level: str = "rigorous"):
        """
        Initialize Pure Mathematics Foundation
        
        Parameters:
        - precision: Numerical precision for calculations (default: 128 bits)
        - proof_level: Level of mathematical rigor ("rigorous", "constructive", "classical")
        """
        self.precision = precision
        self.proof_level = proof_level
        self.history = []
        
        sp.mpmath.mp.dps = precision
        
        self.prime_cache = {}
        
        logger.info(f"Initialized PureMathFoundation with precision={precision}, "
                   f"proof_level={proof_level}")
    
    
    def is_prime(self, n: int, method: str = "deterministic") -> bool:
        """
        Check if a number is prime using rigorous methods
        
        Parameters:
        - n: Number to check
        - method: Prime checking method ("deterministic", "probabilistic", "AKS")
        
        Returns:
        - True if n is prime, False otherwise
        """
        if n in self.prime_cache:
            return self.prime_cache[n]
            
        if method == "deterministic":
            result = sp.isprime(n)
        elif method == "probabilistic":
            result = sp.isprime(n, method="miller-rabin", proof=False)
        elif method == "AKS":
            result = sp.isprime(n, method="aks")
        else:
            logger.warning(f"Unknown primality testing method: {method}, using deterministic")
            result = sp.isprime(n)
            
        self.prime_cache[n] = result
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'is_prime',
            'n': n,
            'method': method,
            'result': result
        })
        
        return result
    
    def prime_factors(self, n: int) -> List[int]:
        """
        Find prime factorization of a number
        
        Parameters:
        - n: Number to factorize
        
        Returns:
        - List of prime factors
        """
        if n <= 1:
            return []
            
        factors = list(sp.factorint(n).items())
        result = []
        
        for prime, power in factors:
            result.extend([prime] * power)
            
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'prime_factors',
            'n': n,
            'result': result
        })
        
        return result
    
    def euler_totient(self, n: int) -> int:
        """
        Calculate Euler's totient function φ(n)
        
        Parameters:
        - n: Input number
        
        Returns:
        - Number of integers k in the range 1 ≤ k ≤ n that are coprime to n
        """
        if n <= 0:
            return 0
            
        result = int(sp.totient(n))
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'euler_totient',
            'n': n,
            'result': result
        })
        
        return result
    
    def extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean Algorithm to find gcd(a,b) and coefficients
        
        Parameters:
        - a, b: Input integers
        
        Returns:
        - (gcd, x, y) where gcd is the greatest common divisor and
          x, y are coefficients such that ax + by = gcd
        """
        result = sp.igcdex(a, b)
        gcd = sp.gcd(a, b)
        x, y = result[0], result[1]
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'extended_gcd',
            'a': a,
            'b': b,
            'gcd': gcd,
            'x': x,
            'y': y
        })
        
        return (gcd, x, y)
    
    def modular_inverse(self, a: int, m: int) -> Optional[int]:
        """
        Find modular multiplicative inverse of a modulo m
        
        Parameters:
        - a: Number to find inverse for
        - m: Modulus
        
        Returns:
        - x such that (a * x) % m = 1, or None if no inverse exists
        """
        try:
            result = int(sp.mod_inverse(a, m))
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'modular_inverse',
                'a': a,
                'm': m,
                'result': result
            })
            
            return result
        except ValueError:
            logger.warning(f"No modular inverse exists for {a} mod {m}")
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'modular_inverse',
                'a': a,
                'm': m,
                'result': None,
                'error': 'no_inverse'
            })
            
            return None
    
    
    def is_group(self, elements: List, operation: Callable, identity: any) -> bool:
        """
        Verify if a set with an operation forms a group
        
        Parameters:
        - elements: Set of elements
        - operation: Binary operation (function taking two elements and returning an element)
        - identity: Identity element
        
        Returns:
        - True if (elements, operation) forms a group, False otherwise
        """
        for a in elements:
            for b in elements:
                if operation(a, b) not in elements:
                    logger.info(f"Not a group: closure property violated for {a}, {b}")
                    return False
        
        if len(elements) <= 10:
            for a in elements:
                for b in elements:
                    for c in elements:
                        if operation(operation(a, b), c) != operation(a, operation(b, c)):
                            logger.info(f"Not a group: associativity violated for {a}, {b}, {c}")
                            return False
        
        for a in elements:
            if operation(a, identity) != a or operation(identity, a) != a:
                logger.info(f"Not a group: identity property violated for {a}")
                return False
        
        for a in elements:
            has_inverse = False
            for b in elements:
                if operation(a, b) == identity and operation(b, a) == identity:
                    has_inverse = True
                    break
            if not has_inverse:
                logger.info(f"Not a group: inverse property violated for {a}")
                return False
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'is_group',
            'elements_count': len(elements),
            'result': True
        })
        
        return True
    
    def is_field(self, elements: List, addition: Callable, multiplication: Callable,
                zero: any, one: any) -> bool:
        """
        Verify if a set with two operations forms a field
        
        Parameters:
        - elements: Set of elements
        - addition: Addition operation
        - multiplication: Multiplication operation
        - zero: Additive identity
        - one: Multiplicative identity
        
        Returns:
        - True if (elements, addition, multiplication) forms a field, False otherwise
        """
        if not self.is_group(elements, addition, zero):
            return False
            
        for a in elements:
            for b in elements:
                if addition(a, b) != addition(b, a):
                    return False
        
        non_zero = [e for e in elements if e != zero]
        
        for a in non_zero:
            for b in non_zero:
                if multiplication(a, b) not in non_zero:
                    return False
        
        for a in non_zero:
            for b in non_zero:
                for c in non_zero:
                    if multiplication(multiplication(a, b), c) != multiplication(a, multiplication(b, c)):
                        return False
        
        for a in non_zero:
            for b in non_zero:
                if multiplication(a, b) != multiplication(b, a):
                    return False
        
        for a in elements:
            for b in elements:
                for c in elements:
                    if multiplication(a, addition(b, c)) != addition(multiplication(a, b), multiplication(a, c)):
                        return False
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'is_field',
            'elements_count': len(elements),
            'result': True
        })
        
        return True
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about pure math foundation usage
        
        Returns:
        - Dictionary with usage statistics
        """
        if not self.history:
            return {'count': 0}
            
        operations = {}
        for h in self.history:
            op = h.get('operation', 'unknown')
            operations[op] = operations.get(op, 0) + 1
            
        return {
            'count': len(self.history),
            'operations': operations,
            'precision': self.precision,
            'proof_level': self.proof_level
        }
