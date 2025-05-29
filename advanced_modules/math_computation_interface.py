#!/usr/bin/env python3
"""
Mathematical Computation Interface Module

Provides interfaces to powerful mathematical computation tools including:
- SymPy for symbolic mathematics and equation solving
- Mathematica for advanced mathematical computations
- MATLAB for numerical analysis and simulation

This module enables advanced equation solving beyond financial models,
supporting the quantum trading system with rigorous mathematical capabilities.
"""

import numpy as np
import sympy as sp
import mpmath
from sympy import symbols, solve, simplify, expand, factor, integrate, diff, Matrix
import subprocess
import tempfile
import os
import platform
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MathComputationInterface")

class MathComputationInterface:
    """
    Mathematical Computation Interface for advanced equation solving
    
    Provides unified access to powerful mathematical tools:
    - SymPy for symbolic mathematics
    - Mathematica for advanced computations (via subprocess)
    - MATLAB for numerical analysis (via subprocess)
    
    Enables solving complex equations beyond traditional financial models.
    """
    
    def __init__(self, precision: int = 64, use_mathematica: bool = False, 
                 use_matlab: bool = False, mathematica_path: str = None, 
                 matlab_path: str = None):
        """
        Initialize Mathematical Computation Interface
        
        Parameters:
        - precision: Numerical precision for calculations (default: 64 digits)
        - use_mathematica: Whether to enable Mathematica interface (default: False)
        - use_matlab: Whether to enable MATLAB interface (default: False)
        - mathematica_path: Path to Mathematica executable (default: auto-detect)
        - matlab_path: Path to MATLAB executable (default: auto-detect)
        """
        self.precision = precision
        self.use_mathematica = use_mathematica
        self.use_matlab = use_matlab
        self.history = []
        
        mpmath.mp.dps = precision
        
        if use_mathematica and mathematica_path is None:
            self.mathematica_path = self._detect_mathematica_path()
        else:
            self.mathematica_path = mathematica_path
            
        if use_matlab and matlab_path is None:
            self.matlab_path = self._detect_matlab_path()
        else:
            self.matlab_path = matlab_path
            
        self.mathematica_available = self._check_mathematica() if use_mathematica else False
        self.matlab_available = self._check_matlab() if use_matlab else False
        
        logger.info(f"Initialized MathComputationInterface with precision={precision}")
        logger.info(f"Mathematica available: {self.mathematica_available}")
        logger.info(f"MATLAB available: {self.matlab_available}")
    
    def _detect_mathematica_path(self) -> Optional[str]:
        """Detect Mathematica installation path"""
        system = platform.system()
        
        if system == "Windows":
            paths = [
                r"C:\Program Files\Wolfram Research\Mathematica\12.0\math.exe",
                r"C:\Program Files\Wolfram Research\Mathematica\11.3\math.exe"
            ]
        elif system == "Darwin":  # macOS
            paths = [
                "/Applications/Mathematica.app/Contents/MacOS/MathKernel",
                "/Applications/Mathematica.app/Contents/MacOS/WolframKernel"
            ]
        else:  # Linux
            paths = [
                "/usr/local/bin/math",
                "/usr/bin/math",
                "/opt/Wolfram/Mathematica/12.0/Executables/math",
                "/opt/Wolfram/Mathematica/11.3/Executables/math"
            ]
            
        for path in paths:
            if os.path.exists(path):
                return path
                
        return None
    
    def _detect_matlab_path(self) -> Optional[str]:
        """Detect MATLAB installation path"""
        system = platform.system()
        
        if system == "Windows":
            paths = [
                r"C:\Program Files\MATLAB\R2023a\bin\matlab.exe",
                r"C:\Program Files\MATLAB\R2022b\bin\matlab.exe"
            ]
        elif system == "Darwin":  # macOS
            paths = [
                "/Applications/MATLAB_R2023a.app/bin/matlab",
                "/Applications/MATLAB_R2022b.app/bin/matlab"
            ]
        else:  # Linux
            paths = [
                "/usr/local/MATLAB/R2023a/bin/matlab",
                "/usr/local/MATLAB/R2022b/bin/matlab",
                "/opt/matlab/bin/matlab"
            ]
            
        for path in paths:
            if os.path.exists(path):
                return path
                
        return None
    
    def _check_mathematica(self) -> bool:
        """Check if Mathematica is available"""
        if not self.mathematica_path:
            return False
            
        try:
            result = subprocess.run(
                [self.mathematica_path, "-run", "Print[2+2]; Exit[]"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "4" in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Mathematica not available or not working properly")
            return False
    
    def _check_matlab(self) -> bool:
        """Check if MATLAB is available"""
        if not self.matlab_path:
            return False
            
        try:
            result = subprocess.run(
                [self.matlab_path, "-batch", "disp(2+2);"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return "4" in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("MATLAB not available or not working properly")
            return False
    
    
    def solve_equation_sympy(self, equation_str: str, variable_str: str = 'x') -> List:
        """
        Solve equation using SymPy
        
        Parameters:
        - equation_str: String representation of the equation (e.g., "x**2 - 4 = 0")
        - variable_str: Variable to solve for (default: 'x')
        
        Returns:
        - List of solutions
        """
        try:
            if "=" in equation_str:
                lhs_str, rhs_str = equation_str.split("=")
                lhs = sp.sympify(lhs_str.strip())
                rhs = sp.sympify(rhs_str.strip())
                equation = sp.Eq(lhs, rhs)
            else:
                equation = sp.Eq(sp.sympify(equation_str), 0)
                
            variable = sp.symbols(variable_str)
            
            solutions = sp.solve(equation, variable)
            
            result = [float(sol) if sp.im(sol) == 0 else complex(sol) for sol in solutions]
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'solve_equation_sympy',
                'equation': equation_str,
                'variable': variable_str,
                'solutions_count': len(result)
            })
            
            return result
        except Exception as e:
            logger.error(f"Error solving equation with SymPy: {str(e)}")
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'solve_equation_sympy',
                'equation': equation_str,
                'variable': variable_str,
                'error': str(e)
            })
            
            return []
    
    def solve_system_sympy(self, equations: List[str], variables: List[str]) -> List[Dict]:
        """
        Solve system of equations using SymPy
        
        Parameters:
        - equations: List of equation strings
        - variables: List of variable strings
        
        Returns:
        - List of solution dictionaries mapping variables to values
        """
        try:
            parsed_equations = []
            for eq_str in equations:
                if "=" in eq_str:
                    lhs_str, rhs_str = eq_str.split("=")
                    lhs = sp.sympify(lhs_str.strip())
                    rhs = sp.sympify(rhs_str.strip())
                    parsed_equations.append(sp.Eq(lhs, rhs))
                else:
                    parsed_equations.append(sp.Eq(sp.sympify(eq_str), 0))
                    
            parsed_variables = [sp.symbols(var) for var in variables]
            
            solutions = sp.solve(parsed_equations, parsed_variables, dict=True)
            
            result = []
            for sol in solutions:
                solution_dict = {}
                for var, val in sol.items():
                    var_name = str(var)
                    if sp.im(val) == 0:
                        solution_dict[var_name] = float(val)
                    else:
                        solution_dict[var_name] = complex(val)
                result.append(solution_dict)
                
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'solve_system_sympy',
                'equations_count': len(equations),
                'variables_count': len(variables),
                'solutions_count': len(result)
            })
            
            return result
        except Exception as e:
            logger.error(f"Error solving system with SymPy: {str(e)}")
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'solve_system_sympy',
                'equations_count': len(equations),
                'variables_count': len(variables),
                'error': str(e)
            })
            
            return []
    
    def symbolic_differentiate(self, expression_str: str, variable_str: str = 'x', 
                              order: int = 1) -> str:
        """
        Symbolically differentiate an expression
        
        Parameters:
        - expression_str: String representation of the expression
        - variable_str: Variable to differentiate with respect to (default: 'x')
        - order: Order of differentiation (default: 1)
        
        Returns:
        - String representation of the derivative
        """
        try:
            expression = sp.sympify(expression_str)
            variable = sp.symbols(variable_str)
            
            derivative = sp.diff(expression, variable, order)
            
            result = str(sp.simplify(derivative))
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'symbolic_differentiate',
                'expression': expression_str,
                'variable': variable_str,
                'order': order,
                'result': result
            })
            
            return result
        except Exception as e:
            logger.error(f"Error differentiating with SymPy: {str(e)}")
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'symbolic_differentiate',
                'expression': expression_str,
                'variable': variable_str,
                'order': order,
                'error': str(e)
            })
            
            return str(e)
    
    def symbolic_integrate(self, expression_str: str, variable_str: str = 'x',
                          lower_bound: Optional[float] = None, 
                          upper_bound: Optional[float] = None) -> str:
        """
        Symbolically integrate an expression
        
        Parameters:
        - expression_str: String representation of the expression
        - variable_str: Variable to integrate with respect to (default: 'x')
        - lower_bound: Lower bound for definite integral (default: None for indefinite)
        - upper_bound: Upper bound for definite integral (default: None for indefinite)
        
        Returns:
        - String representation of the integral
        """
        try:
            expression = sp.sympify(expression_str)
            variable = sp.symbols(variable_str)
            
            if lower_bound is not None and upper_bound is not None:
                integral = sp.integrate(expression, (variable, lower_bound, upper_bound))
            else:
                integral = sp.integrate(expression, variable)
                
            result = str(sp.simplify(integral))
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'symbolic_integrate',
                'expression': expression_str,
                'variable': variable_str,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'result': result
            })
            
            return result
        except Exception as e:
            logger.error(f"Error integrating with SymPy: {str(e)}")
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'symbolic_integrate',
                'expression': expression_str,
                'variable': variable_str,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'error': str(e)
            })
            
            return str(e)
    
    
    def run_mathematica(self, code: str) -> str:
        """
        Run Mathematica code and return the result
        
        Parameters:
        - code: Mathematica code to execute
        
        Returns:
        - Result as string
        """
        if not self.mathematica_available:
            logger.warning("Mathematica is not available")
            return "Mathematica is not available"
            
        try:
            with tempfile.NamedTemporaryFile(suffix='.m', mode='w', delete=False) as f:
                f.write(code)
                f.write('\nExit[]')
                temp_filename = f.name
                
            result = subprocess.run(
                [self.mathematica_path, "-script", temp_filename],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            os.unlink(temp_filename)
            
            output = result.stdout.strip()
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'run_mathematica',
                'code_length': len(code),
                'output_length': len(output)
            })
            
            return output
        except Exception as e:
            logger.error(f"Error running Mathematica: {str(e)}")
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'run_mathematica',
                'code_length': len(code),
                'error': str(e)
            })
            
            return f"Error: {str(e)}"
    
    
    def run_matlab(self, code: str) -> str:
        """
        Run MATLAB code and return the result
        
        Parameters:
        - code: MATLAB code to execute
        
        Returns:
        - Result as string
        """
        if not self.matlab_available:
            logger.warning("MATLAB is not available")
            return "MATLAB is not available"
            
        try:
            with tempfile.NamedTemporaryFile(suffix='.m', mode='w', delete=False) as f:
                f.write(code)
                temp_filename = f.name
                
            result = subprocess.run(
                [self.matlab_path, "-batch", f"run('{temp_filename}')"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            os.unlink(temp_filename)
            
            output = result.stdout.strip()
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'run_matlab',
                'code_length': len(code),
                'output_length': len(output)
            })
            
            return output
        except Exception as e:
            logger.error(f"Error running MATLAB: {str(e)}")
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'run_matlab',
                'code_length': len(code),
                'error': str(e)
            })
            
            return f"Error: {str(e)}"
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about math computation interface usage
        
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
            'mathematica_available': self.mathematica_available,
            'matlab_available': self.matlab_available
        }
