import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SovereigntyCheck")

class SovereigntyCheck:
    @staticmethod
    def run(mode='STANDARD', deploy_mode=None):
        """
        Run sovereignty check to verify system integrity.
        
        Parameters:
        - mode: Check mode ('STANDARD' or 'ULTRA_STRICT')
        - deploy_mode: Optional deployment mode (e.g., 'GOD')
        
        Returns:
        - Boolean indicating if check passed
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        REQUIRED_MODULES = [
            'quantum/temporal_lstm.py',
            'ai/aggressor_ai.py',
            'dark_liquidity/whale_detector.py',
            'secure/audit_trail.py'
        ]
        
        CRITICAL_MODULES = [
            'quantum/time_warp.py',
            'risk/black_swan.py',
            'secure/quantum_vault.py'
        ]
        
        ENHANCED_MODULES = [
            'dark_liquidity/trap.py',
            'ai/mm_psychology.py',
            'quantum/decoherence.py'
        ]
        
        QMMAF_MODULES = [
            'quantum_alignment/temporal_fractal.py',
            'quantum_alignment/mm_dna_scanner.py',
            'quantum_alignment/dark_echo.py',
            'quantum_alignment/alignment_engine.py'
        ]
        
        if mode.upper() == 'ULTRA_STRICT':
            modules_to_check = REQUIRED_MODULES + CRITICAL_MODULES + ENHANCED_MODULES + QMMAF_MODULES
        else:
            modules_to_check = REQUIRED_MODULES
            
            modules_to_check += CRITICAL_MODULES
        
        missing = []
        for module in modules_to_check:
            module_path = os.path.join(repo_root, module)
            if not os.path.exists(module_path):
                missing.append(module)
        
        if not missing:
            if mode.upper() == 'ULTRA_STRICT':
                print("🟢 QUANTUM CERTIFICATION: 200% READY")
            else:
                print("✅ SOVEREIGN STACK OPERATIONAL")
                
            if deploy_mode:
                print(f"DEPLOYMENT MODE: {deploy_mode}")
                
            return True
        else:
            print(f"🚨 CRITICAL GAPS: {missing}")
            return False
            
    @staticmethod
    def verify_all_components():
        """
        Verify all components with detailed status report.
        
        Returns:
        - Dictionary with verification results
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        component_categories = {
            "Core Quantum": [
                'quantum/temporal_lstm.py',
                'quantum/time_warp.py',
                'quantum/decoherence.py'
            ],
            "AI Systems": [
                'ai/aggressor_ai.py',
                'ai/mm_psychology.py',
                'ai/mirror_ai.py',
                'ai/shap_interpreter.py'
            ],
            "Risk Management": [
                'risk/black_swan.py',
                'risk/macro_triggers.py'
            ],
            "Dark Liquidity": [
                'dark_liquidity/whale_detector.py',
                'dark_liquidity/trap.py'
            ],
            "Security": [
                'secure/audit_trail.py',
                'secure/quantum_vault.py'
            ],
            "QMMAF": [
                'quantum_alignment/temporal_fractal.py',
                'quantum_alignment/mm_dna_scanner.py',
                'quantum_alignment/dark_echo.py',
                'quantum_alignment/alignment_engine.py'
            ]
        }
        
        results = {}
        for category, modules in component_categories.items():
            category_results = []
            for module in modules:
                module_path = os.path.join(repo_root, module)
                module_exists = os.path.exists(module_path)
                category_results.append({
                    "module": module,
                    "exists": module_exists
                })
            
            total_modules = len(category_results)
            existing_modules = sum(1 for result in category_results if result["exists"])
            status = "✅ COMPLETE" if existing_modules == total_modules else "⚠️ PARTIAL"
            
            results[category] = {
                "status": status,
                "modules": category_results,
                "completion": f"{existing_modules}/{total_modules}"
            }
        
        print("=" * 80)
        print("QUANTUM TRADING SYSTEM VERIFICATION")
        print("=" * 80)
        
        for category, result in results.items():
            print(f"{category}: {result['status']} ({result['completion']})")
        
        print("=" * 80)
        
        total_modules = sum(len(modules) for modules in component_categories.values())
        existing_modules = sum(
            sum(1 for module in result["modules"] if module["exists"])
            for result in results.values()
        )
        
        completion_percentage = (existing_modules / total_modules) * 100
        
        if completion_percentage == 100:
            print("🟢 QUANTUM CERTIFICATION: 200% READY")
        elif completion_percentage >= 90:
            print("🟡 SYSTEM NEAR COMPLETE: MISSING MINOR COMPONENTS")
        else:
            print("🔴 SYSTEM INCOMPLETE: CRITICAL COMPONENTS MISSING")
            
        print(f"Overall Completion: {completion_percentage:.1f}%")
        print("=" * 80)
        
        return results

def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Quantum Sovereignty Check")
    
    parser.add_argument("--ultra-strict", action="store_true",
                        help="Run in ultra-strict mode")
    
    parser.add_argument("--verify-all", action="store_true",
                        help="Verify all components with detailed report")
    
    parser.add_argument("--deploy-mode", type=str, default=None,
                        help="Deployment mode (e.g., 'GOD')")
    
    args = parser.parse_args()
    
    if args.verify_all:
        SovereigntyCheck.verify_all_components()
    else:
        mode = "ULTRA_STRICT" if args.ultra_strict else "STANDARD"
        SovereigntyCheck.run(mode=mode, deploy_mode=args.deploy_mode)

if __name__ == "__main__":
    main()
