#!/usr/bin/env python3
"""
Test script for module factory discovery
"""
from module_factory import AdvancedModuleFactory

def test_module_discovery():
    print("Testing module factory discovery...")
    factory = AdvancedModuleFactory()
    
    print(f'Total modules discovered: {factory.get_module_count()}')
    
    for category in factory.module_categories:
        modules = factory.get_category_modules(category)
        print(f'{category}: {len(modules)} modules - {modules}')
        
    if factory.get_module_count() > 0:
        print("✓ Module discovery working")
        return True
    else:
        print("✗ Module discovery failed")
        return False

if __name__ == "__main__":
    test_module_discovery()
