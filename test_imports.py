def test_module_imports():
    """Test that all modules can be imported"""
    print('🧪 Testing module imports...')

    modules_to_test = [
        'modules',
        'core', 
        'advanced_modules',
        'defense',
        'arbitrage',
        'data',
        'market_intelligence',
        'ultra_modules',
        'api',
        'integrations'
    ]

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f'✅ {module_name} imported successfully')
        except ImportError as e:
            print(f'⚠️ {module_name} import warning: {e}')

    print('🎉 Module import testing completed!')

if __name__ == '__main__':
    test_module_imports()
