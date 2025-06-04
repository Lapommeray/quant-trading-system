#!/usr/bin/env python3

import click
import sys
import os
from pathlib import Path

@click.group()
@click.version_option(version="2.5.0")
@click.pass_context
def cli(ctx):
    """Quant Trading System CLI - Advanced Quantum Finance Integration Platform"""
    ctx.ensure_object(dict)

@cli.command()
@click.option('--env', default='development', help='Environment to validate')
def validate(env):
    """Validate system setup and configuration"""
    from scripts.validate_setup import main as validate_main
    sys.argv = ['validate_setup.py']
    validate_main()

@cli.command()
@click.option('--include-data', is_flag=True, help='Include data directory in backup')
@click.option('--include-logs', is_flag=True, help='Include logs directory in backup')
def backup(include_data, include_logs):
    """Create system backup"""
    from scripts.backup import main as backup_main
    args = ['backup.py', 'create']
    if include_data:
        args.append('--include-data')
    if include_logs:
        args.append('--include-logs')
    sys.argv = args
    backup_main()

@cli.command()
@click.argument('environment', type=click.Choice(['development', 'staging', 'production']))
@click.option('--skip-tests', is_flag=True, help='Skip running tests before deployment')
def deploy(environment, skip_tests):
    """Deploy to specified environment"""
    from scripts.deploy import main as deploy_main
    args = ['deploy.py', environment]
    if skip_tests:
        args.append('--skip-tests')
    sys.argv = args
    deploy_main()

@cli.command()
@click.argument('action', type=click.Choice(['up', 'down', 'status']))
@click.option('--steps', type=int, default=1, help='Number of migration steps for down action')
def migrate(action, steps):
    """Database migration management"""
    from scripts.migrate import main as migrate_main
    args = ['migrate.py', action]
    if action == 'down':
        args.extend(['--steps', str(steps)])
    sys.argv = args
    migrate_main()

@cli.command()
@click.option('--host', default='0.0.0.0', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def start_api(host, port, reload):
    """Start the API server"""
    import uvicorn
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)

@cli.command()
@click.option('--host', default='0.0.0.0', help='Dashboard host')
@click.option('--port', default=8501, help='Dashboard port')
def start_dashboard(host, port):
    """Start the Streamlit dashboard"""
    import subprocess
    cmd = [
        'streamlit', 'run', 'dashboard.py',
        '--server.address', host,
        '--server.port', str(port)
    ]
    subprocess.run(cmd)

@cli.command()
def status():
    """Show system status"""
    try:
        import quant_trading_system
        status = quant_trading_system.get_system_status()
        
        click.echo("🚀 Quant Trading System Status")
        click.echo("=" * 40)
        click.echo(f"Version: {status['version']}")
        click.echo(f"Quantum Enabled: {'✅' if status['quantum_enabled'] else '❌'}")
        click.echo()
        
        click.echo("📦 Modules:")
        for module, count in status['modules'].items():
            click.echo(f"  {module}: {count} components")
        
        click.echo()
        click.echo("🔌 Integrations:")
        click.echo(f"  Available: {status['integrations']['available']} integrations")
        
    except ImportError:
        click.echo("❌ System not properly installed. Run 'qts validate' to check setup.")

@cli.command()
def info():
    """Show system information"""
    try:
        import quant_trading_system
        info = quant_trading_system.get_version_info()
        modules = quant_trading_system.list_available_modules()
        
        click.echo("📊 System Information")
        click.echo("=" * 40)
        click.echo(f"Version: {info['version']}")
        click.echo(f"Python: {info['python_version']}")
        click.echo(f"Author: {info['author']}")
        click.echo(f"License: {info['license']}")
        click.echo()
        
        click.echo(f"📋 Available Modules ({len(modules)}):")
        for module in sorted(modules):
            click.echo(f"  • {module}")
            
    except ImportError:
        click.echo("❌ System not properly installed.")

@cli.command()
@click.option('--format', 'output_format', type=click.Choice(['json', 'yaml', 'table']), default='table')
def config(output_format):
    """Show current configuration"""
    import yaml
    
    config_file = Path('config.yaml')
    if not config_file.exists():
        click.echo("❌ Configuration file not found")
        return
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    if output_format == 'json':
        import json
        click.echo(json.dumps(config_data, indent=2))
    elif output_format == 'yaml':
        click.echo(yaml.dump(config_data, default_flow_style=False))
    else:
        click.echo("⚙️  Current Configuration")
        click.echo("=" * 40)
        for section, values in config_data.items():
            click.echo(f"\n[{section}]")
            if isinstance(values, dict):
                for key, value in values.items():
                    click.echo(f"  {key}: {value}")
            else:
                click.echo(f"  {values}")

def main():
    """Main CLI entry point"""
    cli()

if __name__ == '__main__':
    main()
