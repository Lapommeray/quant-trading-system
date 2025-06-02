#!/usr/bin/env python3

import os
import sys
import shutil
import click
import yaml
import json
from pathlib import Path
from datetime import datetime

@click.command()
@click.option('--source', '-s', required=True, help='Source directory or backup file')
@click.option('--target', '-t', default='.', help='Target directory')
@click.option('--backup', '-b', is_flag=True, help='Create backup before migration')
@click.option('--dry-run', is_flag=True, help='Dry run migration')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def migrate_system(source, target, backup, dry_run, verbose):
    """Migrate Quant Trading System configuration and data"""
    
    if verbose:
        click.echo("🔄 Quant Trading System Migration")
        click.echo("=" * 40)
        click.echo(f"📂 Source: {source}")
        click.echo(f"📁 Target: {target}")
        click.echo(f"💾 Backup: {backup}")
        click.echo(f"🧪 Dry run: {dry_run}")
    
    try:
        source_path = Path(source)
        target_path = Path(target)
        
        validate_migration_requirements(source_path, target_path, verbose)
        
        if backup and not dry_run:
            create_migration_backup(target_path, verbose)
        
        migrate_configuration(source_path, target_path, dry_run, verbose)
        migrate_data(source_path, target_path, dry_run, verbose)
        migrate_logs(source_path, target_path, dry_run, verbose)
        migrate_models(source_path, target_path, dry_run, verbose)
        
        if not dry_run:
            validate_migration(target_path, verbose)
        
        if verbose:
            click.echo("🎉 Migration completed successfully!")
        
    except Exception as e:
        if verbose:
            click.echo(f"❌ Migration failed: {e}")
        sys.exit(1)

def validate_migration_requirements(source_path, target_path, verbose):
    """Validate migration requirements"""
    if verbose:
        click.echo("🔍 Validating migration requirements...")
    
    if not source_path.exists():
        raise Exception(f"Source path does not exist: {source_path}")
    
    if source_path.is_file() and not source_path.suffix in ['.tar.gz', '.zip']:
        raise Exception(f"Unsupported backup file format: {source_path.suffix}")
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        click.echo("✅ Migration requirements validated")

def create_migration_backup(target_path, verbose):
    """Create backup before migration"""
    if verbose:
        click.echo("💾 Creating migration backup...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"migration_backup_{timestamp}"
    backup_path = target_path.parent / backup_name
    
    if target_path.exists():
        shutil.copytree(target_path, backup_path)
        
        if verbose:
            click.echo(f"✅ Backup created: {backup_path}")

def migrate_configuration(source_path, target_path, dry_run, verbose):
    """Migrate configuration files"""
    if verbose:
        click.echo("⚙️ Migrating configuration...")
    
    config_files = [
        'config.yaml',
        '.env',
        'Sa_son_code/quant_trading_system/config/config.yaml'
    ]
    
    for config_file in config_files:
        source_config = source_path / config_file
        target_config = target_path / config_file
        
        if source_config.exists():
            if verbose:
                click.echo(f"📄 Migrating: {config_file}")
            
            if not dry_run:
                target_config.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_config, target_config)
                
                if config_file.endswith('.yaml'):
                    update_config_paths(target_config, target_path, verbose)

def migrate_data(source_path, target_path, dry_run, verbose):
    """Migrate data files"""
    if verbose:
        click.echo("📊 Migrating data...")
    
    data_dirs = ['data', 'backups', 'mlruns', 'artifacts']
    
    for data_dir in data_dirs:
        source_data = source_path / data_dir
        target_data = target_path / data_dir
        
        if source_data.exists() and source_data.is_dir():
            if verbose:
                click.echo(f"📁 Migrating: {data_dir}")
            
            if not dry_run:
                if target_data.exists():
                    shutil.rmtree(target_data)
                shutil.copytree(source_data, target_data)

def migrate_logs(source_path, target_path, dry_run, verbose):
    """Migrate log files"""
    if verbose:
        click.echo("📝 Migrating logs...")
    
    logs_dir = source_path / 'logs'
    target_logs = target_path / 'logs'
    
    if logs_dir.exists():
        if verbose:
            click.echo("📁 Migrating: logs")
        
        if not dry_run:
            target_logs.mkdir(exist_ok=True)
            
            for log_file in logs_dir.glob('*.log'):
                if log_file.stat().st_size < 100 * 1024 * 1024:
                    shutil.copy2(log_file, target_logs)

def migrate_models(source_path, target_path, dry_run, verbose):
    """Migrate model files"""
    if verbose:
        click.echo("🤖 Migrating models...")
    
    model_dirs = ['models', 'checkpoints', 'weights']
    
    for model_dir in model_dirs:
        source_models = source_path / model_dir
        target_models = target_path / model_dir
        
        if source_models.exists() and source_models.is_dir():
            if verbose:
                click.echo(f"📁 Migrating: {model_dir}")
            
            if not dry_run:
                if target_models.exists():
                    shutil.rmtree(target_models)
                shutil.copytree(source_models, target_models)

def update_config_paths(config_file, target_path, verbose):
    """Update configuration file paths"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'logging' in config and 'file' in config['logging']:
            config['logging']['file'] = str(target_path / 'logs' / 'qts.log')
        
        if 'backup' in config and 'path' in config['backup']:
            config['backup']['path'] = str(target_path / 'backups')
        
        if 'mlflow' in config and 'artifact_root' in config['mlflow']:
            config['mlflow']['artifact_root'] = str(target_path / 'mlruns')
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        if verbose:
            click.echo(f"✅ Updated paths in: {config_file}")
            
    except Exception as e:
        if verbose:
            click.echo(f"⚠️ Failed to update paths in {config_file}: {e}")

def validate_migration(target_path, verbose):
    """Validate migration results"""
    if verbose:
        click.echo("🔍 Validating migration...")
    
    required_files = [
        'config.yaml',
        'requirements.txt',
        'setup.py',
        'Dockerfile'
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = target_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        raise Exception(f"Migration incomplete. Missing files: {', '.join(missing_files)}")
    
    try:
        import quant_trading_system
        consciousness = quant_trading_system.check_quantum_consciousness()
        
        if consciousness['dimensional_coherence'] != 11:
            raise Exception("Dimensional coherence validation failed")
        
        if not quant_trading_system.validate_sacred_geometry():
            raise Exception("Sacred geometry validation failed")
        
        if verbose:
            click.echo("✅ Migration validation passed")
            
    except ImportError:
        if verbose:
            click.echo("⚠️ Package not installed, skipping quantum validation")

if __name__ == '__main__':
    migrate_system()
