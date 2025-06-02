#!/usr/bin/env python3

import os
import sys
import shutil
import datetime
import argparse
import logging
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackupManager:
    def __init__(self, source_dir: str, backup_dir: str):
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, include_data: bool = True, include_logs: bool = False) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"qts_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        logger.info(f"Creating backup: {backup_path}")
        
        def ignore_patterns(dir_path, names):
            ignored = set()
            
            if not include_data:
                ignored.update(['data', 'backups', 'temp', 'tmp'])
            
            if not include_logs:
                ignored.update(['logs'])
            
            ignored.update([
                '__pycache__',
                '.pytest_cache',
                '.mypy_cache',
                '.tox',
                '.coverage',
                'htmlcov',
                '.git',
                '.venv',
                'venv',
                'node_modules',
                'build',
                'dist',
                '*.egg-info'
            ])
            
            return ignored.intersection(names)
        
        try:
            shutil.copytree(self.source_dir, backup_path, ignore=ignore_patterns)
            logger.info(f"Backup created successfully: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise
    
    def list_backups(self) -> List[str]:
        backups = []
        for item in self.backup_dir.iterdir():
            if item.is_dir() and item.name.startswith('qts_backup_'):
                backups.append(item.name)
        return sorted(backups, reverse=True)
    
    def restore_backup(self, backup_name: str, target_dir: Optional[str] = None) -> None:
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            raise ValueError(f"Backup not found: {backup_name}")
        
        target = Path(target_dir) if target_dir else self.source_dir
        
        logger.info(f"Restoring backup {backup_name} to {target}")
        
        if target.exists():
            backup_existing = target.parent / f"{target.name}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Backing up existing directory to: {backup_existing}")
            shutil.move(str(target), str(backup_existing))
        
        shutil.copytree(backup_path, target)
        logger.info("Restore completed successfully")
    
    def cleanup_old_backups(self, keep_count: int = 10) -> None:
        backups = self.list_backups()
        if len(backups) <= keep_count:
            logger.info(f"Only {len(backups)} backups found, no cleanup needed")
            return
        
        to_remove = backups[keep_count:]
        logger.info(f"Removing {len(to_remove)} old backups")
        
        for backup_name in to_remove:
            backup_path = self.backup_dir / backup_name
            shutil.rmtree(backup_path)
            logger.info(f"Removed backup: {backup_name}")

def main():
    parser = argparse.ArgumentParser(description='Backup and restore QTS system')
    parser.add_argument('action', choices=['create', 'list', 'restore', 'cleanup'], 
                       help='Action to perform')
    parser.add_argument('--source', default='.', help='Source directory to backup')
    parser.add_argument('--backup-dir', default='./backups', help='Backup directory')
    parser.add_argument('--include-data', action='store_true', help='Include data directory')
    parser.add_argument('--include-logs', action='store_true', help='Include logs directory')
    parser.add_argument('--backup-name', help='Backup name for restore operation')
    parser.add_argument('--target-dir', help='Target directory for restore')
    parser.add_argument('--keep-count', type=int, default=10, help='Number of backups to keep')
    
    args = parser.parse_args()
    
    backup_manager = BackupManager(args.source, args.backup_dir)
    
    try:
        if args.action == 'create':
            backup_path = backup_manager.create_backup(
                include_data=args.include_data,
                include_logs=args.include_logs
            )
            print(f"Backup created: {backup_path}")
        
        elif args.action == 'list':
            backups = backup_manager.list_backups()
            if backups:
                print("Available backups:")
                for backup in backups:
                    print(f"  {backup}")
            else:
                print("No backups found")
        
        elif args.action == 'restore':
            if not args.backup_name:
                print("Error: --backup-name is required for restore operation")
                sys.exit(1)
            backup_manager.restore_backup(args.backup_name, args.target_dir)
            print("Restore completed successfully")
        
        elif args.action == 'cleanup':
            backup_manager.cleanup_old_backups(args.keep_count)
            print("Cleanup completed")
    
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
