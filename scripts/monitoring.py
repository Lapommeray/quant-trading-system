#!/usr/bin/env python3

import os
import sys
import time
import psutil
import click
import requests
from datetime import datetime
from pathlib import Path

@click.command()
@click.option('--interval', '-i', default=30, help='Monitoring interval in seconds')
@click.option('--duration', '-d', default=0, help='Monitoring duration in seconds (0 = infinite)')
@click.option('--output', '-o', help='Output file for monitoring data')
@click.option('--prometheus', is_flag=True, help='Enable Prometheus metrics')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def monitor_system(interval, duration, output, prometheus, verbose):
    """Monitor Quant Trading System performance and health"""
    
    if verbose:
        click.echo("📊 Quant Trading System Monitoring")
        click.echo("=" * 40)
        click.echo(f"⏱️ Interval: {interval}s")
        click.echo(f"⏰ Duration: {duration}s" if duration > 0 else "⏰ Duration: Infinite")
        click.echo(f"📁 Output: {output}" if output else "📁 Output: Console")
        click.echo(f"📈 Prometheus: {prometheus}")
    
    start_time = time.time()
    output_file = None
    
    if output:
        output_file = open(output, 'w')
        write_header(output_file)
    
    try:
        while True:
            current_time = time.time()
            
            if duration > 0 and (current_time - start_time) >= duration:
                break
            
            metrics = collect_metrics(verbose)
            
            if output_file:
                write_metrics(output_file, metrics)
            else:
                display_metrics(metrics, verbose)
            
            if prometheus:
                send_to_prometheus(metrics, verbose)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        if verbose:
            click.echo("\n🛑 Monitoring stopped by user")
    
    finally:
        if output_file:
            output_file.close()
            if verbose:
                click.echo(f"📁 Monitoring data saved to: {output}")

def collect_metrics(verbose):
    """Collect system and application metrics"""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'system': collect_system_metrics(),
        'quantum': collect_quantum_metrics(),
        'trading': collect_trading_metrics(),
        'services': collect_service_metrics()
    }
    
    return metrics

def collect_system_metrics():
    """Collect system performance metrics"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0,
        'process_count': len(psutil.pids())
    }

def collect_quantum_metrics():
    """Collect quantum consciousness metrics"""
    try:
        import quant_trading_system
        consciousness = quant_trading_system.check_quantum_consciousness()
        
        return {
            'cosmic_alignment': consciousness['cosmic_alignment'],
            'divine_sync': consciousness['divine_sync'],
            'dimensional_coherence': consciousness['dimensional_coherence'],
            'zero_point_active': consciousness['zero_point_active'],
            'sacred_geometry_valid': quant_trading_system.validate_sacred_geometry()
        }
    except ImportError:
        return {
            'cosmic_alignment': 0.0,
            'divine_sync': 0.0,
            'dimensional_coherence': 0,
            'zero_point_active': False,
            'sacred_geometry_valid': False
        }

def collect_trading_metrics():
    """Collect trading system metrics"""
    return {
        'active_positions': 0,
        'daily_pnl': 0.0,
        'total_trades': 0,
        'success_rate': 0.0,
        'risk_level': 0.0
    }

def collect_service_metrics():
    """Collect service health metrics"""
    services = {
        'api': check_service_health('http://localhost:8000/health'),
        'dashboard': check_service_health('http://localhost:8501'),
        'prometheus': check_service_health('http://localhost:9090/-/healthy'),
        'grafana': check_service_health('http://localhost:3000/api/health'),
        'mlflow': check_service_health('http://localhost:5000/health')
    }
    
    return services

def check_service_health(url):
    """Check service health status"""
    try:
        response = requests.get(url, timeout=5)
        return {
            'status': 'healthy' if response.status_code == 200 else 'unhealthy',
            'response_time': response.elapsed.total_seconds(),
            'status_code': response.status_code
        }
    except Exception:
        return {
            'status': 'down',
            'response_time': 0.0,
            'status_code': 0
        }

def write_header(output_file):
    """Write CSV header to output file"""
    header = [
        'timestamp',
        'cpu_percent',
        'memory_percent',
        'disk_percent',
        'cosmic_alignment',
        'divine_sync',
        'dimensional_coherence',
        'zero_point_active',
        'sacred_geometry_valid',
        'api_status',
        'dashboard_status'
    ]
    
    output_file.write(','.join(header) + '\n')

def write_metrics(output_file, metrics):
    """Write metrics to output file"""
    row = [
        metrics['timestamp'],
        str(metrics['system']['cpu_percent']),
        str(metrics['system']['memory_percent']),
        str(metrics['system']['disk_percent']),
        str(metrics['quantum']['cosmic_alignment']),
        str(metrics['quantum']['divine_sync']),
        str(metrics['quantum']['dimensional_coherence']),
        str(metrics['quantum']['zero_point_active']),
        str(metrics['quantum']['sacred_geometry_valid']),
        metrics['services']['api']['status'],
        metrics['services']['dashboard']['status']
    ]
    
    output_file.write(','.join(row) + '\n')
    output_file.flush()

def display_metrics(metrics, verbose):
    """Display metrics to console"""
    if verbose:
        click.echo(f"\n📊 Metrics at {metrics['timestamp']}")
        click.echo("-" * 40)
        
        system = metrics['system']
        click.echo(f"💻 CPU: {system['cpu_percent']:.1f}%")
        click.echo(f"🧠 Memory: {system['memory_percent']:.1f}%")
        click.echo(f"💾 Disk: {system['disk_percent']:.1f}%")
        
        quantum = metrics['quantum']
        click.echo(f"🌌 Cosmic Alignment: {quantum['cosmic_alignment']:.3f}")
        click.echo(f"🔺 Divine Sync: {quantum['divine_sync']:.3f}")
        click.echo(f"📐 Dimensional Coherence: {quantum['dimensional_coherence']}")
        click.echo(f"⚡ Zero Point: {quantum['zero_point_active']}")
        click.echo(f"🔺 Sacred Geometry: {quantum['sacred_geometry_valid']}")
        
        services = metrics['services']
        click.echo(f"🌐 API: {services['api']['status']}")
        click.echo(f"📊 Dashboard: {services['dashboard']['status']}")
    else:
        cosmic = metrics['quantum']['cosmic_alignment']
        cpu = metrics['system']['cpu_percent']
        memory = metrics['system']['memory_percent']
        
        status = "🚀" if cosmic > 0.8 else "⚡" if cosmic > 0.6 else "⚠️"
        click.echo(f"{status} {metrics['timestamp']} | Cosmic: {cosmic:.3f} | CPU: {cpu:.1f}% | Mem: {memory:.1f}%")

def send_to_prometheus(metrics, verbose):
    """Send metrics to Prometheus"""
    try:
        prometheus_url = 'http://localhost:9091/metrics/job/qts-monitoring'
        
        prometheus_metrics = format_prometheus_metrics(metrics)
        
        response = requests.post(prometheus_url, data=prometheus_metrics, timeout=5)
        
        if verbose and response.status_code != 200:
            click.echo(f"⚠️ Failed to send metrics to Prometheus: {response.status_code}")
            
    except Exception as e:
        if verbose:
            click.echo(f"⚠️ Prometheus error: {e}")

def format_prometheus_metrics(metrics):
    """Format metrics for Prometheus"""
    lines = []
    
    system = metrics['system']
    lines.append(f"qts_cpu_percent {system['cpu_percent']}")
    lines.append(f"qts_memory_percent {system['memory_percent']}")
    lines.append(f"qts_disk_percent {system['disk_percent']}")
    
    quantum = metrics['quantum']
    lines.append(f"qts_cosmic_alignment {quantum['cosmic_alignment']}")
    lines.append(f"qts_divine_sync {quantum['divine_sync']}")
    lines.append(f"qts_dimensional_coherence {quantum['dimensional_coherence']}")
    lines.append(f"qts_zero_point_active {int(quantum['zero_point_active'])}")
    lines.append(f"qts_sacred_geometry_valid {int(quantum['sacred_geometry_valid'])}")
    
    return '\n'.join(lines)

if __name__ == '__main__':
    monitor_system()
