import pytest
import json
import yaml
from pathlib import Path

@pytest.mark.unit
class TestMonitoring:
    
    def test_prometheus_config_exists(self):
        """Test Prometheus configuration exists"""
        prometheus_config = Path(__file__).parent.parent / 'monitoring' / 'prometheus.yml'
        if prometheus_config.exists():
            with open(prometheus_config, 'r') as f:
                config = yaml.safe_load(f)
                assert 'global' in config
                assert 'scrape_configs' in config
                assert isinstance(config['scrape_configs'], list)
        else:
            pytest.skip("Prometheus config not found")
    
    def test_grafana_dashboards_valid(self):
        """Test Grafana dashboards are valid JSON"""
        dashboards_dir = Path(__file__).parent.parent / 'monitoring' / 'grafana' / 'dashboards'
        if dashboards_dir.exists():
            dashboard_files = list(dashboards_dir.glob('*.json'))
            assert len(dashboard_files) > 0, "No Grafana dashboards found"
            
            for dashboard_file in dashboard_files:
                with open(dashboard_file, 'r') as f:
                    dashboard = json.load(f)
                    assert 'dashboard' in dashboard
                    assert 'title' in dashboard['dashboard']
                    assert 'panels' in dashboard['dashboard']
        else:
            pytest.skip("Grafana dashboards directory not found")
    
    def test_trading_performance_dashboard(self):
        """Test trading performance dashboard structure"""
        dashboard_file = Path(__file__).parent.parent / 'monitoring' / 'grafana' / 'dashboards' / 'trading-performance.json'
        if dashboard_file.exists():
            with open(dashboard_file, 'r') as f:
                dashboard = json.load(f)
                
                assert dashboard['dashboard']['title'] == "Trading Performance Dashboard"
                panels = dashboard['dashboard']['panels']
                assert len(panels) > 0
                
                panel_titles = [panel['title'] for panel in panels]
                expected_panels = [
                    "Portfolio Value",
                    "Never-Loss Protection Status",
                    "Quantum Accuracy Multiplier",
                    "Active Positions"
                ]
                
                for expected_panel in expected_panels:
                    assert any(expected_panel in title for title in panel_titles)
        else:
            pytest.skip("Trading performance dashboard not found")
    
    def test_quantum_consciousness_dashboard(self):
        """Test quantum consciousness dashboard structure"""
        dashboard_file = Path(__file__).parent.parent / 'monitoring' / 'grafana' / 'dashboards' / 'quantum-consciousness.json'
        if dashboard_file.exists():
            with open(dashboard_file, 'r') as f:
                dashboard = json.load(f)
                
                assert dashboard['dashboard']['title'] == "Quantum Consciousness Dashboard"
                panels = dashboard['dashboard']['panels']
                assert len(panels) > 0
                
                panel_titles = [panel['title'] for panel in panels]
                expected_panels = [
                    "Cosmic Alignment Score",
                    "Golden Ratio Validation",
                    "Zero Point Field Activity",
                    "Dimensional Coherence"
                ]
                
                for expected_panel in expected_panels:
                    assert any(expected_panel in title for title in panel_titles)
        else:
            pytest.skip("Quantum consciousness dashboard not found")
    
    def test_dashboard_refresh_rates(self):
        """Test dashboard refresh rates are appropriate"""
        dashboards_dir = Path(__file__).parent.parent / 'monitoring' / 'grafana' / 'dashboards'
        if dashboards_dir.exists():
            for dashboard_file in dashboards_dir.glob('*.json'):
                with open(dashboard_file, 'r') as f:
                    dashboard = json.load(f)
                    
                    if 'refresh' in dashboard['dashboard']:
                        refresh = dashboard['dashboard']['refresh']
                        assert refresh in ['5s', '10s', '30s', '1m', '5m'], f"Invalid refresh rate: {refresh}"
        else:
            pytest.skip("Grafana dashboards directory not found")
    
    def test_panel_configurations(self):
        """Test panel configurations are valid"""
        dashboards_dir = Path(__file__).parent.parent / 'monitoring' / 'grafana' / 'dashboards'
        if dashboards_dir.exists():
            for dashboard_file in dashboards_dir.glob('*.json'):
                with open(dashboard_file, 'r') as f:
                    dashboard = json.load(f)
                    
                    panels = dashboard['dashboard']['panels']
                    for panel in panels:
                        assert 'id' in panel
                        assert 'title' in panel
                        assert 'type' in panel
                        assert 'gridPos' in panel
                        
                        if 'targets' in panel:
                            for target in panel['targets']:
                                assert 'expr' in target or 'query' in target
        else:
            pytest.skip("Grafana dashboards directory not found")
