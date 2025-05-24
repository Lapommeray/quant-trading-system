import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from advanced_modules import QuantumTremorScanner, SpectralSignalFusion, DNABreath, DNAOverlord, VoidTraderChartRenderer
from core.meta_conscious_routing_layer import MetaConsciousRoutingLayer

class WalkForwardQuantumBacktester:
    """
    Enhanced quantum backtesting system with walk-forward validation to prevent look-ahead bias.
    Preserves the quantum-inspired approach while ensuring no future data leakage.
    """
    
    def __init__(self, train_days=180, test_days=30, dimensions=9):
        self.train_days = train_days
        self.test_days = test_days
        self.dimensions = dimensions
        self.branch_data = {}
        self.results_cache = {}
        
    def simulate_branch_entropy(self, timeline_seed):
        """Generate entropy values for each branch without look-ahead bias"""
        np.random.seed(timeline_seed)
        return {
            event: np.random.dirichlet(np.ones(3))[0]
            for event in self.branch_data
        }
        
    def run_quantum_backtest(self, data, start_date, end_date):
        """Run quantum backtest with walk-forward validation"""
        results = []
        current_date = start_date + timedelta(days=self.train_days)
        
        while current_date < end_date:
            train_start = current_date - timedelta(days=self.train_days)
            train_end = current_date
            test_start = current_date
            test_end = min(current_date + timedelta(days=self.test_days), end_date)
            
            train_data = self._extract_data_range(data, train_start, train_end)
            test_data = self._extract_data_range(data, test_start, test_end)
            
            if not train_data or len(train_data) < self.train_days // 2:
                current_date = test_end
                continue
                
            timelines = self._generate_branching_simulations(train_data, depth=self.dimensions)
            
            self._train_quantum_models(timelines, train_data)
            
            test_results = self._test_quantum_models(test_data)
            results.extend(test_results)
            
            current_date = test_end
            
        return self._aggregate_results(results)
        
    def _extract_data_range(self, data, start_date, end_date):
        """Extract data within date range ensuring no future leakage"""
        if isinstance(data, dict):
            range_data = {}
            for key, df in data.items():
                if hasattr(df, 'index'):
                    mask = (df.index >= start_date) & (df.index < end_date)
                    range_data[key] = df.loc[mask].copy() if any(mask) else pd.DataFrame()
            return range_data
        elif hasattr(data, 'index'):
            mask = (data.index >= start_date) & (data.index < end_date)
            return data.loc[mask].copy() if any(mask) else pd.DataFrame()
        return None
        
    def _generate_branching_simulations(self, data, depth=9):
        """Generate branching timeline simulations without look-ahead bias"""
        base_seed = int(datetime.now().timestamp())
        timelines = []
        
        for i in range(2**depth):
            timeline_seed = base_seed + i
            np.random.seed(timeline_seed)
            
            timeline = {
                'seed': timeline_seed,
                'branch_factors': np.random.random(depth),
                'entropy': np.random.random(),
                'data': self._create_timeline_data(data, timeline_seed)
            }
            
            timelines.append(timeline)
            
        return timelines
        
    def _create_timeline_data(self, data, seed):
        """Create timeline data with quantum-inspired variations"""
        np.random.seed(seed)
        
        if isinstance(data, dict):
            timeline_data = {}
            for key, df in data.items():
                if df.empty:
                    timeline_data[key] = df
                    continue
                    
                noise_factor = 0.001 + 0.002 * np.random.random()  # Small noise
                
                if 'Close' in df.columns:
                    noise = df['Close'] * noise_factor * np.random.randn(len(df))
                    df_copy = df.copy()
                    df_copy['Close'] = df['Close'] + noise
                    timeline_data[key] = df_copy
                else:
                    timeline_data[key] = df
            return timeline_data
        else:
            return data
            
    def _train_quantum_models(self, timelines, train_data):
        """Train quantum models on historical data only"""
        self.quantum_tremor_scanner = QuantumTremorScanner()
        self.spectral_fusion = SpectralSignalFusion()
        self.dna_breath = DNABreath()
        self.dna_overlord = DNAOverlord()
        
        for timeline in timelines:
            timeline_data = timeline['data']
            
            if hasattr(self.quantum_tremor_scanner, 'train'):
                self.quantum_tremor_scanner.train(timeline_data)
                
            if hasattr(self.spectral_fusion, 'train'):
                self.spectral_fusion.train(timeline_data)
                
            if hasattr(self.dna_breath, 'train'):
                self.dna_breath.train(timeline_data)
                
            if hasattr(self.dna_overlord, 'train'):
                self.dna_overlord.train(timeline_data)
                
    def _test_quantum_models(self, test_data):
        """Test quantum models on out-of-sample data"""
        results = []
        
        if isinstance(test_data, dict) and any(test_data.values()):
            for key, df in test_data.items():
                if not df.empty and 'Close' in df.columns:
                    test_df = df
                    break
            else:
                return results
        elif hasattr(test_data, 'empty') and not test_data.empty:
            test_df = test_data
        else:
            return results
            
        for i in range(len(test_df) - 1):  # -1 to avoid using the last day which might be incomplete
            day_data = test_df.iloc[i:i+1]
            
            if day_data.empty:
                continue
                
            tremor_signal = self._get_tremor_scanner_prediction(day_data)
            fusion_signal = self._get_spectral_fusion_prediction(day_data)
            breath_signal = self._get_dna_breath_prediction(day_data)
            
            combined_signal = self._combine_signals(tremor_signal, fusion_signal, breath_signal)
            
            next_day = test_df.iloc[i+1] if i+1 < len(test_df) else None
            if next_day is not None:
                price_change = (next_day['Close'] - day_data['Close'].iloc[0]) / day_data['Close'].iloc[0]
                result = {
                    'date': day_data.index[0],
                    'signal': combined_signal,
                    'price_change': price_change,
                    'success': (combined_signal > 0 and price_change > 0) or (combined_signal < 0 and price_change < 0)
                }
                results.append(result)
                
        return results
        
    def _get_tremor_scanner_prediction(self, data):
        """Get prediction from Quantum Tremor Scanner"""
        try:
            return self.quantum_tremor_scanner.predict(data)
        except:
            return 0
            
    def _get_spectral_fusion_prediction(self, data):
        """Get prediction from Spectral Signal Fusion"""
        try:
            return self.spectral_fusion.predict(data)
        except:
            return 0
            
    def _get_dna_breath_prediction(self, data):
        """Get prediction from DNA Breath"""
        try:
            return self.dna_breath.predict(data)
        except:
            return 0
            
    def _combine_signals(self, tremor, fusion, breath):
        """Combine signals from different components"""
        weights = [0.4, 0.4, 0.2]  # Adjust weights as needed
        combined = weights[0] * tremor + weights[1] * fusion + weights[2] * breath
        
        if combined > 0.1:
            return 1  # Buy signal
        elif combined < -0.1:
            return -1  # Sell signal
        else:
            return 0  # No signal
            
    def _aggregate_results(self, results):
        """Aggregate results from all test periods"""
        if not results:
            return {
                'win_rate': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_profit': 0,
                'avg_loss': 0
            }
            
        winning_trades = [r for r in results if r['success']]
        losing_trades = [r for r in results if not r['success']]
        
        total_trades = len(results)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_profit = np.mean([r['price_change'] for r in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([r['price_change'] for r in losing_trades]) if losing_trades else 0
        
        return {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'results': results
        }

def run_walk_forward_backtest(data, start_date, end_date):
    """Run walk-forward backtest with the quantum system"""
    backtest = WalkForwardQuantumBacktester(train_days=180, test_days=30)
    return backtest.run_quantum_backtest(data, start_date, end_date)
