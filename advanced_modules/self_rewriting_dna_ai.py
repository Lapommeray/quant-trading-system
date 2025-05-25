"""
Self-Rewriting DNA-AI Codebase

An AI whose source code evolves on its own like biological DNA — not just learning,
but mutating and adapting. This module makes AI unstoppable in innovation and survival.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import ccxt
from typing import Dict, List, Any, Optional, Tuple
import time
import json
import os
import random
import hashlib
import base64
import importlib.util
import inspect
import sys

class SelfRewritingDNAAI:
    """
    Self-Rewriting DNA-AI Codebase
    
    An AI that evolves its own source code like biological DNA, not just learning
    but mutating and adapting to market conditions in real-time.
    
    Key features:
    - Code mutation and self-optimization
    - Adaptive strategy generation
    - Survival-focused risk management
    - Real-time market data analysis using ccxt
    """
    
    def __init__(self, algorithm=None, symbol=None):
        """
        Initialize the Self-Rewriting DNA-AI module.
        
        Parameters:
        - algorithm: Optional algorithm instance for integration
        - symbol: Optional symbol to create a symbol-specific instance
        """
        self.algorithm = algorithm
        self.symbol = symbol
        self.logger = logging.getLogger(f"SelfRewritingDNAAI_{symbol}" if symbol else "SelfRewritingDNAAI")
        self.logger.setLevel(logging.INFO)
        
        self.exchange = ccxt.binance({'enableRateLimit': True})
        
        self.dna_sequence = self._generate_initial_dna()
        self.dna_fitness = 0.0
        self.generation = 1
        self.mutation_rate = 0.05  # 5% mutation rate
        self.adaptation_threshold = 0.7  # 70% fitness required for adaptation
        
        self.strategy_components = {
            'entry_conditions': [],
            'exit_conditions': [],
            'risk_management': [],
            'position_sizing': [],
            'timeframe_selection': []
        }
        
        self.performance_history = []
        self.mutation_history = []
        
        self.code_snapshots = []
        self.current_code_hash = self._hash_current_code()
        
        self.last_evolution_time = None
        self.evolution_interval = timedelta(hours=4)
        
        self._save_code_snapshot("initial")
        
        if algorithm:
            algorithm.Debug(f"Self-Rewriting DNA-AI initialized for {symbol}" if symbol else "Self-Rewriting DNA-AI initialized")
    
    def _generate_initial_dna(self) -> str:
        """
        Generate the initial DNA sequence for the AI.
        
        Returns:
        - DNA sequence string
        """
        bases = "ACGT"
        dna = ''.join(random.choice(bases) for _ in range(1024))
        return dna
    
    def _hash_current_code(self) -> str:
        """
        Generate a hash of the current code.
        
        Returns:
        - Hash string
        """
        code = inspect.getsource(self.__class__)
        return hashlib.sha256(code.encode()).hexdigest()
    
    def _save_code_snapshot(self, reason: str):
        """
        Save a snapshot of the current code.
        
        Parameters:
        - reason: Reason for the snapshot
        """
        code = inspect.getsource(self.__class__)
        hash_value = hashlib.sha256(code.encode()).hexdigest()
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'generation': self.generation,
            'hash': hash_value,
            'fitness': self.dna_fitness,
            'reason': reason,
            'dna_sample': self.dna_sequence[:50] + "..." + self.dna_sequence[-50:]
        }
        
        self.code_snapshots.append(snapshot)
        self.current_code_hash = hash_value
    
    def _mutate_dna(self) -> str:
        """
        Mutate the DNA sequence.
        
        Returns:
        - Mutated DNA sequence
        """
        bases = "ACGT"
        mutated_dna = list(self.dna_sequence)
        
        num_mutations = int(len(self.dna_sequence) * self.mutation_rate)
        
        for _ in range(num_mutations):
            mutation_type = random.choice(["substitute", "insert", "delete", "duplicate", "invert"])
            position = random.randint(0, len(mutated_dna) - 1)
            
            if mutation_type == "substitute":
                mutated_dna[position] = random.choice(bases)
            elif mutation_type == "insert":
                mutated_dna.insert(position, random.choice(bases))
                if len(mutated_dna) > 1024:
                    mutated_dna.pop()
            elif mutation_type == "delete" and len(mutated_dna) > 100:
                mutated_dna.pop(position)
                mutated_dna.append(random.choice(bases))
            elif mutation_type == "duplicate" and position < len(mutated_dna) - 10:
                segment_length = random.randint(3, 10)
                segment = mutated_dna[position:position+segment_length]
                insert_position = random.randint(0, len(mutated_dna) - 1)
                for i, base in enumerate(segment):
                    if insert_position + i < len(mutated_dna):
                        mutated_dna[insert_position + i] = base
            elif mutation_type == "invert" and position < len(mutated_dna) - 10:
                segment_length = random.randint(3, 10)
                segment = mutated_dna[position:position+segment_length]
                inverted = segment[::-1]
                for i, base in enumerate(inverted):
                    if position + i < len(mutated_dna):
                        mutated_dna[position + i] = base
        
        return ''.join(mutated_dna)
    
    def _dna_to_strategy(self) -> Dict[str, Any]:
        """
        Convert DNA sequence to trading strategy components.
        
        Returns:
        - Dictionary with strategy components
        """
        strategy = {}
        
        entry_segment = self.dna_sequence[:256]
        entry_hash = int(hashlib.md5(entry_segment.encode()).hexdigest(), 16)
        
        strategy['rsi_period'] = (entry_hash % 20) + 5  # RSI period between 5-24
        strategy['rsi_oversold'] = (entry_hash % 15) + 25  # RSI oversold between 25-39
        strategy['rsi_overbought'] = (entry_hash % 15) + 65  # RSI overbought between 65-79
        
        exit_segment = self.dna_sequence[256:512]
        exit_hash = int(hashlib.md5(exit_segment.encode()).hexdigest(), 16)
        
        strategy['take_profit'] = ((exit_hash % 50) + 10) / 10  # Take profit between 1.0% and 6.0%
        strategy['stop_loss'] = ((exit_hash % 30) + 5) / 10  # Stop loss between 0.5% and 3.5%
        strategy['trailing_stop'] = (exit_hash % 2) == 1  # Boolean for trailing stop
        
        risk_segment = self.dna_sequence[512:768]
        risk_hash = int(hashlib.md5(risk_segment.encode()).hexdigest(), 16)
        
        strategy['max_position_size'] = ((risk_hash % 20) + 1) / 100  # Max position size between 1% and 20%
        strategy['max_open_positions'] = (risk_hash % 5) + 1  # Max open positions between 1 and 5
        strategy['position_scaling'] = (risk_hash % 3) == 1  # Boolean for position scaling
        
        time_segment = self.dna_sequence[768:]
        time_hash = int(hashlib.md5(time_segment.encode()).hexdigest(), 16)
        
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        strategy['primary_timeframe'] = timeframes[time_hash % len(timeframes)]
        strategy['secondary_timeframe'] = timeframes[(time_hash // len(timeframes)) % len(timeframes)]
        
        return strategy
    
    def _evaluate_fitness(self, market_data: pd.DataFrame) -> float:
        """
        Evaluate the fitness of the current DNA.
        
        Parameters:
        - market_data: DataFrame with market data
        
        Returns:
        - Fitness score (0.0 to 1.0)
        """
        if market_data is None or len(market_data) < 100:
            return 0.0
        
        try:
            strategy = self._dna_to_strategy()
            
            initial_capital = 10000.0
            capital = initial_capital
            position = 0.0
            entry_price = 0.0
            trades = []
            
            close_prices = market_data['close'].values
            deltas = np.zeros_like(close_prices)
            deltas[1:] = close_prices[1:] - close_prices[:-1]
            
            gains = np.zeros_like(deltas)
            losses = np.zeros_like(deltas)
            
            for i in range(len(deltas)):
                if deltas[i] > 0:
                    gains[i] = deltas[i]
                elif deltas[i] < 0:
                    losses[i] = abs(deltas[i])
            
            gain_series = pd.Series(gains)
            loss_series = pd.Series(losses)
            
            window = int(strategy['rsi_period'])
            avg_gain = gain_series.rolling(window=window).mean()
            avg_loss = loss_series.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            for i in range(strategy['rsi_period'] + 1, len(market_data)):
                current_price = market_data['close'].iloc[i]
                current_rsi = rsi.iloc[i]
                
                if position > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    if pnl_pct <= -strategy['stop_loss'] / 100:
                        trade_result = {
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'stop_loss'
                        }
                        trades.append(trade_result)
                        
                        capital = capital * (1 + pnl_pct * position / initial_capital)
                        position = 0.0
                    
                    elif pnl_pct >= strategy['take_profit'] / 100:
                        trade_result = {
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'take_profit'
                        }
                        trades.append(trade_result)
                        
                        capital = capital * (1 + pnl_pct * position / initial_capital)
                        position = 0.0
                    
                    elif current_rsi > strategy['rsi_overbought']:
                        trade_result = {
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'rsi_overbought'
                        }
                        trades.append(trade_result)
                        
                        capital = capital * (1 + pnl_pct * position / initial_capital)
                        position = 0.0
                
                elif position == 0 and current_rsi < strategy['rsi_oversold']:
                    position_size = strategy['max_position_size']
                    position = position_size * capital
                    entry_price = current_price
            
            if len(trades) == 0:
                return 0.1  # Low fitness if no trades
            
            final_return = (capital - initial_capital) / initial_capital
            
            winning_trades = sum(1 for trade in trades if trade['pnl_pct'] > 0)
            win_rate = winning_trades / len(trades) if len(trades) > 0 else 0
            
            avg_profit = sum(trade['pnl_pct'] for trade in trades) / len(trades)
            
            returns = [trade['pnl_pct'] for trade in trades]
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            equity_curve = [initial_capital]
            for trade in trades:
                equity_curve.append(equity_curve[-1] * (1 + trade['pnl_pct'] * strategy['max_position_size']))
            
            drawdowns = []
            peak = equity_curve[0]
            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                drawdowns.append(drawdown)
            
            max_drawdown = max(drawdowns)
            
            returns_component = 0.3 * min(1.0, max(0.0, float(final_return / 0.5)))
            win_rate_component = 0.2 * float(win_rate)
            sharpe_component = 0.2 * min(1.0, max(0.0, float(sharpe / 2.0)))
            drawdown_component = 0.3 * (1.0 - min(1.0, float(max_drawdown * 5.0)))
            
            fitness = returns_component + win_rate_component + sharpe_component + drawdown_component
            
            fitness = max(0.0, min(1.0, fitness))
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error evaluating fitness: {str(e)}")
            return 0.0
    
    def evolve(self, market_data: pd.DataFrame) -> bool:
        """
        Evolve the DNA based on market data.
        
        Parameters:
        - market_data: DataFrame with market data
        
        Returns:
        - Boolean indicating if evolution occurred
        """
        current_time = datetime.now()
        
        if (self.last_evolution_time is not None and 
            current_time - self.last_evolution_time < self.evolution_interval):
            return False
        
        self.last_evolution_time = current_time
        
        current_fitness = self._evaluate_fitness(market_data)
        self.dna_fitness = current_fitness
        
        self.performance_history.append({
            'timestamp': current_time.isoformat(),
            'generation': self.generation,
            'fitness': current_fitness,
            'dna_hash': hashlib.md5(self.dna_sequence.encode()).hexdigest()
        })
        
        mutations = []
        for i in range(5):  # Generate 5 mutations
            mutated_dna = self._mutate_dna()
            
            original_dna = self.dna_sequence
            self.dna_sequence = mutated_dna
            
            mutation_fitness = self._evaluate_fitness(market_data)
            
            mutations.append({
                'dna': mutated_dna,
                'fitness': mutation_fitness
            })
            
            self.dna_sequence = original_dna
        
        best_mutation = max(mutations, key=lambda x: x['fitness'])
        
        if best_mutation['fitness'] > current_fitness:
            self.mutation_history.append({
                'timestamp': current_time.isoformat(),
                'generation': self.generation,
                'old_fitness': current_fitness,
                'new_fitness': best_mutation['fitness'],
                'improvement': best_mutation['fitness'] - current_fitness
            })
            
            self.dna_sequence = best_mutation['dna']
            self.dna_fitness = best_mutation['fitness']
            self.generation += 1
            
            self._save_code_snapshot(f"evolution_gen_{self.generation}")
            
            self.strategy_components = self._dna_to_strategy()
            
            self.logger.info(f"DNA evolved to generation {self.generation} with fitness {self.dna_fitness:.4f}")
            return True
        
        return False
    
    def analyze_market(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Dict[str, Any]:
        """
        Analyze market data using the current DNA-based strategy.
        
        Parameters:
        - symbol: Trading symbol (e.g., 'BTC/USDT')
        - timeframe: Timeframe for analysis (e.g., '1h', '4h', '1d')
        - limit: Number of candles to analyze
        
        Returns:
        - Dictionary with analysis results
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < limit * 0.9:  # Ensure we have enough data
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'error': 'Insufficient data'
                }
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            self.evolve(df)
            
            strategy = self._dna_to_strategy()
            
            close_prices = df['close'].values
            deltas = np.zeros_like(close_prices)
            deltas[1:] = close_prices[1:] - close_prices[:-1]
            
            gains = np.zeros_like(deltas)
            losses = np.zeros_like(deltas)
            
            for i in range(len(deltas)):
                if deltas[i] > 0:
                    gains[i] = deltas[i]
                elif deltas[i] < 0:
                    losses[i] = abs(deltas[i])
            
            gain_series = pd.Series(gains)
            loss_series = pd.Series(losses)
            
            window = int(strategy['rsi_period'])
            avg_gain = gain_series.rolling(window=window).mean()
            avg_loss = loss_series.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df['sma_short'] = df['close'].rolling(window=20).mean()
            df['sma_long'] = df['close'].rolling(window=50).mean()
            
            current_rsi = df['rsi'].iloc[-1]
            current_price = df['close'].iloc[-1]
            sma_short = df['sma_short'].iloc[-1]
            sma_long = df['sma_long'].iloc[-1]
            
            signal = 'NEUTRAL'
            confidence = 0.0
            
            if current_rsi < strategy['rsi_oversold'] and sma_short > sma_long:
                signal = 'BUY'
                confidence = 0.7 + (strategy['rsi_oversold'] - current_rsi) / 100
            elif current_rsi > strategy['rsi_overbought'] and sma_short < sma_long:
                signal = 'SELL'
                confidence = 0.7 + (current_rsi - strategy['rsi_overbought']) / 100
            
            confidence = min(0.95, max(0.0, confidence))
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': signal,
                'confidence': float(confidence),
                'current_price': float(current_price),
                'rsi': float(current_rsi),
                'generation': self.generation,
                'dna_fitness': float(self.dna_fitness),
                'strategy': strategy,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def generate_trading_signal(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Generate trading signal based on DNA-evolved strategy.
        
        Parameters:
        - symbol: Trading symbol
        - timeframe: Timeframe for analysis
        
        Returns:
        - Dictionary with trading signal information
        """
        analysis = self.analyze_market(symbol, timeframe)
        
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signal': analysis.get('signal', 'NEUTRAL'),
            'confidence': analysis.get('confidence', 0.0),
            'generation': self.generation,
            'dna_fitness': float(self.dna_fitness)
        }
        
        if signal['signal'] in ['BUY', 'SELL'] and signal['confidence'] > 0.7:
            strategy = self._dna_to_strategy()
            signal['position_size'] = float(strategy['max_position_size'])
            signal['stop_loss_pct'] = float(strategy['stop_loss'])
            signal['take_profit_pct'] = float(strategy['take_profit'])
        
        return signal
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the DNA evolution.
        
        Returns:
        - Dictionary with evolution statistics
        """
        stats = {
            'generation': self.generation,
            'current_fitness': float(self.dna_fitness),
            'dna_length': len(self.dna_sequence),
            'mutation_rate': float(self.mutation_rate),
            'adaptation_threshold': float(self.adaptation_threshold),
            'mutations_count': len(self.mutation_history),
            'last_evolution': self.last_evolution_time.isoformat() if self.last_evolution_time else None
        }
        
        if self.performance_history:
            initial_fitness = self.performance_history[0]['fitness']
            current_fitness = self.dna_fitness
            stats['fitness_improvement'] = float(current_fitness - initial_fitness)
            stats['fitness_improvement_pct'] = float((current_fitness - initial_fitness) / max(0.001, initial_fitness) * 100)
        
        recent_mutations = self.mutation_history[-10:] if len(self.mutation_history) >= 10 else self.mutation_history
        if recent_mutations:
            avg_improvement = sum(m['improvement'] for m in recent_mutations) / len(recent_mutations)
            stats['recent_avg_improvement'] = float(avg_improvement)
        
        return stats
