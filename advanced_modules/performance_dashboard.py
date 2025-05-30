import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

class PerformanceDashboard:
    """
    Streamlit dashboard for monitoring trading performance
    """
    def __init__(self):
        self.performance_data = {}
        self.trades = []
        self.positions = {}
        self.metrics = {}
        
    def add_performance_data(self, date, equity, returns, drawdown):
        """
        Add daily performance data
        """
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d")
            
        self.performance_data[date] = {
            'equity': equity,
            'returns': returns,
            'drawdown': drawdown
        }
        
    def add_trade(self, symbol, entry_time, entry_price, exit_time=None, exit_price=None, 
                 quantity=1, direction="long", pnl=None, status="open"):
        """
        Add trade to history
        """
        trade_id = len(self.trades) + 1
        
        trade = {
            'id': trade_id,
            'symbol': symbol,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'quantity': quantity,
            'direction': direction,
            'pnl': pnl,
            'status': status
        }
        
        self.trades.append(trade)
        return trade_id
        
    def update_trade(self, trade_id, exit_time, exit_price, pnl, status="closed"):
        """
        Update trade with exit information
        """
        for trade in self.trades:
            if trade['id'] == trade_id:
                trade['exit_time'] = exit_time
                trade['exit_price'] = exit_price
                trade['pnl'] = pnl
                trade['status'] = status
                return True
                
        return False
        
    def update_position(self, symbol, quantity, avg_price, current_price=None):
        """
        Update current position
        """
        self.positions[symbol] = {
            'quantity': quantity,
            'avg_price': avg_price,
            'current_price': current_price,
            'market_value': quantity * (current_price or avg_price)
        }
        
    def calculate_metrics(self):
        """
        Calculate performance metrics
        """
        if not self.performance_data:
            return {}
            
        df = pd.DataFrame.from_dict(self.performance_data, orient='index')
        df.sort_index(inplace=True)
        
        returns = df['returns'].values.astype(float)
        equity = df['equity'].values.astype(float)
        
        # Calculate metrics with proper type handling
        self.metrics = {
            'total_return': float((equity[-1] / equity[0]) - 1),
            'annualized_return': float(((equity[-1] / equity[0]) ** (252 / len(equity))) - 1),
            'sharpe_ratio': float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0,
            'max_drawdown': float(df['drawdown'].min()),
            'volatility': float(np.std(returns) * np.sqrt(252)),
            'win_rate': float(sum(1 for t in self.trades if t['pnl'] and t['pnl'] > 0) / len(self.trades)) if self.trades else 0.0,
            'profit_factor': float(sum(t['pnl'] for t in self.trades if t['pnl'] and t['pnl'] > 0) / abs(sum(t['pnl'] for t in self.trades if t['pnl'] and t['pnl'] < 0))) if sum(t['pnl'] for t in self.trades if t['pnl'] and t['pnl'] < 0) else float('inf')
        }
        
        return self.metrics
        
    def generate_dashboard_data(self):
        """
        Generate data for Streamlit dashboard
        """
        self.calculate_metrics()
        
        return {
            'performance': self.performance_data,
            'trades': self.trades,
            'positions': self.positions,
            'metrics': self.metrics
        }
        
    def export_to_json(self, file_path):
        """
        Export dashboard data to JSON
        """
        data = self.generate_dashboard_data()
        
        data_serializable = json.loads(
            json.dumps(data, default=lambda obj: obj.isoformat() if isinstance(obj, datetime) else str(obj))
        )
        
        with open(file_path, 'w') as f:
            json.dump(data_serializable, f, indent=2)
            
        return file_path
