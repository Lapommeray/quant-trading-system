import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class AssetClass(Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITIES = "commodities"
    INDICES = "indices"

class DNAOverlord:
    def __init__(self):
        self.assets = [
            {'symbol': 'BTC', 'class': AssetClass.CRYPTO},
            {'symbol': 'ETH', 'class': AssetClass.CRYPTO},
            {'symbol': 'XRP', 'class': AssetClass.CRYPTO},
            {'symbol': 'EUR/USD', 'class': AssetClass.FOREX},
            {'symbol': 'GBP/USD', 'class': AssetClass.FOREX},
            {'symbol': 'USD/JPY', 'class': AssetClass.FOREX},
            {'symbol': 'GOLD', 'class': AssetClass.COMMODITIES},
            {'symbol': 'SILVER', 'class': AssetClass.COMMODITIES},
            {'symbol': 'OIL', 'class': AssetClass.COMMODITIES},
            {'symbol': 'SPY', 'class': AssetClass.INDICES},
            {'symbol': 'QQQ', 'class': AssetClass.INDICES},
            {'symbol': 'DIA', 'class': AssetClass.INDICES}
        ]
        self.last_scan = datetime.now().timestamp()

    def _calculate_dominance_score(self, spirit: float, liquidity: float) -> float:
        """Computes unified dominance metric"""
        return 0.7 * spirit + 0.3 * liquidity

    def evaluate_asset(self, symbol: str, asset_class: AssetClass) -> float:
        """Evaluate asset dominance score based on spirit and liquidity metrics"""
        try:
            base_score = 0.5
            
            spirit_multiplier = {
                AssetClass.CRYPTO: 1.2,
                AssetClass.FOREX: 1.0,
                AssetClass.COMMODITIES: 1.1,
                AssetClass.INDICES: 0.9
            }.get(asset_class, 1.0)
            
            liquidity_bonus = 0.0
            if symbol in ['BTC', 'ETH', 'EUR/USD', 'GBP/USD', 'GOLD', 'SPY', 'QQQ']:
                liquidity_bonus = 0.3
            elif symbol in ['XRP', 'ADA', 'USD/JPY', 'SILVER', 'DIA', 'IWM']:
                liquidity_bonus = 0.2
            else:
                liquidity_bonus = 0.1
            
            dominance_score = (base_score * spirit_multiplier) + liquidity_bonus
            
            return min(1.0, max(0.0, dominance_score))
            
        except Exception:
            return 0.5

    def evaluate_assets(self) -> Dict[AssetClass, List[Dict]]:
        """Scores all assets across dimensions"""
        results = {}
        
        for asset_class in AssetClass:
            class_assets = [asset for asset in self.assets if asset['class'] == asset_class]
            
            for asset in class_assets:
                asset['score'] = self.evaluate_asset(asset['symbol'], asset['class'])
            
            sorted_assets = sorted(class_assets, key=lambda x: x['score'], reverse=True)
            results[asset_class] = sorted_assets
            
        return results

    def select_hierarchy(self) -> Dict[str, str]:
        """Select top asset from each class based on dominance scores"""
        try:
            hierarchy = {}
            
            for asset_class in AssetClass:
                class_assets = [asset for asset in self.assets if asset['class'] == asset_class]
                
                if not class_assets:
                    continue
                
                best_asset = None
                best_score = -1
                
                for asset in class_assets:
                    score = self.evaluate_asset(asset['symbol'], asset['class'])
                    if score > best_score:
                        best_score = score
                        best_asset = asset
                
                if best_asset:
                    hierarchy[asset_class.value] = best_asset['symbol']
            
            self.last_scan = datetime.now().timestamp()
            return hierarchy
            
        except Exception:
            return {
                'crypto': 'BTC',
                'forex': 'EUR/USD', 
                'commodities': 'GOLD',
                'indices': 'SPY'
            }
