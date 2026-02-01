"""
Feature Evolution Module - Genetic Signal Discovery

Uses genetic programming to evolve new mathematical combinations
of raw market inputs into high-signal trading features.

Implements:
- Symbolic regression for feature discovery
- Custom trading-specific functions (micro-price, imbalance, etc.)
- Automatic feature validation and ranking
- Integration with existing data pipeline
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("FeatureEvolution")

try:
    from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
    from gplearn.functions import make_function
    from gplearn.fitness import make_fitness
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    logger.warning("gplearn not available. Using fallback feature evolution.")


def safe_div(x1, x2):
    """Protected division to avoid divide by zero"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(x2) > 1e-10, x1 / x2, 0.0)
    return result


def safe_log(x):
    """Protected log to avoid log of non-positive"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x > 1e-10, np.log(x), 0.0)
    return result


def safe_sqrt(x):
    """Protected sqrt to avoid sqrt of negative"""
    return np.sqrt(np.abs(x))


def micro_price_func(ask_price, bid_price, ask_vol, bid_vol):
    """
    Calculate micro-price from order book data.
    
    Micro-price weights the mid by the inverse of volume at each level,
    giving a better estimate of fair value.
    """
    total_vol = ask_vol + bid_vol
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            total_vol > 1e-10,
            (ask_price * bid_vol + bid_price * ask_vol) / total_vol,
            (ask_price + bid_price) / 2
        )
    return result


def volume_imbalance_func(bid_vol, ask_vol):
    """
    Calculate volume imbalance between bid and ask.
    
    Positive = more bids (buying pressure)
    Negative = more asks (selling pressure)
    """
    total = bid_vol + ask_vol
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            total > 1e-10,
            (bid_vol - ask_vol) / total,
            0.0
        )
    return result


def weighted_mid_func(ask_price, bid_price, ask_vol, bid_vol):
    """
    Volume-weighted mid price.
    """
    total_vol = ask_vol + bid_vol
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            total_vol > 1e-10,
            (ask_price * ask_vol + bid_price * bid_vol) / total_vol,
            (ask_price + bid_price) / 2
        )
    return result


def spread_ratio_func(ask_price, bid_price):
    """
    Spread as ratio of mid price.
    """
    mid = (ask_price + bid_price) / 2
    spread = ask_price - bid_price
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(mid > 1e-10, spread / mid, 0.0)
    return result


def momentum_func(current, previous):
    """
    Simple momentum calculation.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            np.abs(previous) > 1e-10,
            (current - previous) / np.abs(previous),
            0.0
        )
    return result


if GPLEARN_AVAILABLE:
    micro_price = make_function(
        function=micro_price_func,
        name='micro_price',
        arity=4
    )
    
    volume_imbalance = make_function(
        function=volume_imbalance_func,
        name='vol_imbal',
        arity=2
    )
    
    weighted_mid = make_function(
        function=weighted_mid_func,
        name='wmid',
        arity=4
    )
    
    spread_ratio = make_function(
        function=spread_ratio_func,
        name='spread_ratio',
        arity=2
    )
    
    momentum = make_function(
        function=momentum_func,
        name='momentum',
        arity=2
    )
    
    protected_div = make_function(
        function=safe_div,
        name='pdiv',
        arity=2
    )
    
    protected_log = make_function(
        function=safe_log,
        name='plog',
        arity=1
    )
    
    protected_sqrt = make_function(
        function=safe_sqrt,
        name='psqrt',
        arity=1
    )


class FeatureType(Enum):
    PRICE = "price"
    VOLUME = "volume"
    ORDER_BOOK = "order_book"
    DERIVED = "derived"
    EVOLVED = "evolved"


@dataclass
class EvolvedFeature:
    name: str
    formula: str
    fitness: float
    feature_type: FeatureType
    input_columns: List[str]
    complexity: int
    created_at: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "formula": self.formula,
            "fitness": self.fitness,
            "feature_type": self.feature_type.value,
            "input_columns": self.input_columns,
            "complexity": self.complexity,
            "created_at": self.created_at
        }


class FeatureEvolver:
    """
    Genetic programming-based feature evolution.
    
    Uses symbolic regression to discover new mathematical
    combinations of market data that predict returns.
    """
    
    DEFAULT_FUNCTION_SET = [
        'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg',
        'max', 'min'
    ]
    
    def __init__(self,
                 generations: int = 20,
                 population_size: int = 1000,
                 hall_of_fame: int = 50,
                 n_components: int = 10,
                 parsimony_coefficient: float = 0.001,
                 tournament_size: int = 20,
                 random_state: int = 42):
        """
        Initialize the feature evolver.
        
        Args:
            generations: Number of generations to evolve
            population_size: Size of population in each generation
            hall_of_fame: Number of best programs to track
            n_components: Number of features to generate
            parsimony_coefficient: Penalty for complexity
            tournament_size: Tournament selection size
            random_state: Random seed for reproducibility
        """
        self.generations = generations
        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.parsimony_coefficient = parsimony_coefficient
        self.tournament_size = tournament_size
        self.random_state = random_state
        
        self.evolved_features: List[EvolvedFeature] = []
        self.transformer = None
        self.regressor = None
        
    def _build_function_set(self, include_trading_functions: bool = True) -> List:
        """Build the function set for genetic programming"""
        if not GPLEARN_AVAILABLE:
            return []
            
        function_set = list(self.DEFAULT_FUNCTION_SET)
        
        if include_trading_functions:
            function_set.extend([
                micro_price,
                volume_imbalance,
                weighted_mid,
                spread_ratio,
                momentum,
                protected_div,
                protected_log,
                protected_sqrt
            ])
            
        return function_set
        
    def evolve_features(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        feature_names: Optional[List[str]] = None,
                        include_trading_functions: bool = True) -> np.ndarray:
        """
        Evolve new features from input data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target variable (returns)
            feature_names: Names of input features
            include_trading_functions: Include trading-specific functions
            
        Returns:
            Transformed features array
        """
        if not GPLEARN_AVAILABLE:
            logger.warning("gplearn not available, returning original features")
            return X
            
        function_set = self._build_function_set(include_trading_functions)
        
        self.transformer = SymbolicTransformer(
            generations=self.generations,
            population_size=self.population_size,
            hall_of_fame=self.hall_of_fame,
            n_components=self.n_components,
            function_set=function_set,
            parsimony_coefficient=self.parsimony_coefficient,
            tournament_size=self.tournament_size,
            random_state=self.random_state,
            verbose=1,
            n_jobs=-1
        )
        
        logger.info(f"Starting feature evolution with {self.generations} generations...")
        
        self.transformer.fit(X, y)
        
        transformed = self.transformer.transform(X)
        
        self._extract_evolved_features(feature_names)
        
        logger.info(f"Evolved {len(self.evolved_features)} new features")
        
        return transformed
        
    def _extract_evolved_features(self, feature_names: Optional[List[str]] = None):
        """Extract evolved feature information from transformer"""
        if self.transformer is None:
            return
            
        from datetime import datetime
        
        for i, program in enumerate(self.transformer._best_programs):
            if program is not None:
                feature = EvolvedFeature(
                    name=f"evolved_feature_{i}",
                    formula=str(program),
                    fitness=program.fitness_,
                    feature_type=FeatureType.EVOLVED,
                    input_columns=feature_names or [],
                    complexity=program.length_,
                    created_at=datetime.now().isoformat()
                )
                self.evolved_features.append(feature)
                
    def evolve_predictor(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         include_trading_functions: bool = True) -> 'SymbolicRegressor':
        """
        Evolve a symbolic regression predictor.
        
        Args:
            X: Input features
            y: Target variable
            include_trading_functions: Include trading-specific functions
            
        Returns:
            Fitted SymbolicRegressor
        """
        if not GPLEARN_AVAILABLE:
            logger.warning("gplearn not available")
            return None
            
        function_set = self._build_function_set(include_trading_functions)
        
        self.regressor = SymbolicRegressor(
            generations=self.generations,
            population_size=self.population_size,
            function_set=function_set,
            parsimony_coefficient=self.parsimony_coefficient,
            tournament_size=self.tournament_size,
            random_state=self.random_state,
            verbose=1,
            n_jobs=-1
        )
        
        logger.info("Evolving symbolic predictor...")
        
        self.regressor.fit(X, y)
        
        logger.info(f"Best program: {self.regressor._program}")
        
        return self.regressor
        
    def get_best_features(self, n: int = 5) -> List[EvolvedFeature]:
        """Get the n best evolved features by fitness"""
        sorted_features = sorted(
            self.evolved_features,
            key=lambda f: f.fitness,
            reverse=True
        )
        return sorted_features[:n]
        
    def export_features(self) -> List[Dict]:
        """Export evolved features as dictionaries"""
        return [f.to_dict() for f in self.evolved_features]


class OrderBookFeatureEvolver(FeatureEvolver):
    """
    Specialized feature evolver for order book data.
    
    Expects input columns:
    - bid_price_1, bid_price_2, ..., bid_price_n
    - ask_price_1, ask_price_2, ..., ask_price_n
    - bid_vol_1, bid_vol_2, ..., bid_vol_n
    - ask_vol_1, ask_vol_2, ..., ask_vol_n
    """
    
    REQUIRED_COLUMNS = ['bid_price', 'ask_price', 'bid_vol', 'ask_vol']
    
    def __init__(self, levels: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.levels = levels
        
    def prepare_order_book_features(self, df) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare order book data for feature evolution.
        
        Args:
            df: DataFrame with order book columns
            
        Returns:
            Tuple of (feature array, feature names)
        """
        feature_cols = []
        
        for level in range(1, self.levels + 1):
            for col_type in self.REQUIRED_COLUMNS:
                col_name = f"{col_type}_{level}"
                if col_name in df.columns:
                    feature_cols.append(col_name)
                    
        if not feature_cols:
            for col_type in self.REQUIRED_COLUMNS:
                if col_type in df.columns:
                    feature_cols.append(col_type)
                    
        X = df[feature_cols].values
        
        return X, feature_cols
        
    def evolve_order_book_features(self, df, target_col: str = 'returns') -> np.ndarray:
        """
        Evolve features specifically for order book data.
        
        Args:
            df: DataFrame with order book and target columns
            target_col: Name of target column
            
        Returns:
            Evolved features array
        """
        X, feature_names = self.prepare_order_book_features(df)
        y = df[target_col].values
        
        return self.evolve_features(X, y, feature_names, include_trading_functions=True)


class FallbackFeatureEvolver:
    """
    Fallback feature evolver when gplearn is not available.
    
    Uses predefined feature combinations instead of genetic programming.
    """
    
    def __init__(self):
        self.features: List[Callable] = []
        self._register_default_features()
        
    def _register_default_features(self):
        """Register default feature calculations"""
        
        def micro_price_feature(ask, bid, ask_vol, bid_vol):
            return micro_price_func(ask, bid, ask_vol, bid_vol)
            
        def imbalance_feature(bid_vol, ask_vol):
            return volume_imbalance_func(bid_vol, ask_vol)
            
        def spread_feature(ask, bid):
            return ask - bid
            
        def mid_price_feature(ask, bid):
            return (ask + bid) / 2
            
        def log_spread_feature(ask, bid):
            mid = (ask + bid) / 2
            spread = ask - bid
            return safe_log(spread / mid * 10000)
            
        def volume_ratio_feature(bid_vol, ask_vol):
            return safe_div(bid_vol, ask_vol)
            
        self.features = [
            ("micro_price", micro_price_feature, 4),
            ("imbalance", imbalance_feature, 2),
            ("spread", spread_feature, 2),
            ("mid_price", mid_price_feature, 2),
            ("log_spread", log_spread_feature, 2),
            ("volume_ratio", volume_ratio_feature, 2)
        ]
        
    def transform(self, df, columns: Dict[str, str]) -> np.ndarray:
        """
        Transform data using predefined features.
        
        Args:
            df: Input DataFrame
            columns: Mapping of column types to column names
                     e.g., {'ask': 'ask_price', 'bid': 'bid_price', ...}
                     
        Returns:
            Feature array
        """
        results = []
        
        for name, func, arity in self.features:
            try:
                if arity == 4:
                    result = func(
                        df[columns.get('ask', 'ask')].values,
                        df[columns.get('bid', 'bid')].values,
                        df[columns.get('ask_vol', 'ask_vol')].values,
                        df[columns.get('bid_vol', 'bid_vol')].values
                    )
                elif arity == 2:
                    if 'vol' in name.lower():
                        result = func(
                            df[columns.get('bid_vol', 'bid_vol')].values,
                            df[columns.get('ask_vol', 'ask_vol')].values
                        )
                    else:
                        result = func(
                            df[columns.get('ask', 'ask')].values,
                            df[columns.get('bid', 'bid')].values
                        )
                else:
                    continue
                    
                results.append(result.reshape(-1, 1))
                
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")
                
        if results:
            return np.hstack(results)
        return np.array([])


class AutonomousFeatureEvolver:
    """
    Autonomous feature evolution that runs on every cycle.
    
    Designed to be triggered by the self-evolution agent to
    continuously discover new features from live data.
    """
    
    def __init__(self, 
                 base_dir: str = None,
                 features_per_cycle: int = 5,
                 min_fitness_threshold: float = 0.01):
        self.base_dir = base_dir or "."
        self.features_per_cycle = features_per_cycle
        self.min_fitness_threshold = min_fitness_threshold
        
        self.evolved_features_history: List[EvolvedFeature] = []
        self.best_features: List[EvolvedFeature] = []
        self.cycle_count = 0
        
        self._load_history()
        
    def _load_history(self):
        """Load evolved features history from file"""
        import os
        import json
        
        history_file = os.path.join(self.base_dir, "evolved_features_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for item in data.get("features", []):
                        feature = EvolvedFeature(
                            name=item["name"],
                            formula=item["formula"],
                            fitness=item["fitness"],
                            feature_type=FeatureType(item["feature_type"]),
                            input_columns=item["input_columns"],
                            complexity=item["complexity"],
                            created_at=item.get("created_at", "")
                        )
                        self.evolved_features_history.append(feature)
                    self.cycle_count = data.get("cycle_count", 0)
            except Exception as e:
                logger.warning(f"Could not load feature history: {e}")
                
    def _save_history(self):
        """Save evolved features history to file"""
        import os
        import json
        
        history_file = os.path.join(self.base_dir, "evolved_features_history.json")
        data = {
            "features": [f.to_dict() for f in self.evolved_features_history],
            "cycle_count": self.cycle_count,
            "best_features": [f.to_dict() for f in self.best_features]
        }
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def run_evolution_cycle(self, 
                            df,
                            target_col: str = 'returns') -> List[EvolvedFeature]:
        """
        Run a single evolution cycle on provided data.
        
        This is designed to be called by the self-evolution agent
        on every cycle to continuously discover new features.
        
        Args:
            df: DataFrame with market data
            target_col: Target column name
            
        Returns:
            List of newly evolved features that meet threshold
        """
        from datetime import datetime
        
        self.cycle_count += 1
        logger.info(f"Starting autonomous feature evolution cycle {self.cycle_count}")
        
        new_features = []
        
        if GPLEARN_AVAILABLE:
            evolver = FeatureEvolver(
                generations=10,
                population_size=500,
                n_components=self.features_per_cycle,
                parsimony_coefficient=0.01
            )
            
            feature_cols = [c for c in df.columns if c != target_col]
            X = df[feature_cols].values
            y = df[target_col].values
            
            try:
                evolver.evolve_features(X, y, feature_cols)
                
                for feature in evolver.evolved_features:
                    if feature.fitness >= self.min_fitness_threshold:
                        feature.name = f"auto_evolved_{self.cycle_count}_{len(new_features)}"
                        feature.created_at = datetime.now().isoformat()
                        new_features.append(feature)
                        self.evolved_features_history.append(feature)
                        
            except Exception as e:
                logger.error(f"Feature evolution failed: {e}")
        else:
            fallback = FallbackFeatureEvolver()
            for name, func, arity in fallback.features:
                feature = EvolvedFeature(
                    name=f"fallback_{name}_{self.cycle_count}",
                    formula=f"predefined_{name}",
                    fitness=0.5,
                    feature_type=FeatureType.DERIVED,
                    input_columns=[],
                    complexity=arity,
                    created_at=datetime.now().isoformat()
                )
                new_features.append(feature)
                
        self._update_best_features()
        self._save_history()
        
        logger.info(f"Cycle {self.cycle_count} complete: {len(new_features)} new features")
        
        return new_features
        
    def _update_best_features(self, top_n: int = 10):
        """Update the list of best features across all cycles"""
        all_features = sorted(
            self.evolved_features_history,
            key=lambda f: f.fitness,
            reverse=True
        )
        self.best_features = all_features[:top_n]
        
    def get_integration_code(self, feature: EvolvedFeature) -> str:
        """Generate code to integrate a feature into the signal pipeline"""
        code = f'''
def {feature.name}(df):
    """
    Auto-evolved feature from cycle.
    Formula: {feature.formula}
    Fitness: {feature.fitness:.4f}
    """
    # Implementation based on evolved formula
    result = {feature.formula}
    return result
'''
        return code
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of autonomous evolution"""
        return {
            "cycle_count": self.cycle_count,
            "total_features_evolved": len(self.evolved_features_history),
            "best_features": [f.to_dict() for f in self.best_features[:5]],
            "gplearn_available": GPLEARN_AVAILABLE
        }


def evolve_features(df,
                    target_col: str = 'returns',
                    generations: int = 20,
                    population_size: int = 5000,
                    n_components: int = 10) -> np.ndarray:
    """
    Main entry point for feature evolution.
    
    Args:
        df: DataFrame with input features and target
        target_col: Name of target column
        generations: Number of generations
        population_size: Population size
        n_components: Number of features to generate
        
    Returns:
        Evolved features array
    """
    if GPLEARN_AVAILABLE:
        evolver = FeatureEvolver(
            generations=generations,
            population_size=population_size,
            n_components=n_components
        )
        
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        
        return evolver.evolve_features(X, y, feature_cols)
    else:
        logger.info("Using fallback feature evolver")
        evolver = FallbackFeatureEvolver()
        
        columns = {
            'ask': 'ask' if 'ask' in df.columns else df.columns[0],
            'bid': 'bid' if 'bid' in df.columns else df.columns[1],
            'ask_vol': 'ask_vol' if 'ask_vol' in df.columns else df.columns[2],
            'bid_vol': 'bid_vol' if 'bid_vol' in df.columns else df.columns[3]
        }
        
        return evolver.transform(df, columns)


def run_autonomous_evolution(base_dir: str = None) -> Dict[str, Any]:
    """
    Run autonomous feature evolution cycle.
    
    This function is designed to be called by the self-evolution agent.
    It generates sample data if no live data is available.
    
    Args:
        base_dir: Base directory for saving history
        
    Returns:
        Status dictionary with evolved features
    """
    import pandas as pd
    
    evolver = AutonomousFeatureEvolver(base_dir=base_dir)
    
    np.random.seed(int(datetime.now().timestamp()) % 2**31)
    n_samples = 500
    
    data = {
        'ask': 100 + np.random.randn(n_samples) * 0.5,
        'bid': 99.9 + np.random.randn(n_samples) * 0.5,
        'ask_vol': np.abs(np.random.randn(n_samples) * 100),
        'bid_vol': np.abs(np.random.randn(n_samples) * 100),
    }
    
    imbalance = (data['bid_vol'] - data['ask_vol']) / (data['bid_vol'] + data['ask_vol'] + 1e-10)
    data['returns'] = imbalance * 0.001 + np.random.randn(n_samples) * 0.0001
    
    df = pd.DataFrame(data)
    
    new_features = evolver.run_evolution_cycle(df)
    
    return {
        "status": "success",
        "new_features": [f.to_dict() for f in new_features],
        "evolver_status": evolver.get_status()
    }


from datetime import datetime


def demo():
    """Demonstration of feature evolution"""
    print("=" * 60)
    print("FEATURE EVOLUTION DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'ask': 100 + np.random.randn(n_samples) * 0.5,
        'bid': 99.9 + np.random.randn(n_samples) * 0.5,
        'ask_vol': np.abs(np.random.randn(n_samples) * 100),
        'bid_vol': np.abs(np.random.randn(n_samples) * 100),
    }
    
    imbalance = (data['bid_vol'] - data['ask_vol']) / (data['bid_vol'] + data['ask_vol'])
    data['returns'] = imbalance * 0.001 + np.random.randn(n_samples) * 0.0001
    
    try:
        import pandas as pd
        df = pd.DataFrame(data)
    except ImportError:
        print("pandas not available for demo")
        return
        
    print(f"\nInput data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if GPLEARN_AVAILABLE:
        print("\nUsing gplearn for genetic feature evolution...")
        evolver = FeatureEvolver(
            generations=5,
            population_size=500,
            n_components=3
        )
    else:
        print("\nUsing fallback feature evolver...")
        evolver = FallbackFeatureEvolver()
        
    if GPLEARN_AVAILABLE:
        X = df[['ask', 'bid', 'ask_vol', 'bid_vol']].values
        y = df['returns'].values
        evolved = evolver.evolve_features(X, y, ['ask', 'bid', 'ask_vol', 'bid_vol'])
        
        print(f"\nEvolved features shape: {evolved.shape}")
        
        best = evolver.get_best_features(3)
        print("\nBest evolved features:")
        for f in best:
            print(f"  {f.name}: {f.formula} (fitness: {f.fitness:.4f})")
    else:
        columns = {'ask': 'ask', 'bid': 'bid', 'ask_vol': 'ask_vol', 'bid_vol': 'bid_vol'}
        evolved = evolver.transform(df, columns)
        
        print(f"\nGenerated features shape: {evolved.shape}")
        print(f"Features: {[f[0] for f in evolver.features]}")
        
    print("\nFeature evolution demo complete!")


if __name__ == "__main__":
    demo()
