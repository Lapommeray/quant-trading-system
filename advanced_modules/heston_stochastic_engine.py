import numpy as np
import torch
import torchsde
import logging
from datetime import datetime

class HestonModel(torch.nn.Module):
    """
    Heston stochastic volatility model implementation using torchsde
    """
    def __init__(self, kappa=1.0, theta=0.04, xi=0.2, rho=-0.7):
        super().__init__()
        self.kappa = kappa  # Mean reversion speed
        self.theta = theta  # Long-term variance
        self.xi = xi        # Volatility of volatility
        self.rho = rho      # Correlation between price and volatility
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def f(self, t, state):
        """
        Drift function for the SDE
        """
        s, v = state[:, 0], state[:, 1]
        ds = -0.5 * v * s  # Risk-neutral drift
        dv = self.kappa * (self.theta - v)  # Mean-reverting variance process
        return torch.stack([ds, dv], dim=1)
        
    def g(self, t, state):
        """
        Diffusion function for the SDE
        """
        s, v = state[:, 0], state[:, 1]
        dWs = torch.sqrt(v) * s  # Price diffusion
        dWv = self.xi * torch.sqrt(v)  # Volatility diffusion
        
        g1 = torch.stack([dWs, torch.zeros_like(dWs)], dim=1)
        g2 = torch.stack([self.rho * dWv, torch.sqrt(1 - self.rho**2) * dWv], dim=1)
        
        return torch.stack([g1, g2], dim=2)
        
    def simulate_paths(self, initial_state, times, dt=0.01, num_paths=1):
        """
        Simulate price paths using the Heston model
        
        Args:
            initial_state: Tensor of shape (num_paths, 2) with initial price and variance
            times: Tensor of time points
            dt: Time step for simulation
            num_paths: Number of paths to simulate
            
        Returns:
            Tensor of shape (len(times), num_paths, 2) with price and variance paths
        """
        try:
            if not isinstance(initial_state, torch.Tensor):
                initial_state = torch.tensor(initial_state, dtype=torch.float32)
                
            if initial_state.dim() == 1:
                initial_state = initial_state.unsqueeze(0)
                
            if not isinstance(times, torch.Tensor):
                times = torch.tensor(times, dtype=torch.float32)
                
            initial_state[:, 1] = torch.clamp(initial_state[:, 1], min=1e-6)
                
            paths = torchsde.sdeint(self, initial_state, times, dt=dt)
            
            paths[:, :, 1] = torch.clamp(paths[:, :, 1], min=1e-6)
            
            return paths
        except Exception as e:
            self.logger.error(f"Error simulating paths: {str(e)}")
            return None
            
    def calibrate(self, market_data, initial_params=None, learning_rate=0.01, epochs=100):
        """
        Calibrate model parameters to market data
        
        Args:
            market_data: DataFrame with columns 'price' and 'implied_vol'
            initial_params: Dictionary with initial parameters
            learning_rate: Learning rate for optimization
            epochs: Number of epochs for optimization
            
        Returns:
            Dictionary with calibrated parameters
        """
        try:
            if initial_params:
                self.kappa = initial_params.get('kappa', self.kappa)
                self.theta = initial_params.get('theta', self.theta)
                self.xi = initial_params.get('xi', self.xi)
                self.rho = initial_params.get('rho', self.rho)
                
            prices = torch.tensor(market_data['price'].values, dtype=torch.float32)
            implied_vols = torch.tensor(market_data['implied_vol'].values, dtype=torch.float32)
            
            kappa = torch.tensor(self.kappa, requires_grad=True)
            theta = torch.tensor(self.theta, requires_grad=True)
            xi = torch.tensor(self.xi, requires_grad=True)
            rho = torch.tensor(self.rho, requires_grad=True)
            
            optimizer = torch.optim.Adam([kappa, theta, xi, rho], lr=learning_rate)
            
            for epoch in range(epochs):
                initial_state = torch.tensor([[prices[0], implied_vols[0]**2]], dtype=torch.float32)
                times = torch.linspace(0, 1, len(prices))
                
                self.kappa = kappa.item()
                self.theta = theta.item()
                self.xi = xi.item()
                self.rho = rho.item()
                
                paths = self.simulate_paths(initial_state, times)
                
                if paths is None:
                    continue
                    
                simulated_prices = paths[:, 0, 0]
                simulated_vols = torch.sqrt(paths[:, 0, 1])
                
                price_loss = torch.mean((simulated_prices - prices)**2)
                vol_loss = torch.mean((simulated_vols - implied_vols)**2)
                
                loss = price_loss + vol_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    kappa.clamp_(0.1, 10.0)
                    theta.clamp_(0.001, 0.1)
                    xi.clamp_(0.01, 1.0)
                    rho.clamp_(-0.99, 0.99)
                    
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                    
            self.kappa = kappa.item()
            self.theta = theta.item()
            self.xi = xi.item()
            self.rho = rho.item()
            
            return {
                'kappa': self.kappa,
                'theta': self.theta,
                'xi': self.xi,
                'rho': self.rho
            }
        except Exception as e:
            self.logger.error(f"Error calibrating model: {str(e)}")
            return {
                'kappa': self.kappa,
                'theta': self.theta,
                'xi': self.xi,
                'rho': self.rho
            }
            
    def price_options(self, spot, strike, maturity, risk_free_rate=0.0, num_paths=10000):
        """
        Price European options using Monte Carlo simulation
        
        Args:
            spot: Initial spot price
            strike: Option strike price
            maturity: Option maturity in years
            risk_free_rate: Risk-free interest rate
            num_paths: Number of Monte Carlo paths
            
        Returns:
            Dictionary with call and put prices
        """
        try:
            initial_state = torch.tensor([[spot, self.theta]], dtype=torch.float32).repeat(num_paths, 1)
            
            initial_state[:, 1] = torch.clamp(
                initial_state[:, 1] * (1 + 0.1 * torch.randn(num_paths)),
                min=1e-6
            )
            
            times = torch.linspace(0, maturity, 252)
            
            paths = self.simulate_paths(initial_state, times)
            
            if paths is None:
                return {'call': None, 'put': None}
                
            terminal_prices = paths[-1, :, 0]
            
            call_payoffs = torch.clamp(terminal_prices - strike, min=0)
            put_payoffs = torch.clamp(strike - terminal_prices, min=0)
            
            discount_factor = torch.exp(-risk_free_rate * maturity)
            call_price = discount_factor * torch.mean(call_payoffs)
            put_price = discount_factor * torch.mean(put_payoffs)
            
            return {
                'call': call_price.item(),
                'put': put_price.item()
            }
        except Exception as e:
            self.logger.error(f"Error pricing options: {str(e)}")
            return {'call': None, 'put': None}
