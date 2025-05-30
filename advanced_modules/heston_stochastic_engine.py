import torch
import torchsde
import numpy as np

class HestonModel(torch.nn.Module):
    """Stochastic volatility model (Heston)"""
    def __init__(self, kappa=1.0, theta=0.04, xi=0.2, rho=-0.7):
        super().__init__()
        self.kappa, self.theta, self.xi, self.rho = kappa, theta, xi, rho
    
    def f(self, t, state):  # Drift
        s, v = state[:, 0], state[:, 1]
        ds = -0.5 * v * s
        dv = self.kappa * (self.theta - v)
        return torch.stack([ds, dv], dim=1)
    
    def g(self, t, state):  # Diffusion
        s, v = state[:, 0], state[:, 1]
        dWs = torch.sqrt(v) * s
        dWv = self.xi * torch.sqrt(v)
        return torch.stack([dWs, self.rho * dWv], dim=1)

def simulate_heston_paths(initial_spot=1.0, initial_vol=0.04, T=1.0, steps=252):
    sde = HestonModel()
    state0 = torch.tensor([[initial_spot, initial_vol]])  # (spot, vol)
    ts = torch.linspace(0, T, steps)
    paths = torchsde.sdeint(sde, state0, ts, dt=0.01)
    return paths
