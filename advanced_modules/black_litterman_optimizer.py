import cvxpy as cp
import numpy as np

def black_litterman_optimization(returns, cov_matrix, tau=0.05, P=None, Q=None, omega=None):
    """
    Black-Litterman portfolio optimization
    
    Parameters:
    - returns: Expected returns vector
    - cov_matrix: Covariance matrix
    - tau: Uncertainty parameter
    - P: Picking matrix (views)
    - Q: View portfolio returns
    - omega: Uncertainty matrix for views
    """
    n = len(returns)
    
    if P is None:  # No views - use market equilibrium
        return np.ones(n) / n
    
    if omega is None:
        omega = tau * (P @ cov_matrix @ P.T)
    
    tau_sigma = tau * cov_matrix
    M1 = np.linalg.inv(tau_sigma)
    M2 = P.T @ np.linalg.inv(omega) @ P
    M3 = np.linalg.inv(tau_sigma) @ returns
    M4 = P.T @ np.linalg.inv(omega) @ Q
    
    mu_bl = np.linalg.inv(M1 + M2) @ (M3 + M4)
    sigma_bl = np.linalg.inv(M1 + M2)
    
    w = cp.Variable(n)
    objective = cp.Maximize(mu_bl.T @ w - 0.5 * cp.quad_form(w, sigma_bl))
    constraints = [cp.sum(w) == 1, w >= 0]  # Long-only constraint
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return w.value if w.value is not None else np.ones(n) / n
