
import numpy as np
import cvxpy as cp
from scipy.stats import norm
from scipy.integrate import simpson as simps

def bs_price(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return np.maximum(0, S - K)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_objective(sigma, S, K, T, r, q, C_obs):
    C_model = bs_price(S, K, T, r, sigma, q)
    return np.mean((C_model - C_obs)**2)

def lognormal_pdf(ST, mu, sigma):
    ST = np.maximum(ST, 1e-9)
    return np.exp(-(np.log(ST) - mu)**2 / (2 * sigma**2)) / (ST * sigma * np.sqrt(2 * np.pi))

def spd(ST_grid, pis, mus, sigma):
    f = np.zeros_like(ST_grid, dtype=float)
    for pi, mu in zip(pis, mus):
        if pi > 0: 
            f += pi * lognormal_pdf(ST_grid, mu, sigma)
    return f

def make_true_spd(ST_grid, S0, sigma=0.2, T=30/365.0, n_components=5):
    pi_true = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
    mus_true = np.linspace(np.log(S0 * 0.8), np.log(S0 * 1.2), n_components)
    effective_sigma = sigma 
    spd_true = np.zeros_like(ST_grid, dtype=float)
    for i in range(n_components):
        spd_true += pi_true[i] * lognormal_pdf(ST_grid, mus_true[i], effective_sigma)
    spd_true /= np.trapz(spd_true, ST_grid)
    return pi_true, mus_true, spd_true

def add_strike_dependent_noise(C_true, K, S0):
    rel_distance = np.abs(K - S0) / S0                  
    noise_levels = 0.03 + 0.15 * rel_distance           
    noise = np.random.normal(loc=0.0, scale=noise_levels * C_true)  
    C_obs = np.maximum(C_true + noise, 1e-6)            
    return C_obs, noise                                  

def call_price_from_spd(K, ST_grid, spd_vals, r, T):
    payoff = np.maximum(ST_grid - K, 0)
    integrand = payoff * spd_vals
    price = np.exp(-r * T) * simps(integrand, ST_grid)
    return price

def call_prices_for_components(K_list, ST_grid, mus, sigma, r, T):
    m = len(mus)
    n_options = len(K_list)
    component_prices = np.zeros((m, n_options))
    
    for i, mu in enumerate(mus):
        spd_component = lognormal_pdf(ST_grid, mu, sigma)
        prices_for_component = np.array([call_price_from_spd(k, ST_grid, spd_component, r, T) for k in K_list])
        component_prices[i, :] = prices_for_component
        
    return component_prices


def optimize_pi_cvxpy(mus, K_list, C_obs, ST_grid, r, T, sigma, F_tT, tolerance=0.02):
    m = len(mus)
    component_prices = call_prices_for_components(K_list, ST_grid, mus, sigma, r, T)
    pi_vars = cp.Variable(m, name="pi_weights", nonneg=True)
    model_prices = component_prices.T @ pi_vars
    objective = cp.Minimize(cp.sum_squares(model_prices - C_obs))
    sum_to_one_constraint = [cp.sum(pi_vars) == 1]
    exp_terms = np.exp(mus + sigma**2 / 2)
    martingale_expr = cp.sum(cp.multiply(pi_vars, exp_terms))
    
    if tolerance > 0:
        martingale_constraint = [
            martingale_expr >= F_tT * (1 - tolerance),
            martingale_expr <= F_tT * (1 + tolerance)
        ]
    else:
        martingale_constraint = [martingale_expr == F_tT]

    constraints = sum_to_one_constraint + martingale_constraint
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    if problem.status in ['optimal', 'optimal_inaccurate']:
        return np.maximum(0, pi_vars.value) / np.sum(np.maximum(0, pi_vars.value))
    else:
        print(f"Warning: CVXPY optimization failed with status: {problem.status}. Returning uniform weights.")
        return np.ones(m) / m 

   





