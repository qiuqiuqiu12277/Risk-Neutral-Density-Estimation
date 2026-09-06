
import numpy as np
from tqdm import tqdm 
from scipy.optimize import minimize_scalar
from spd_model import make_true_spd, call_price_from_spd, optimize_pi_cvxpy, spd, bs_objective, bs_price,lognormal_pdf, add_strike_dependent_noise

def run_monte_carlo(setup, model_mus, model_sigma, n_runs=500):
    print(f"\nStarting self-contained Monte Carlo simulation with {n_runs} replications...")

    K, S0, T, r, q = setup['K'], setup['S0'], setup['T'], setup['r'], setup['q']
    ST_grid = np.linspace(S0 * 0.5, S0 * 2.0, 500)
    true_sigma = 0.2
    pi_true, mus_true, spd_true = make_true_spd(ST_grid, S0, sigma=true_sigma, T=T, n_components=5)
    C_true = np.array([call_price_from_spd(k, ST_grid, spd_true, r, T) for k in K])
    C_obs, noise = add_strike_dependent_noise(C_true, K, S0)

    price_samples = np.zeros((n_runs, len(K)))
    spd_samples = np.zeros((n_runs, len(ST_grid)))
    bs_vol_samples = np.zeros(n_runs) 

    F_tT = S0 * np.exp((r - q) * T)
    n_components = len(model_mus)

    try:
        res_vol = minimize_scalar(bs_objective, bounds=(0.01, 1.0), method='bounded', 
                                args=(S0, K, T, r, q, C_true))
        initial_bs_vol = res_vol.x
        print(f"Initial Black-Scholes volatility fit: {initial_bs_vol:.4f}")
    except:
        initial_bs_vol = 0.2
        print(f"Failed to fit initial BS volatility, using default: {initial_bs_vol:.4f}")

    for i in tqdm(range(n_runs), desc="Monte Carlo simulation progress"):
        try:
            C_obs, noise = add_strike_dependent_noise(C_true, K, S0)
            try:
                res_vol = minimize_scalar(bs_objective, bounds=(0.01, 1.0), method='bounded', 
                                       args=(S0, K, T, r, q, C_obs))
                bs_vol = res_vol.x
                bs_vol_samples[i] = bs_vol
                sigma_for_model = 0.75 * bs_vol* np.sqrt(T)
            except:
                bs_vol = initial_bs_vol
                bs_vol_samples[i] = bs_vol
                sigma_for_model = 0.75 * bs_vol* np.sqrt(T)
                print(f"\nWarning: BS vol fit failed in run {i+1}, using initial vol")
            
            mu_eq = np.log(S0) + (r - q - 0.5 * sigma_for_model**2) * T
            M = 2.0 * sigma_for_model 
            iter_model_mus = mu_eq + np.linspace(-M, M, n_components)
            pi_est = optimize_pi_cvxpy(iter_model_mus, K, C_obs, ST_grid, r, T, sigma_for_model, F_tT)
            
            spd_est_run = spd(ST_grid, pi_est, iter_model_mus, sigma_for_model)
            C_est_run = np.array([call_price_from_spd(k, ST_grid, spd_est_run, r, T) for k in K])
            mu_bs = np.log(S0) + (r - q - 0.5 * bs_vol**2) * T
            bs_spd = lognormal_pdf(ST_grid, mu_bs, bs_vol*np.sqrt(T))
            bs_prices = np.array([bs_price(S0, k, T, r, bs_vol, q) for k in K])

            price_samples[i, :] = C_est_run
            spd_samples[i, :] = spd_est_run

        except Exception as e:
            print(f"\nWarning: Simulation run {i+1} failed: {str(e)}")
            if i > 0:
                price_samples[i, :] = price_samples[i-1, :]
                spd_samples[i, :] = spd_samples[i-1, :]
                bs_vol_samples[i] = bs_vol_samples[i-1]
            else:
                price_samples[i, :] = C_true
                spd_samples[i, :] = np.zeros_like(ST_grid)
                bs_vol_samples[i] = initial_bs_vol

    print("Simulation loop ended. Analyzing results...")
     

    results = {
        'K': K, 'ST_grid': ST_grid, 'S0': S0,
        'C_true': C_true, 'C_obs': C_obs, 'spd_true': spd_true, 'pi_true': pi_true,
        'mean_est_prices': np.mean(price_samples, axis=0),
        'lower_ci_prices': np.percentile(price_samples, 2.5, axis=0),
        'upper_ci_prices': np.percentile(price_samples, 97.5, axis=0),
        'mean_est_spd': np.mean(spd_samples, axis=0),
        'lower_ci_spd': np.percentile(spd_samples, 2.5, axis=0),
        'upper_ci_spd': np.percentile(spd_samples, 97.5, axis=0),
        'mse_prices': np.mean((price_samples - C_true)**2, axis=0),
        'price_samples': price_samples, 'spd_samples': spd_samples,
        'bs_prices': bs_prices,'bs_spd': bs_spd,
        'bs_vol': np.mean(bs_vol_samples),
        'r': r, 'q': q, 'T': T
    }

    print("Monte Carlo study completed.")
    return results