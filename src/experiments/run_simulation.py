import numpy as np
from data_handler import get_simulation_setup
from simulation import run_monte_carlo
from plotting import plot_simulation_study_results

def initialize_mus(S0, T, r, q, sigma_fixed, m=5,width=0.1):
    mu_eq = np.log(S0) + (r - q - 0.5 * sigma_fixed**2) * T
    M = 2.0 * sigma_fixed * np.sqrt(T)
    mus = mu_eq + np.linspace(-M, M, m)
    return mus

def main():
    print("="*30)
    print("Starting simulation study workflow...")
    print("="*30)
    
    setup = get_simulation_setup()
    model_sigma = 0.2
    m = 5 
    model_mus = initialize_mus(setup['S0'], setup['T'], setup['r'], setup['q'], model_sigma, m)
    n_simulations = 500
    study_results = run_monte_carlo(
        setup=setup,
        model_mus=model_mus,
        model_sigma=model_sigma,
        n_runs=n_simulations
    )
    
    plot_simulation_study_results(study_results)
    print("\n" + "="*30)
    print("Simulation study workflow completed successfully!")
    print("="*30)

if __name__ == "__main__":
    main()