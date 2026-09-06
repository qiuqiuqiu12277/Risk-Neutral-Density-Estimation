import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import os
from spd_model import lognormal_pdf

sns.set_theme(style="whitegrid", palette="deep")

def plot_simulation_study_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Simulation Study Results Analysis", fontsize=16)
    
    K = results['K']
    ST_grid = results['ST_grid']
    S0 = results.get('S0', 1365.0)  
    C_true = results['C_true']
    C_obs = results['C_obs']
    spd_true = results['spd_true']
    mean_est_prices = results['mean_est_prices']
    mean_est_spd = results['mean_est_spd']
    lower_ci_prices = results['lower_ci_prices']
    upper_ci_prices = results['upper_ci_prices']
    lower_ci_spd = results['lower_ci_spd']
    upper_ci_spd = results['upper_ci_spd']
    price_samples = results['price_samples']
    mse_prices = results['mse_prices']
    bs_prices = results.get('bs_prices', None)
    bs_spd = results.get('bs_spd', None)
    r = results.get('r', 0.045)
    q = results.get('q', 0.025)
    T = results.get('T', 0.082)
    # Plot 1: Option Price Function
    axes[0, 0].scatter(K, C_obs, color='black', label='True Function')
    axes[0, 0].plot(K, mean_est_prices, color='red', linewidth=2, label='Average Estimate')
    axes[0, 0].fill_between(K, lower_ci_prices, upper_ci_prices, color='red', alpha=0.2, label='95% Confidence Interval')
    if bs_prices is not None:
        axes[0, 0].plot(K, bs_prices, color='green', linestyle='--', label='Black-Scholes Fit')
    axes[0, 0].set_xlabel('Strike Price (K)')
    axes[0, 0].set_ylabel('Option Price (C)')
    axes[0, 0].set_title('Figure 1: Option Price Function Fit')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # Plot 2: State Price Density
    log_returns = np.log(ST_grid / S0)
    
    jacobian = ST_grid
    spd_true_transformed = spd_true * jacobian
    mean_est_spd_transformed = mean_est_spd * jacobian
    lower_ci_spd_transformed = lower_ci_spd * jacobian
    upper_ci_spd_transformed = upper_ci_spd * jacobian
    bs_spd_transformed = bs_spd * jacobian
    mask = (log_returns > -0.8) & (log_returns < 0.8)
    
    axes[0, 1].plot(log_returns[mask], spd_true_transformed[mask], color='black', linestyle='--', label='True Function')
    axes[0, 1].plot(log_returns[mask], mean_est_spd_transformed[mask], color='red', linewidth=2, label='Average Estimate')
    axes[0, 1].fill_between(log_returns[mask], lower_ci_spd_transformed[mask], upper_ci_spd_transformed[mask], 
                    color='red', alpha=0.2, label='95% Confidence Interval')
    if bs_spd_transformed is not None:
        axes[0, 1].plot(log_returns[mask], bs_spd_transformed[mask], 
                      color='green', linestyle='--', linewidth=1.5, 
                      label='Black-Scholes SPD')
    axes[0, 1].set_xlabel('log-return')
    axes[0, 1].set_ylabel('State Price Density (SPD)')
    axes[0, 1].set_title('Figure 2: State Price Density Fit')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    
    errors = results['price_samples'] - C_true.reshape(1, -1)
    all_errors = errors.flatten()
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    
    axes[1, 0].hist(all_errors, bins=40, alpha=0.75, color='lightblue')
    density = sns.kdeplot(all_errors, ax=axes[1, 0], color='blue')
    axes[1, 0].axvline(mean_error, color='black', linestyle='--', 
                    label=f'Mean Error: {mean_error:.4f}')
    
    axes[1, 0].text(0.05, 0.95, f'Mean Error: {mean_error:.4f}\nStd Dev: {std_error:.4f}',
                transform=axes[1, 0].transAxes, verticalalignment='top')
    
    axes[1, 0].set_xlabel('Estimation Error (Estimated - True)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Figure 3: Price Estimation Error Distribution')
    axes[1, 0].grid(True)
    
    # Plot 4: MSE vs. Strike Price
    axes[1, 1].semilogy(K, mse_prices, color='purple', marker='o', 
                    linestyle='-', label='Mean Squared Error (MSE) of Price')
    axes[1, 1].set_xlabel('Strike Price (K)')
    axes[1, 1].set_ylabel('Mean Squared Error (MSE)')
    axes[1, 1].set_title('Figure 4: MSE vs. Strike Price')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figure_simulation_study_results.png', dpi=300, bbox_inches='tight')
    print(f"Figure saved as 'figure_simulation_study_results.png'")
    plt.show()

def plot_empirical_study_results(all_days_results):
    
    print("Generating plots for multiple days empirical study...")
    if not all_days_results or len(all_days_results) == 0:
        print("Error: No valid multi-day data")
        return
    
    num_days = len(all_days_results)
    print(f"Processing data for {num_days} days...")
    save_dir = "/Users/zhuzhuqiu/Desktop/study/复现/empirical_study_results"
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(20, 12))
    rows = min(2, (num_days + 4) // 5)
    cols = min(5, num_days)
    cmap = plt.cm.get_cmap('viridis', num_days)
    
    #Figure 5: 期权价格拟合对比
    for i, results in enumerate(all_days_results):
        K= results['K_fit']
        C_market = results['C_fit']
        C_fitted = results['C_model']
        bs_prices = results.get('bs_prices')
        trading_date = results.get('trading_date', f"Day {i+1}")
        if hasattr(trading_date, 'strftime'):
            date_str = trading_date.strftime('%Y-%m-%d')
        else:
            date_str = str(trading_date)
        S0 = results['S0']
        plt.subplot(rows, cols, i + 1)
        plt.scatter(K, C_market, color='black', s=10, alpha=0.7, label='Market Prices')
        plt.plot(K, C_fitted, color='red', linewidth=1.5, label='Mixture Model Fit')
        if bs_prices is not None:
            plt.plot(K, bs_prices, color='green', linestyle='--', linewidth=1, label='BS Model Fit')
       
        plt.title(f"{date_str} (S0=${S0:.1f})", fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.xlabel('')  
        plt.ylabel('')  
        if i == 0:
            plt.legend(fontsize=9, loc='upper right')
        
        plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.08, 1, 1.0])
    plt.figtext(0.5, 0.02, 'Figure 5: Estimated call option prices vs Market Prices', 
                ha='center', fontsize=24, weight='bold')
    
    save_path = os.path.join(save_dir, "figure5_empirical_study_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 5 saved to: {save_path}")
    plt.show()
    plt.figure(figsize=(20, 12))
  
    #Figure 6: 状态价格密度估计
    for i, results in enumerate(all_days_results):
        trading_date = results.get('trading_date', f"Day {i+1}")
        if hasattr(trading_date, 'strftime'):
            date_str = trading_date.strftime('%Y-%m-%d')
        else:
            date_str = str(trading_date)
        
        S0 = results['S0']
        T = results['T'] 
        r = results['r']
        epsilon = 1e-9
        ST_grid_extended = np.linspace(S0 * 0.5, S0 * 1.8, 500) 
        x_transformed = np.log(ST_grid_extended / S0 + epsilon)-r*T
        
        from spd_model import spd, lognormal_pdf
        mus = results['mus']
        sigma = results['sigma'] 
        pi = results['pi']
        spd_est_extended = spd(ST_grid_extended, pi, mus, sigma)
    
        T, r, q = results['T'], results['r'], results['q']
        bs_vol = results['bs_vol']
        mu_bs = results['mu_bs']  # 使用结果中的BS波动率
        S0 = results['S0']
        bs_sigma_T = bs_vol 
        bs_spd_extended = lognormal_pdf(ST_grid_extended, mu_bs, bs_sigma_T)
       
        plt.subplot(rows, cols, i + 1)
        plt.plot(x_transformed, spd_est_extended, color='red', linewidth=1.5, label='Model SPD')
        if bs_spd_extended is not None:
            plt.plot(x_transformed, bs_spd_extended, color='green', linestyle='--', linewidth=1, label='BS SPD')
        plt.title(f"{date_str} (S0=${S0:.1f})", fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.xlabel('') 
        plt.ylabel('') 
        
        if i == 0:
            plt.legend(fontsize=9, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlim(-0.8, 0.6) 
    
    plt.tight_layout(rect=[0, 0.08, 1, 1.0])
    plt.figtext(0.5, 0.02, 'Figure 6: Estimated state price density (SPD) vs the excess log return', 
                ha='center', fontsize=24, weight='bold')
    
    save_path = os.path.join(save_dir, "figure6_empirical_study_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 6 saved to: {save_path}")
    plt.show()
        
  
