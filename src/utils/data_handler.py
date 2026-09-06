
"""
Created on Wed Jul  2 11:46:25 2025

@author: zhuzhuqiu
"""

import numpy as np
from datetime import datetime, timedelta
import warnings
import yfinance as yf
import pandas as pd

warnings.filterwarnings('ignore')


def get_simulation_setup():
    print("\n" + "#"*25 + " Getting Simulation SETUP " + "#"*25)
    
    setup = {
        'S0': 1365.0,
        'r': 0.045,
        'q': 0.025,
        'T': 30 / 365.0,
        'n_options': 25,
        'K_min': 1000,
        'K_max': 1700,
    }
    
    setup['K'] = np.linspace(setup['K_min'], setup['K_max'], setup['n_options'])
    setup['expiry'] = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    print(f"Setup created: S0=${setup['S0']:.2f}, T={setup['T']:.3f} years, {len(setup['K'])} options.")
    
    return setup

def get_real_data_for_date(target_date, symbol="SPY"):
    if isinstance(target_date, datetime):
        target_date = target_date.date()
    try:
        ticker = yf.Ticker(symbol)
        start_date = target_date - timedelta(days=7)
        end_date = target_date + timedelta(days=3)
        hist_data = ticker.history(start=start_date, end=end_date)
        if hist_data.empty:
            print(f"    Unable to get stock price data for {target_date}")
            return None
        
        available_dates = hist_data.index.date
        closest_date_idx = min(range(len(available_dates)), 
                              key=lambda i: abs((available_dates[i] - target_date).days))
        closest_date = hist_data.index[closest_date_idx]
        S0 = hist_data.loc[closest_date, 'Close']
        
        expiry_dates = ticker.options
        if not expiry_dates:
            print(f"    Unable to get option expiry dates")
            return None
            
        expiry_str = None
        for exp in expiry_dates:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            days_diff = (exp_date - target_date).days
            if 25 <= days_diff <= 65:
                expiry_str = exp
                break
        
        if not expiry_str:
            expiry_str = expiry_dates[0] 
        
        expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
        T = (expiry_date - target_date).days / 365.0
        opt_chain = ticker.option_chain(expiry_str)
        calls = opt_chain.calls

        def get_option_price(row):
            if row['bid'] > 0 and row['ask'] > 0:
                return (row['bid'] + row['ask']) / 2
            elif row['lastPrice'] > 0:
                return row['lastPrice']
            else:
                return np.nan
        calls['marketPrice'] = calls.apply(get_option_price, axis=1)
        valid_calls = calls[
            (calls['marketPrice'].notna()) & 
            (calls['marketPrice'] > 0.01) & 
            (calls['volume'].fillna(0) > 1) 
        ].copy()

        print(f"    DEBUG: After flexible filter: {len(valid_calls)} options")

        if len(valid_calls) < 5:
            print(f"    Insufficient option data after filtering: {len(valid_calls)} options")
            return None
     
        valid_calls = valid_calls.sort_values('strike')
        strike_min, strike_max = S0 * 0.9, S0 * 1.1 
        selected_calls = valid_calls[(valid_calls['strike'] >= strike_min) & 
                                   (valid_calls['strike'] <= strike_max)]
     
        if len(selected_calls) < 5:
            print(f"    Not enough options in the desired strike range.")
            valid_calls['distance_to_S0'] = abs(valid_calls['strike'] - S0)
            selected_calls = valid_calls.sort_values('distance_to_S0').head(15).sort_values('strike')
        K_fit = selected_calls['strike'].values
        C_fit = selected_calls['marketPrice'].values
        
        return {
            'K_fit': K_fit,
            'C_fit': C_fit,
            'S0': S0,
            'T': T,
            'r': 0.0525,  
            'q': 0.018,   
            'trading_date': target_date,
            'symbol': symbol
        }  
    except Exception as e:
        print(f"    Failed to get data for {target_date}: {e}")
        return None



def get_market_data(symbol="SPY", num_days=10):
    """动态获取过去N个交易日的数据"""
    hist = yf.Ticker(symbol).history(period="20d")
    if hist.empty:
        print("Error: Could not fetch historical data to determine recent business days.")
        return []
    
    business_days = [d.date() for d in hist.index.to_pydatetime()][-num_days:]
    
    print(f"Target dates: {business_days[0]} to {business_days[-1]}")
    
    all_days_data = []
    for i, date in enumerate(business_days):
        print(f"  Fetching day {i+1}/{len(business_days)}: {date}")
        day_data = get_real_data_for_date(date, symbol)
        if day_data:
            all_days_data.append(day_data)
            print(f"    Success: S0=${day_data['S0']:.2f}, {len(day_data['K_fit'])} options")
        else:
            print(f"    Failed")
    
    print(f"\nCompleted! Successfully fetched {len(all_days_data)}/{len(business_days)} days of data")
    return all_days_data
def get_real_market_data(symbol="SPY", num_days=10):
    print("\n" + "="*20 + " Starting Real Market Data Fetch " + "="*20)
    
    all_days_raw_data = get_market_data(symbol=symbol, num_days=num_days)
    
    if not all_days_raw_data:
        print(" Unable to fetch market data")
        return []
    
    print("\nFitting models to daily data...")
    
    from spd_model import optimize_pi_cvxpy, spd, call_price_from_spd, bs_price, lognormal_pdf, bs_objective
    from scipy.optimize import minimize_scalar
    import numpy as np
    
    all_days_results = []
    for i, day_data in enumerate(all_days_raw_data):
        date_str = day_data['trading_date'].strftime('%Y-%m-%d')
        print(f"  Processing day {i+1}/{len(all_days_raw_data)}: {date_str}")
        
        try:
            K_fit, C_fit = day_data['K_fit'], day_data['C_fit']
            
            if len(K_fit) == 0 or len(C_fit) == 0:
                print(f"     No valid options for this day, skipping.")
                continue
                
            S0, T, r, q = day_data['S0'], day_data['T'], day_data['r'], day_data['q']
            res_vol = minimize_scalar(bs_objective, bounds=(0.01, 1.0), method='bounded', 
                                    args=(S0, K_fit, T, r, q, C_fit))
            bs_vol = res_vol.x 
            
            sigma_fixed = bs_vol * 0.75  
            F_tT = S0 * np.exp((r - q) * T)
            ST_grid = np.linspace(K_fit.min() * 0.8, K_fit.max() * 1.2, 200)
            log_F = np.log(F_tT)
            mus_total_std = sigma_fixed * np.sqrt(T)  
            mus = np.linspace(log_F - 2 * mus_total_std, log_F + 2 * mus_total_std, 12)
            
            pi = optimize_pi_cvxpy(mus, K_fit, C_fit, ST_grid, r, T, mus_total_std, F_tT, tolerance=0.05)
            spd_est = spd(ST_grid, pi, mus, mus_total_std)
            
            C_model = np.array([call_price_from_spd(k, ST_grid, spd_est, r, T) for k in K_fit])
            bs_prices = np.array([bs_price(S0, k, T, r, bs_vol*np.sqrt(T), q) for k in K_fit])
            mu_bs = np.log(S0) + (r - q - 0.5 * bs_vol**2) * T
        
            result = {
                **day_data,
                'C_model': C_model,
                'ST_grid': ST_grid,
                'spd_est': spd_est,
                'bs_prices': bs_prices,
                'mus': mus,
                'sigma': sigma_fixed,
                'pi': pi,
                'bs_vol': bs_vol,  
                'mu_bs': mu_bs,
            }
            
            all_days_results.append(result)
            print(f"     Processing successful: volatility={bs_vol:.3f}")
            
        except Exception as e:
            print(f"     Processing failed: {e}")
            continue
    
    print(f"\nProcessing completed! Successfully processed {len(all_days_results)} days of real data")
    return all_days_results
