import numpy as np
from data_handler import get_real_market_data
from plotting import plot_empirical_study_results

def main():
    print("="*60)
    print("Empirical Study: Real Market Data Analysis")
    print("="*60)
    print("Fetching and analyzing 5 days of real market data...")
    print("="*60)
    
    processed_data_for_plotting = get_real_market_data(symbol="SPY",num_days=10)
    if processed_data_for_plotting:
        print(f"\Data processing completed, {len(processed_data_for_plotting)} days of data")
        print("Generating charts...")
        plot_empirical_study_results(processed_data_for_plotting)
        print("Empirical study completed!")
        print("Charts saved to empirical_study_results/ folder")
    else:
        print("No data available for plotting.")

if __name__ == "__main__":
    main()


