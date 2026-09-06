# Nonparametric Risk-Neutral Density Estimation

A quantitative modelling framework for estimating risk-neutral densities (RNDs) from option prices using nonparametric mixture models and constrained optimisation.

## Overview

Risk-neutral densities provide information about the market-implied distribution of future asset prices. This project implements a nonparametric estimation approach based on lognormal mixture models, following the state price density framework, and evaluates the method through simulation and empirical studies.

## Methodology

The workflow consists of:

1. Generating option prices under simulated market settings
2. Constructing nonparametric lognormal mixture representations
3. Estimating mixture weights through constrained optimisation
4. Enforcing financial consistency through:
   - Probability constraints
   - Martingale constraints
5. Recovering risk-neutral density and state price density
6. Comparing estimated distributions with benchmark models

## Key Features

- Nonparametric mixture-based density estimation
- Constrained optimisation using convex programming
- Simulation-based validation
- Empirical option data analysis
- Visualisation of implied distributions and pricing performance

## Project Structure

```text
Risk-Neutral-Density-Estimation/

├── src/
│   ├── models/
│   │   └── spd_model.py
│   ├── experiments/
│   │   ├── simulation.py
│   │   ├── run_simulation.py
│   │   └── run_empirical.py
│   └── utils/
│       ├── data_handler.py
│       └── plotting.py
│
├── figures/
├── results/
├── references/
├── README.md
└── requirements.txt
```

## Results

The framework is able to recover non-Gaussian features of market-implied distributions, including:

- Volatility smile effects
- Skewness
- Fat-tail behaviour
- Asymmetric risk-neutral distributions

Simulation and empirical experiments demonstrate the effectiveness of the proposed estimation framework.

## Applications

This project connects statistical modelling, optimisation, and quantitative finance, with applications in:

- Option pricing
- Risk management
- Distributional forecasting
- Financial econometrics

## Technologies

Python · NumPy · SciPy · CVXPY · Matplotlib

## References

- Yuan (2009), State Price Density Estimation
- Research on risk-neutral densities and option-implied distributions

