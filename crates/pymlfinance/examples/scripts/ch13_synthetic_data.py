"""
Chapter 13: Synthetic Data Generation
======================================
AFML Ch. 13 — Ornstein-Uhlenbeck process for synthetic data.

Demonstrates:
- Simulating OU processes with different parameters
- Estimating OU parameters from data
- Parameter recovery accuracy
"""

import numpy as np
import pymlfinance

np.random.seed(42)

print("=" * 60)
print("Chapter 13: Synthetic Data Generation")
print("=" * 60)

# --- Simulate OU Process ---
print(f"\n--- Ornstein-Uhlenbeck Process ---")
print(f"  dX = theta * (mu - X) * dt + sigma * dW")
print()

# Different parameter sets
params = [
    {"theta": 1.0, "mu": 0.0, "sigma": 0.5, "label": "Fast mean-reversion"},
    {"theta": 0.1, "mu": 0.0, "sigma": 0.5, "label": "Slow mean-reversion"},
    {"theta": 1.0, "mu": 5.0, "sigma": 0.5, "label": "Non-zero mean"},
    {"theta": 1.0, "mu": 0.0, "sigma": 2.0, "label": "High volatility"},
]

for p in params:
    path = pymlfinance.backtesting.simulate_ou(
        theta=p["theta"], mu=p["mu"], sigma=p["sigma"],
        x0=p["mu"], dt=0.01, n_steps=1000, seed=42
    )
    print(f"  {p['label']:30s} theta={p['theta']}, mu={p['mu']}, sigma={p['sigma']}")
    print(f"    Path: mean={np.mean(path):.4f}, std={np.std(path):.4f}, "
          f"min={np.min(path):.4f}, max={np.max(path):.4f}")

# --- Parameter Estimation ---
print(f"\n--- Parameter Estimation (Recovery Test) ---")
true_theta, true_mu, true_sigma = 2.0, 1.0, 0.3
n_steps = 5000
dt = 0.01

path = pymlfinance.backtesting.simulate_ou(
    theta=true_theta, mu=true_mu, sigma=true_sigma,
    x0=true_mu, dt=dt, n_steps=n_steps, seed=42
)

est_theta, est_mu, est_sigma = pymlfinance.backtesting.estimate_ou_params(path, dt)
print(f"  True:      theta={true_theta:.4f}, mu={true_mu:.4f}, sigma={true_sigma:.4f}")
print(f"  Estimated: theta={est_theta:.4f}, mu={est_mu:.4f}, sigma={est_sigma:.4f}")
print(f"  Error:     theta={abs(est_theta-true_theta)/true_theta:.1%}, "
      f"mu={abs(est_mu-true_mu)/max(abs(true_mu), 1e-10):.1%}, "
      f"sigma={abs(est_sigma-true_sigma)/true_sigma:.1%}")

# --- Multiple seeds for estimation variance ---
print(f"\n--- Estimation Variance (10 runs) ---")
thetas, mus, sigmas = [], [], []
for seed in range(10):
    path = pymlfinance.backtesting.simulate_ou(
        theta=true_theta, mu=true_mu, sigma=true_sigma,
        x0=true_mu, dt=dt, n_steps=n_steps, seed=seed
    )
    t, m, s = pymlfinance.backtesting.estimate_ou_params(path, dt)
    thetas.append(t)
    mus.append(m)
    sigmas.append(s)

print(f"  theta: {np.mean(thetas):.4f} +/- {np.std(thetas):.4f} (true: {true_theta})")
print(f"  mu:    {np.mean(mus):.4f} +/- {np.std(mus):.4f} (true: {true_mu})")
print(f"  sigma: {np.mean(sigmas):.4f} +/- {np.std(sigmas):.4f} (true: {true_sigma})")

# --- Effect of sample size on estimation ---
print(f"\n--- Sample Size vs Estimation Quality ---")
for n in [100, 500, 1000, 5000, 10000]:
    path = pymlfinance.backtesting.simulate_ou(
        theta=true_theta, mu=true_mu, sigma=true_sigma,
        x0=true_mu, dt=dt, n_steps=n, seed=42
    )
    t, m, s = pymlfinance.backtesting.estimate_ou_params(path, dt)
    theta_err = abs(t - true_theta) / true_theta
    print(f"  n={n:>5d}: theta_error={theta_err:.1%}, est_theta={t:.4f}")

# --- Exercises ---
print(f"\n--- Exercises ---")
print("1. Vary theta from 0.1 to 10.0 and observe how quickly paths revert")
print("2. Use estimated parameters to simulate new paths and compare distributions")
print("3. Test parameter recovery with different dt values (0.001, 0.01, 0.1)")
