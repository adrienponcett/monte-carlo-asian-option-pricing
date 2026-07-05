import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def simulate_paths(S0, r, sigma, T, N, m, seed=69):
    """Simulate N GBM price paths over m time steps. Returns array of shape (N, m+1)."""
    rng = np.random.default_rng(seed)
    dt = T / m
    mat = rng.standard_normal((N, m))
    inc = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * mat
    inc = np.cumsum(inc, axis=1)                # cumulate log-returns along time
    paths = np.exp(np.log(S0) + inc)            # back to price space
    S0_col = np.full((N, 1), S0)                # every path starts at S0
    paths = np.hstack([S0_col, paths])          # shape (N, m+1)
    return paths


def price_asian_call(S0, r, sigma, T, K, N, m, seed=69):
    """Price an arithmetic Asian call by Monte Carlo.

    Returns the discounted price estimate and its standard error.
    The average is taken over the fixing dates t_1..t_m only; the initial
    price S0 is known and non-random, so it is excluded from the average.
    """
    paths = simulate_paths(S0, r, sigma, T, N, m, seed=seed)
    means = np.mean(paths[:, 1:], axis=1)       # exclude S0 (not a fixing date)
    payoffs = np.exp(-r * T) * np.maximum(means - K, 0)
    price = payoffs.mean()
    se = payoffs.std(ddof=1) / np.sqrt(N)
    return price, se


def Black_Scholes_price_call(S0, K, r, sigma, T):
    """Closed-form Black-Scholes price of a European call."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


if __name__ == "__main__":
    # ---- Parameters ----
    S0, r, sigma, T, K = 100, 0.05, 0.2, 1, 100
    m = 252                       # 252 = daily fixing over one year (market standard)

    # ---- Numerical validation ----
    # Asian call price (Monte Carlo)
    price_asian, se = price_asian_call(S0, r, sigma, T, K, N=100_000, m=m)
    print(f"Asian call   : {price_asian:.4f}  (SE {se:.4f})")

    # European call price (closed-form Black-Scholes, used as the European benchmark)
    price_euro = Black_Scholes_price_call(S0, K, r, sigma, T)
    print(f"European call: {price_euro:.4f}")

    # Consistency check: the Asian option must be cheaper
    # (averaging lowers effective volatility, and the call has positive vega)
    print(f"Asian < European ? {price_asian < price_euro}")

    # ---- Convergence study ----
    N_values = np.logspace(2, 6, 15).astype(int)

    # ==== Method 1: empirical standard error (self-contained, smooth) ====
    # We do NOT measure a real error here: we compute an *estimator* of the
    # standard error (sigma_payoff / sqrt(N)), which predicts the estimator's
    # own dispersion. This quantity is smooth by construction.
    ses = []
    for N in N_values:
        _, se = price_asian_call(S0, r, sigma, T, K, N, m, seed=69)
        ses.append(se)
    ses = np.array(ses)
    slope1 = np.polyfit(np.log(N_values), np.log(ses), 1)[0]

    # ==== Method 2: measured error vs a high-resolution reference (noisier) ====
    # Reference price (pseudo-truth). N is capped at 2e6 rather than higher:
    # an (N x 252) float64 path matrix at N=1e7 would need ~19 GB of RAM.
    # 2e6 already yields a reference far more precise than the tested estimates,
    # so it is a sufficient pseudo-truth given the memory budget.
    price_ref, se_ref = price_asian_call(S0, r, sigma, T, K, N=2_000_000, m=m, seed=0)

    # Here we DO measure a real error: |price_N - reference|, one realization
    # per N, so this curve fluctuates (single-draw noise) around the -0.5 slope.
    # Distinct seed per estimate, all different from the reference seed (0).
    errors, prices, price_ses = [], [], []
    for i, N in enumerate(N_values):
        price_N, se_N = price_asian_call(S0, r, sigma, T, K, N, m, seed=i + 1)
        errors.append(abs(price_N - price_ref))
        prices.append(price_N)        # for the linear price panel below
        price_ses.append(se_N)        # for its confidence bands
    errors = np.array(errors)
    prices = np.array(prices)
    price_ses = np.array(price_ses)
    slope2 = np.polyfit(np.log(N_values), np.log(errors), 1)[0]

    # ==== Three panels in a single figure ====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Method 1 (predicted standard error, log-log) ---
    axes[0].loglog(N_values, ses, 'o-', label=f'Empirical SE (slope {slope1:.3f})')
    axes[0].loglog(N_values, ses[-1] * np.sqrt(N_values[-1] / N_values),
                   '--', label=r'Theoretical $1/\sqrt{N}$ slope')
    axes[0].set_xlabel('Number of paths N')
    axes[0].set_ylabel('Standard error')
    axes[0].set_title('Method 1: predicted standard error')
    axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)

    # --- Panel 2: Method 2 (measured error vs reference, log-log) ---
    axes[1].loglog(N_values, errors, 'o-', color='tab:green',
                   label=f'Absolute error (slope {slope2:.3f})')
    axes[1].loglog(N_values, errors[-1] * np.sqrt(N_values[-1] / N_values),
                   '--', color='tab:red', label=r'Theoretical $1/\sqrt{N}$ slope')
    axes[1].set_xlabel('Number of paths N')
    axes[1].set_ylabel('Absolute error vs reference')
    axes[1].set_title('Method 2: measured error vs reference')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)

    # --- Panel 3: price convergence with shrinking 95% CI (linear y-axis) ---
    # Linear scale here (unlike the log-log panels): confidence bands are
    # symmetric around the price and read naturally on a linear axis.
    axes[2].errorbar(N_values, prices, yerr=1.96 * price_ses,
                     fmt='o-', capsize=3, label='MC price ± 95% CI')
    axes[2].axhline(price_ref, color='tab:red', linestyle='--',
                    label=f'Reference price ({price_ref:.3f})')
    axes[2].set_xscale('log')          # log x to span N, linear y for the price
    axes[2].set_xlabel('Number of paths N')
    axes[2].set_ylabel('Asian call price')
    axes[2].set_title('Method 3: price convergence with 95% CI')
    axes[2].legend()
    axes[2].grid(True, which='both', alpha=0.3)

    fig.suptitle('Monte Carlo convergence of the Asian call price', fontsize=14)
    fig.tight_layout()
    fig.savefig('convergence_three_panels.png', dpi=150, bbox_inches='tight')
    plt.show()