#!/usr/bin/env python3
# mc_var_engine.py
# Note: quick-and-dirty research driver; not production infra.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, genpareto

import torch
import matplotlib.pyplot as plt


# ---------- small helpers (keep them boring on purpose) ----------

def read_returns(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column.")
    # Accept either precomputed daily_return or compute it if missing
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "daily_return" not in df.columns:
        if "price" not in df.columns:
            raise ValueError("Need 'daily_return' or 'price' to compute returns.")
        prev = df["price"].shift(1)
        df["daily_return"] = (df["price"] - prev) / prev

    # Trim to the window we care about
    mask = (df["date"] >= "2018-01-01") & (df["date"] <= "2025-12-31")
    df = df.loc[mask].dropna(subset=["daily_return"]).reset_index(drop=True)
    return df


def fit_distribution(returns: pd.Series) -> dict:
    # Keep it empirical; we only use μ/σ for baseline normal MC
    return {
        "mean": float(returns.mean()),
        "stdev": float(returns.std()),
        "skew": float(skew(returns)),
        "kurtosis": float(kurtosis(returns)),
    }


def pick_device() -> str:
    # prefer GPU when present; print once so we know where it ran
    return "cuda" if torch.cuda.is_available() else "cpu"


def mc_normal_var(mu, sigma, n_paths=100_000, horizon=1, device="cpu", seed=0):
    # simple normal MC over horizon; sum returns
    # note: you can comment out seed for more randomness
    torch.manual_seed(seed)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sd_t = torch.tensor(sigma, dtype=torch.float32, device=device)
    R = torch.normal(mu_t, sd_t, size=(n_paths, horizon), generator=None, device=device)
    R_sum = R.sum(dim=1)
    q05 = torch.quantile(R_sum, 0.05).item()
    q01 = torch.quantile(R_sum, 0.01).item()
    q005 = torch.quantile(R_sum, 0.005).item()
    return {"95": q05, "99": q01, "99.5": q005}, R_sum  # return MC for reuse if needed


def fit_evt_left_tail(returns: pd.Series, tail_pct=5.0):
    # threshold at p% (left tail). We fit GPD to exceedances beyond threshold.
    thr = np.percentile(returns, tail_pct)
    tail = returns[returns < thr]
    # Fit on positive "excess losses": -(x - thr), so support is >=0
    shape, loc, scale = genpareto.fit(-(tail - thr))
    return float(thr), (shape, loc, scale)


def mc_evt_adjusted(mu, sigma, evt_params, threshold, n_paths=100_000, device="cpu", seed=1):
    # Start with normal draws, then replace worst ~tail% with GPD losses
    torch.manual_seed(seed)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sd_t = torch.tensor(sigma, dtype=torch.float32, device=device)
    base = torch.normal(mu_t, sd_t, size=(n_paths,), device=device)

    tail_frac = 0.05  # matches threshold definition above
    n_tail = int(tail_frac * n_paths)
    shape, loc, scale = evt_params

    # Draw GPD exceedances; convert back to returns below threshold
    # g ~ GPD >= 0 on the excess; return = threshold - g (negative)
    g = genpareto.rvs(c=shape, loc=loc, scale=scale, size=n_tail, random_state=seed)
    evt_draws = threshold - g  # negative tail heavier than normal

    # Replace worst base samples with EVT draws
    base_sorted, idx = torch.sort(base)  # ascending
    base_sorted[:n_tail] = torch.tensor(evt_draws, dtype=torch.float32, device=device)

    v95 = torch.quantile(base_sorted, 0.05).item()
    v99 = torch.quantile(base_sorted, 0.01).item()
    v995 = torch.quantile(base_sorted, 0.005).item()
    return {"95": v95, "99": v99, "99.5": v995}


def backtest_period(df: pd.DataFrame, start: str, end: str, var_levels: dict, notional=10_000_000):
    window = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    pnl = window["daily_return"].to_numpy() * notional
    results = {}
    for lev_txt, q in var_levels.items():
        # var q is in returns; multiply by notional to compare to losses
        thr_loss = q * notional
        exc = int((pnl < thr_loss).sum())
        level = float(lev_txt) / 100.0
        exp = round(len(pnl) * (1 - level))
        err = abs(exc - exp) / len(pnl) if len(pnl) else np.nan
        results[lev_txt] = {"VaR_ret": q, "VaR_loss": thr_loss, "exceedances": exc, "expected": exp, "error": err}
    return window, results


def plot_covid(window_df: pd.DataFrame, var_evt: dict, out_png: Path | None):
    plt.figure(figsize=(12, 6))
    plt.plot(window_df["date"], window_df["daily_return"], label="FTSE 100 daily return", alpha=0.75)
    # dashed VaR lines
    plt.axhline(var_evt["95"], linestyle="--", label="EVT VaR 95%")
    plt.axhline(var_evt["99"], linestyle="--", label="EVT VaR 99%")
    plt.axhline(var_evt["99.5"], linestyle="--", label="EVT VaR 99.5%")
    plt.title("COVID crash: returns vs EVT VaR (Feb–Jun 2020)")
    plt.xlabel("date")
    plt.ylabel("return")
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=140)
    plt.show()


# ---------- main flow ----------

def main():
    ap = argparse.ArgumentParser(description="Monte Carlo + EVT VaR (FTSE100)")
    ap.add_argument("--csv", type=Path, default=Path(r"C:\Users\Haise\Downloads\FTSE_100_Cleaned.csv"))
    ap.add_argument("--paths", type=int, default=100_000)
    ap.add_argument("--horizon", type=int, default=1, help="days; currently summed iid daily returns")
    ap.add_argument("--notional", type=float, default=10_000_000.0)
    ap.add_argument("--out_summary", type=Path, default=Path("var_summary.csv"))
    ap.add_argument("--out_plot", type=Path, default=Path("covid_evt_var.png"))
    ap.add_argument("--seed", type=int, default=1337)  # tweak per run
    args = ap.parse_args()

    df = read_returns(args.csv)
    returns = df["daily_return"]

    # 1) fit distro
    stats = fit_distribution(returns)
    print("fit:", stats)

    # 2) baseline MC (normal)
    device = pick_device()
    print("device:", device)
    var_mc, _ = mc_normal_var(stats["mean"], stats["stdev"], n_paths=args.paths, horizon=args.horizon, device=device, seed=args.seed)
    print("\nMC (normal) VaR (returns):", var_mc)

    # 3) EVT tail
    thr, evt = fit_evt_left_tail(returns, tail_pct=5.0)
    print(f"EVT fit: threshold={thr:.6f}, shape={evt[0]:.4f}, scale={evt[2]:.4f}")

    var_evt = mc_evt_adjusted(stats["mean"], stats["stdev"], evt, threshold=thr, n_paths=args.paths, device=device, seed=args.seed + 1)
    print("\nEVT-adjusted MC VaR (returns):", var_evt)

    # 4) backtest on COVID (Feb–Jun 2020)
    covid_window, covid_bt = backtest_period(df, "2020-02-01", "2020-06-30", var_evt, notional=args.notional)
    print("\nCOVID backtest (EVT VaR, losses @ £):")
    for k, v in covid_bt.items():
        print(f"{k}%  VaR_loss={v['VaR_loss']:.0f}  exc={v['exceedances']}  exp={v['expected']}  err={v['error']:.3f}")

    # 5) write summary (helps with README/repro)
    out_rows = []
    for label, q in var_mc.items():
        out_rows.append({"model": "MC_normal", "level": label, "VaR_return": q})
    for label, q in var_evt.items():
        out_rows.append({"model": "MC_EVT", "level": label, "VaR_return": q})
    for label, v in covid_bt.items():
        out_rows.append({
            "model": "COVID_backtest_EVT",
            "level": label,
            "VaR_return": v["VaR_ret"],
            "VaR_loss": v["VaR_loss"],
            "exceedances": v["exceedances"],
            "expected": v["expected"],
            "error": v["error"],
        })
    pd.DataFrame(out_rows).to_csv(args.out_summary, index=False)
    print(f"\nWrote summary → {args.out_summary.resolve()}")

    # 6) plot COVID window vs EVT VaR
    # (visuals are persuasive in a README; keep it simple)
    try:
        plot_covid(covid_window, var_evt, args.out_plot)
        print(f"Saved plot → {args.out_plot.resolve()}")
    except Exception as e:
        # if you're on a headless env, just skip plotting
        print(f"plot skipped: {e}")

    # TODO: add 10-day horizon option with square-root-time sanity check vs direct sim
    # TODO: add rolling (expanding) backtest for whole 2018–2025 period


if __name__ == "__main__":
    main()
