"""Exp2: synthetic parameter/covariance/downstream recovery."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outputs" / "exp2"


def compute_recovery(task: tuple[int, bool, str]) -> dict[str, object]:
    seed, make_curves, m4_backend = task
    p_true = common.get_true_params()
    comps = common.get_fragility_components()
    loss_p = common.get_loss_params()
    x_train, z_train = common.simulate_dataset(1200, p_true, seed=seed)
    x_test, z_test = common.simulate_dataset(400, p_true, seed=seed + 10000)
    p_hat, fit_info = common.fit_M4(x_train, z_train, return_info=True, backend=m4_backend)

    x_grid = np.linspace(-2.0, 1.0, 120)
    mu_true = common.mu_fun(x_grid, p_true)
    mu_hat = common.mu_fun(x_grid, p_hat)
    s_true = common.cov_fun(x_grid, p_true)
    s_hat = common.cov_fun(x_grid, p_hat)

    mean_true, var_true = common.loss_moments_closed_form(x_grid, p_true, comps, loss_p)
    mean_hat, var_hat = common.loss_moments_closed_form(x_grid, p_hat, comps, loss_p)
    frag_true = common.fragility_exceedance(x_grid, p_true, comps[0], 0)
    frag_hat = common.fragility_exceedance(x_grid, p_hat, comps[0], 0)

    metrics: dict[str, object] = {
        "seed": seed,
        "mu_rmse_1": common.rmse(mu_true[:, 0], mu_hat[:, 0]),
        "mu_rmse_2": common.rmse(mu_true[:, 1], mu_hat[:, 1]),
        "mu_rmse_3": common.rmse(mu_true[:, 2], mu_hat[:, 2]),
        "avg_cov_rel_frob_error": float(np.mean([common.rel_frob(s_true[i], s_hat[i]) for i in range(len(x_grid))])),
        "fragility_rmse": common.rmse(frag_true, frag_hat),
        "mean_loss_avg_rel_error": float(np.mean(np.abs(mean_true - mean_hat) / np.maximum(np.abs(mean_true), 1.0e-8))),
        "var_loss_avg_rel_error": float(np.mean(np.abs(var_true - var_hat) / np.maximum(np.abs(var_true), 1.0e-8))),
        "test_nll_joint_3d": common.test_nll_M4(x_test, z_test, p_hat),
        "m4_success": bool(fit_info["success"]),
        "m4_status": int(fit_info["status"]),
        "m4_message": fit_info["message"],
        "m4_fun": float(fit_info["fun"]),
        "m4_nit": int(fit_info["nit"]),
        "m4_nfev": int(fit_info["nfev"]),
        "m4_njev": int(fit_info["njev"]),
        "m4_optimizer": fit_info["optimizer"],
    }

    if make_curves:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        curve = pd.DataFrame(
            {
                "x": x_grid,
                "frag_true": frag_true,
                "frag_hat": frag_hat,
                "mean_loss_true": mean_true,
                "mean_loss_hat": mean_hat,
                "var_loss_true": var_true,
                "var_loss_hat": var_hat,
            }
        )
        for j in range(common.M):
            curve[f"mu_true_{j+1}"] = mu_true[:, j]
            curve[f"mu_hat_{j+1}"] = mu_hat[:, j]
            curve[f"var_true_{j+1}"] = np.diagonal(s_true, axis1=1, axis2=2)[:, j]
            curve[f"var_hat_{j+1}"] = np.diagonal(s_hat, axis1=1, axis2=2)[:, j]
        curve.to_csv(OUTDIR / "exp2_reference_curves.csv", index=False)

        names = ["ln(IDR)", "ln(PFA)", "ln(RD)"]
        for j, name in enumerate(names):
            plt.figure(figsize=(6.2, 4.2))
            plt.plot(x_grid, mu_true[:, j], label="True")
            plt.plot(x_grid, mu_hat[:, j], "--", label="Estimated")
            plt.xlabel("x = ln(IM)")
            plt.ylabel(f"{name} mean")
            plt.legend()
            plt.title(f"Exp2 mean recovery: {name}")
            plt.tight_layout()
            plt.savefig(OUTDIR / f"exp2_mu_recovery_{j+1}.png", dpi=220, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(6.2, 4.2))
            plt.plot(x_grid, np.diagonal(s_true, axis1=1, axis2=2)[:, j], label="True")
            plt.plot(x_grid, np.diagonal(s_hat, axis1=1, axis2=2)[:, j], "--", label="Estimated")
            plt.xlabel("x = ln(IM)")
            plt.ylabel(f"{name} variance")
            plt.legend()
            plt.title(f"Exp2 variance recovery: {name}")
            plt.tight_layout()
            plt.savefig(OUTDIR / f"exp2_var_recovery_{j+1}.png", dpi=220, bbox_inches="tight")
            plt.close()

        plt.figure(figsize=(6.2, 4.2))
        plt.plot(x_grid, frag_true, label="True")
        plt.plot(x_grid, frag_hat, "--", label="Estimated")
        plt.xlabel("x = ln(IM)")
        plt.ylabel("Fragility")
        plt.legend()
        plt.title("Exp2 fragility recovery")
        plt.tight_layout()
        plt.savefig(OUTDIR / "exp2_fragility_recovery.png", dpi=220, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6.2, 4.2))
        plt.plot(x_grid, mean_true, label="True")
        plt.plot(x_grid, mean_hat, "--", label="Estimated")
        plt.xlabel("x = ln(IM)")
        plt.ylabel("E[L|x]")
        plt.legend()
        plt.title("Exp2 mean loss recovery")
        plt.tight_layout()
        plt.savefig(OUTDIR / "exp2_mean_loss_recovery.png", dpi=220, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6.2, 4.2))
        plt.plot(x_grid, var_true, label="True")
        plt.plot(x_grid, var_hat, "--", label="Estimated")
        plt.xlabel("x = ln(IM)")
        plt.ylabel("Var[L|x]")
        plt.legend()
        plt.title("Exp2 loss variance recovery")
        plt.tight_layout()
        plt.savefig(OUTDIR / "exp2_var_loss_recovery.png", dpi=220, bbox_inches="tight")
        plt.close()

    return metrics


def run(repeats: int, workers: int, m4_backend: str) -> dict[str, object]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = [202600 + i for i in range(repeats)]
    reference = compute_recovery((seeds[0], True, m4_backend))
    remaining = seeds[1:]
    rows = [reference]
    if remaining:
        with ProcessPoolExecutor(max_workers=min(workers, len(remaining))) as executor:
            rows.extend(executor.map(compute_recovery, [(seed, False, m4_backend) for seed in remaining]))

    table = pd.DataFrame(rows).sort_values("seed")
    table.to_csv(OUTDIR / "exp2_recovery_repeats.csv", index=False)
    metric_cols = [
        "mu_rmse_1",
        "mu_rmse_2",
        "mu_rmse_3",
        "avg_cov_rel_frob_error",
        "fragility_rmse",
        "mean_loss_avg_rel_error",
        "var_loss_avg_rel_error",
        "test_nll_joint_3d",
        "m4_fun",
        "m4_nit",
        "m4_nfev",
        "m4_njev",
    ]
    numeric = table[metric_cols]
    summary = {
        "repeats": repeats,
        "workers": workers,
        "m4_backend_requested": m4_backend,
        "m4_optimizer_observed": sorted(table["m4_optimizer"].unique().tolist()),
        "m4_success_rate": float(table["m4_success"].mean()),
        "metrics_median": numeric.median().to_dict(),
        "metrics_p90": numeric.quantile(0.9).to_dict(),
        "reference_seed": seeds[0],
    }
    (OUTDIR / "exp2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--m4-backend", choices=["scipy", "torch", "auto"], default="auto")
    args = parser.parse_args()
    summary = run(repeats=args.repeats, workers=args.workers, m4_backend=args.m4_backend)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
