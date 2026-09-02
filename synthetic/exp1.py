"""Exp1: closed-form fragility/loss moments versus Monte Carlo."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outputs" / "exp1"


def mc_one_point(args: tuple[int, float, int]) -> dict[str, float]:
    index, x, n_mc = args
    p = common.get_true_params()
    comps = common.get_fragility_components()
    comp = comps[0]  # drift class supplies the Fig. 2(a) fragility panel
    loss_p = common.get_loss_params()
    rg = np.random.default_rng(2026070501 + index)

    mu = common.mu_fun(np.array([x]), p)[0]
    sigma = common.cov_fun(np.array([x]), p)[0]
    z = rg.multivariate_normal(mu, sigma, size=n_mc)

    # Total system loss over all three component classes with mutually
    # exclusive ordered damage states.  Within each class one shared
    # class-level capacity error drives both ordered thresholds; capacity
    # variables of different classes are independent.
    total_loss = np.zeros(n_mc)
    ds1 = ds2 = None
    for c_index, component in enumerate(comps):
        eps = rg.normal(0.0, math.sqrt(component.zeta2), size=n_mc)
        w = z @ component.A + eps
        exceed1 = (w >= component.b[0]).astype(int)
        exceed2 = (w >= component.b[1]).astype(int)
        state = exceed1 + exceed2  # 0, 1, 2 nested by construction
        if c_index == 0:
            ds1, ds2 = exceed1, exceed2
        q = loss_p.q[c_index]
        mk = loss_p.m_state[c_index]
        sk = np.sqrt(loss_p.s2_state[c_index])
        for s in (1, 2):
            n_s = int((state == s).sum())
            if n_s:
                total_loss[state == s] += q * rg.normal(mk[s], sk[s], size=n_s)

    return {
        "x": float(x),
        "frag_mc": float(ds1.mean()),
        "frag_mc_se": float(math.sqrt(ds1.mean() * (1.0 - ds1.mean()) / n_mc)),
        "frag2_mc": float(ds2.mean()),
        "frag2_mc_se": float(math.sqrt(ds2.mean() * (1.0 - ds2.mean()) / n_mc)),
        "mean_mc": float(total_loss.mean()),
        "var_mc": float(total_loss.var(ddof=0)),
    }


def closed_form_values(x_grid: np.ndarray) -> pd.DataFrame:
    p = common.get_true_params()
    comps = common.get_fragility_components()
    comp = comps[0]  # drift class supplies the Fig. 2(a) fragility panel
    loss_p = common.get_loss_params()
    frag = common.fragility_exceedance(x_grid, p, comp, 0)
    frag2 = common.fragility_exceedance(x_grid, p, comp, 1)

    # Full three-class mutually exclusive loss moments, including the exact
    # Cross-component covariance of Eq. (43) via joint state probabilities.
    mean_loss, var_loss = common.loss_moments_closed_form(x_grid, p, comps, loss_p)

    return pd.DataFrame(
        {
            "x": x_grid,
            "frag_cf": frag,
            "frag2_cf": frag2,
            "mean_cf": mean_loss,
            "var_cf": var_loss,
        }
    )


def summarize_consistency(df: pd.DataFrame, n_mc: int, workers: int, check: dict[str, object]) -> dict[str, object]:
    active = df["frag_cf"] > 1.0e-3
    return {
        "n_mc_per_intensity": n_mc,
        "workers": workers,
        "covariance_definition_check": check,
        "max_frag_abs_err": float(df["frag_abs_err"].max()),
        "max_frag2_abs_err": float(df["frag2_abs_err"].max()),
        "max_mean_abs_err": float((df["mean_cf"] - df["mean_mc"]).abs().max()),
        "max_var_abs_err": float((df["var_cf"] - df["var_mc"]).abs().max()),
        "max_mean_rel_err_active_frag_gt_1e3": float(df.loc[active, "mean_rel_err"].max()),
        "max_var_rel_err_active_frag_gt_1e3": float(df.loc[active, "var_rel_err"].max()),
        "mean_frag_abs_err": float(df["frag_abs_err"].mean()),
        "active_point_count_frag_gt_1e3": int(active.sum()),
    }


def evaluate_at_n_mc(x_grid: np.ndarray, n_mc: int, workers: int) -> pd.DataFrame:
    closed = closed_form_values(x_grid)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(mc_one_point, [(i, float(x), n_mc) for i, x in enumerate(x_grid)]))
    mc = pd.DataFrame(rows)
    df = closed.merge(mc, on="x")
    df["frag_abs_err"] = (df["frag_cf"] - df["frag_mc"]).abs()
    df["frag2_abs_err"] = (df["frag2_cf"] - df["frag2_mc"]).abs()
    df["mean_rel_err"] = (df["mean_cf"] - df["mean_mc"]).abs() / np.maximum(df["mean_mc"].abs(), 1.0e-8)
    df["var_rel_err"] = (df["var_cf"] - df["var_mc"]).abs() / np.maximum(df["var_mc"].abs(), 1.0e-8)
    return df


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close()


def run(n_mc: int, workers: int) -> dict[str, object]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    check = common.covariance_definition_check()
    x_grid = np.linspace(0.8, 1.8, 21)
    df = evaluate_at_n_mc(x_grid, n_mc, workers)
    df.to_csv(OUTDIR / "exp1_closed_form_consistency.csv", index=False)

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(df["x"], df["frag_cf"], label="Closed form")
    plt.errorbar(df["x"], df["frag_mc"], yerr=1.96 * df["frag_mc_se"], fmt="o", ms=3, label="Monte Carlo 95% CI")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Fragility")
    plt.legend()
    plt.title("Exp1 fragility consistency")
    savefig("exp1_fragility_consistency")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(df["x"], df["mean_cf"], label="Closed form")
    plt.scatter(df["x"], df["mean_mc"], s=18, label="Monte Carlo")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("E[L|x]")
    plt.legend()
    plt.title("Exp1 mean loss consistency")
    savefig("exp1_mean_loss_consistency")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(df["x"], df["var_cf"], label="Closed form")
    plt.scatter(df["x"], df["var_mc"], s=18, label="Monte Carlo")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Var[L|x]")
    plt.legend()
    plt.title("Exp1 loss variance consistency")
    savefig("exp1_var_loss_consistency")

    summary = summarize_consistency(df, n_mc, workers, check)
    (OUTDIR / "exp1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_convergence(n_values: list[int], workers: int) -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    check = common.covariance_definition_check()
    x_grid = np.linspace(0.8, 1.8, 21)
    rows = []
    for n_mc in n_values:
        df = evaluate_at_n_mc(x_grid, n_mc, workers)
        summary = summarize_consistency(df, n_mc, workers, check)
        rows.append(
            {
                "n_mc_per_intensity": n_mc,
                "max_frag_abs_err": summary["max_frag_abs_err"],
                "mean_frag_abs_err": summary["mean_frag_abs_err"],
                "max_mean_rel_err_active_frag_gt_1e3": summary["max_mean_rel_err_active_frag_gt_1e3"],
                "max_var_rel_err_active_frag_gt_1e3": summary["max_var_rel_err_active_frag_gt_1e3"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUTDIR / "exp1_mc_convergence.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mc", type=int, default=120000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--convergence-n-mc", type=str, default="")
    args = parser.parse_args()
    summary = run(n_mc=args.n_mc, workers=args.workers)
    if args.convergence_n_mc:
        n_values = [int(value.strip()) for value in args.convergence_n_mc.split(",") if value.strip()]
        table = run_convergence(n_values=n_values, workers=args.workers)
        summary["mc_convergence_file"] = "exp1_mc_convergence.csv"
        summary["mc_convergence_n_values"] = n_values
        summary["mc_convergence_last_row"] = table.iloc[-1].to_dict()
        (OUTDIR / "exp1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
