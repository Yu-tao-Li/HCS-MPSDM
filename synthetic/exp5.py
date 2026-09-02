"""Exp5: broad synthetic S1--S4 comparison."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import common


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outputs" / "exp5"
PUBLIC_LABELS = {"M1": "S1", "M2": "S2", "M3": "S3", "M4": "S4"}


def _publicize_tokens(value):
    if isinstance(value, dict):
        return {_publicize_tokens(key): _publicize_tokens(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_publicize_tokens(item) for item in value]
    if isinstance(value, str):
        for old, new in PUBLIC_LABELS.items():
            value = value.replace(old, new)
    return value


def _publicize_curve(path: Path) -> None:
    frame = pd.read_csv(path)
    rename = {}
    for column in frame.columns:
        for old, new in PUBLIC_LABELS.items():
            if column.endswith(f"_{old}"):
                rename[column] = column[: -len(old)] + new
    frame.rename(columns=rename).to_csv(path, index=False)


def fit_and_score(task: tuple[int, int, int, str]) -> dict[str, object]:
    seed, n_train, n_test, m4_backend = task
    p_true = common.get_true_params()
    x_train, z_train = common.simulate_dataset(n_train, p_true, seed=seed)
    x_test, z_test = common.simulate_dataset(n_test, p_true, seed=seed + 10000)

    m1 = common.fit_M1_independent_homoscedastic(x_train, z_train)
    m2 = common.fit_M2_multivar_homoscedastic(x_train, z_train)
    m3 = common.fit_M3_multivar_hetero_constcorr(x_train, z_train)
    m4, m4_info = common.fit_M4(x_train, z_train, return_info=True, backend=m4_backend)

    nlls = {
        "M1": common.test_nll_M1(x_test, z_test, m1),
        "M2": common.test_nll_M2M3(x_test, z_test, m2, "M2"),
        "M3": common.test_nll_M2M3(x_test, z_test, m3, "M3"),
        "M4": common.test_nll_M4(x_test, z_test, m4),
    }
    row: dict[str, object] = {
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "m4_success": bool(m4_info["success"]),
        "m4_status": int(m4_info["status"]),
        "m4_message": m4_info["message"],
        "m4_fun": float(m4_info["fun"]),
        "m4_nit": int(m4_info["nit"]),
        "m4_nfev": int(m4_info["nfev"]),
        "m4_njev": int(m4_info["njev"]),
        "m4_optimizer": m4_info["optimizer"],
    }
    for model, value in nlls.items():
        row[f"nll_{model}"] = value
        row[f"nll_per_dim_{model}"] = value / common.M
    row["ranking_best_to_worst"] = "<".join(sorted(nlls, key=nlls.get))
    return row


def run(repeats: int, workers: int, n_train: int, n_test: int, m4_backend: str) -> dict[str, object]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    common.OUTDIR = OUTDIR

    # One reference run writes the loss curves and single-split summary.
    reference = common.experiment_5_loss_comparison(
        m4_backend=m4_backend,
        n_train=n_train,
        n_test=n_test,
        train_seed=304,
        test_seed=305,
    )
    _publicize_curve(OUTDIR / "exp5_loss_comparison.csv")
    rows = []
    ref_nll = reference["test_nll_per_observation_joint_3d"]
    ref_info = reference["m4_optimizer_info"]
    ref_row: dict[str, object] = {
        "seed": 304,
        "n_train": n_train,
        "n_test": n_test,
        "ranking_best_to_worst": "<".join(reference["test_nll_ranking_best_to_worst"]),
        "m4_success": bool(ref_info["success"]),
        "m4_status": int(ref_info["status"]),
        "m4_message": ref_info["message"],
        "m4_fun": float(ref_info["fun"]),
        "m4_nit": int(ref_info["nit"]),
        "m4_nfev": int(ref_info["nfev"]),
        "m4_njev": int(ref_info["njev"]),
        "m4_optimizer": ref_info["optimizer"],
    }
    for model, value in ref_nll.items():
        ref_row[f"nll_{model}"] = value
        ref_row[f"nll_per_dim_{model}"] = value / common.M
    rows.append(ref_row)

    extra_count = max(repeats - 1, 0)
    seeds = [202650 + i for i in range(extra_count)]
    if seeds:
        with ProcessPoolExecutor(max_workers=min(workers, len(seeds))) as executor:
            rows.extend(executor.map(fit_and_score, [(seed, n_train, n_test, m4_backend) for seed in seeds]))

    table = pd.DataFrame(rows).sort_values("seed")
    public_table = table.copy()
    public_table = public_table.rename(
        columns={
            **{f"nll_{old}": f"nll_{new}" for old, new in PUBLIC_LABELS.items()},
            **{f"nll_per_dim_{old}": f"nll_per_dim_{new}" for old, new in PUBLIC_LABELS.items()},
        }
    )
    public_table["ranking_best_to_worst"] = public_table["ranking_best_to_worst"].replace(PUBLIC_LABELS, regex=True)
    public_table.to_csv(OUTDIR / "exp5_repeated_joint_nll.csv", index=False)
    nll_cols = [f"nll_{model}" for model in ["M1", "M2", "M3", "M4"]]
    per_dim_cols = [f"nll_per_dim_{model}" for model in ["M1", "M2", "M3", "M4"]]
    win_counts = table["ranking_best_to_worst"].str.split("<").str[0].value_counts().to_dict()
    summary = {
        "repeats_total_including_reference": len(rows),
        "workers": workers,
        "n_train": n_train,
        "n_test": n_test,
        "m4_backend_requested": m4_backend,
        "m4_optimizer_observed": sorted(table["m4_optimizer"].unique().tolist()),
        "m4_success_rate": float(table["m4_success"].mean()),
        "reference_single_split": reference,
        "joint_nll_median": table[nll_cols].median().to_dict(),
        "joint_nll_p10": table[nll_cols].quantile(0.1).to_dict(),
        "joint_nll_p90": table[nll_cols].quantile(0.9).to_dict(),
        "per_dimension_nll_median": table[per_dim_cols].median().to_dict(),
        "best_model_counts": win_counts,
        "all_models_evaluated_on_same_3d_response_vector": True,
        "m1_definition": "three independent scalar homoscedastic Gaussian models; joint density is product of three marginals",
        "m3_definition": "linear mean, log-linear marginal variance, constant correlation fitted by joint Gaussian NLL",
        "loss_proxy_note": "mean + 1.645*sqrt(variance) is a normal-moment proxy, not an empirical 95% quantile",
    }
    public_summary = _publicize_tokens(summary)
    public_summary.update(
        {
            "experiment_role": "broad correctly specified-versus-simplified comparison",
            "public_label_definitions": {
                "S1": "three independent scalar homoscedastic Gaussian models",
                "S2": "linear-mean multivariate homoscedastic Gaussian model",
                "S3": "linear-mean multivariate Gaussian model with log-linear marginal variances and constant correlation",
                "S4": "full correctly specified synthetic generating model with bilinear mean, bounded heteroscedastic variances, and intensity-dependent rank-one correlation",
            },
            "layer_isolating_ablation": False,
            "loss_operator": "All S1-S4 curves use the same mutually exclusive ordered damage-state operator, including full cross-class covariance.",
        }
    )
    (OUTDIR / "exp5_summary.json").write_text(json.dumps(public_summary, indent=2), encoding="utf-8")
    return public_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    # Eight splits in total (reference seed 304 plus seven repeated splits),
    # using eight repeated splits.  The seed
    # sequence is fixed: 304, 202650, ..., 202656.
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--m4-backend", choices=["scipy", "torch", "auto"], default="auto")
    args = parser.parse_args()
    summary = run(
        repeats=args.repeats,
        workers=args.workers,
        n_train=args.n_train,
        n_test=args.n_test,
        m4_backend=args.m4_backend,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
