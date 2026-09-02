"""Fit the multivariate demand model to the processed response table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcs_mpsdm import fit, transform_responses  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "msa_responses.csv",
    )
    parser.add_argument("--max-scale-factor", type=float)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="train",
    )
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "fit_parameters.json",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    selected = frame[frame["convergence"].astype(bool) & ~frame["collapse"].astype(bool)].copy()
    if args.split != "all":
        selected = selected[selected["split"] == args.split].copy()
    if args.max_scale_factor is not None:
        selected = selected[selected["scale_factor"] <= args.max_scale_factor].copy()
    if selected.empty:
        raise ValueError("No usable non-collapse observations remain after filtering")

    ln_im = np.log(selected["target_IM_g"].to_numpy(dtype=float))
    log_demands = transform_responses(
        selected["max_IDR"].to_numpy(dtype=float),
        selected["max_PFA_g"].to_numpy(dtype=float),
        selected["RD"].to_numpy(dtype=float),
    )
    parameter_set = (
        "scale_factor_le_3_noncollapse"
        if args.max_scale_factor is not None and args.max_scale_factor <= 3.0
        else "all_noncollapse"
    )
    result = fit(
        ln_im,
        log_demands,
        parameter_set=parameter_set,
        maxiter=args.maxiter,
    )

    payload = {
        "input": args.input.name,
        "observations": int(len(selected)),
        "split": args.split,
        "max_scale_factor": args.max_scale_factor,
        "success": bool(result.success),
        "message": str(result.message),
        "mean_negative_log_likelihood": float(result.fun),
        "theta": np.asarray(result.x, dtype=float).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
