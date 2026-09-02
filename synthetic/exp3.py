"""Exp3: correlation misspecification mechanism and bias."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outputs" / "exp3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    common.OUTDIR = OUTDIR
    summary = common.experiment_3_correlation_bias()
    summary["experiment_role"] = "mechanism/stress-test, not evidence of typical field magnitude"
    (OUTDIR / "exp3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
