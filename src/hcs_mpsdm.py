"""Core multivariate seismic demand model utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeResult, minimize


M = 3
PARAMETER_COUNT = 27
PARAMETER_FILE = Path(__file__).with_name("parameters.json")


def load_parameters(name: str = "all_noncollapse") -> np.ndarray:
    """Load one stored 27-coordinate parameter vector."""
    payload = json.loads(PARAMETER_FILE.read_text(encoding="utf-8"))
    try:
        theta = np.asarray(payload["fits"][name]["theta"], dtype=float)
    except KeyError as exc:
        available = ", ".join(sorted(payload.get("fits", {})))
        raise KeyError(f"Unknown parameter set {name!r}; available: {available}") from exc
    if theta.shape != (PARAMETER_COUNT,):
        raise ValueError(f"Expected {PARAMETER_COUNT} parameters, got {theta.shape}")
    return theta


def transform_responses(
    max_idr: np.ndarray,
    max_pfa_g: np.ndarray,
    residual_drift: np.ndarray,
) -> np.ndarray:
    """Return [ln(IDR), ln(PFA), ln(RD + 5e-4)] response columns."""
    idr = np.asarray(max_idr, dtype=float)
    pfa = np.asarray(max_pfa_g, dtype=float)
    rd = np.asarray(residual_drift, dtype=float)
    if idr.shape != pfa.shape or idr.shape != rd.shape:
        raise ValueError("Response arrays must have identical shapes")
    if np.any(idr <= 0.0) or np.any(pfa <= 0.0) or np.any(rd < 0.0):
        raise ValueError("IDR and PFA must be positive; RD must be nonnegative")
    return np.column_stack((np.log(idr), np.log(pfa), np.log(rd + 5.0e-4)))


def predict(theta: np.ndarray, ln_im: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate conditional means and covariance matrices at logarithmic IM values."""
    theta = np.asarray(theta, dtype=float)
    x = np.atleast_1d(np.asarray(ln_im, dtype=float))
    if theta.shape != (PARAMETER_COUNT,):
        raise ValueError(f"Expected {PARAMETER_COUNT} parameters, got {theta.shape}")

    index = 0
    alpha = theta[index : index + M]
    index += M
    beta = theta[index : index + M]
    index += M
    gamma = theta[index : index + M]
    index += M
    kappa_mu = float(theta[index])
    index += 1
    log_base = theta[index : index + M]
    index += M
    log_slope = theta[index : index + M]
    index += M
    log_eta = theta[index : index + M]
    index += M
    kappa_var = float(theta[index])
    index += 1
    b0 = theta[index : index + M]
    index += M
    b1 = theta[index : index + M]
    index += M
    kappa_corr = float(theta[index])

    mean = (
        alpha[None, :]
        + beta[None, :] * x[:, None]
        + gamma[None, :] * np.maximum(x[:, None] - kappa_mu, 0.0)
    )

    base = np.exp(log_base)
    slope = np.exp(log_slope)
    eta = np.exp(log_eta)
    hinge = np.maximum(x[:, None] - kappa_var, 0.0)
    kernel = -np.expm1(-eta[None, :] * hinge) / eta[None, :]
    variance = base[None, :] + slope[None, :] * kernel
    standard_deviation = np.sqrt(np.maximum(variance, 1.0e-12))

    loading = b0[None, :] + b1[None, :] * np.maximum(x - kappa_corr, 0.0)[:, None]
    correlation = np.empty((len(x), M, M), dtype=float)
    for row in range(len(x)):
        omega = np.outer(loading[row], loading[row]) + np.eye(M)
        scale = np.sqrt(np.diag(omega))
        correlation[row] = omega / np.outer(scale, scale)

    covariance = (
        standard_deviation[:, :, None]
        * correlation
        * standard_deviation[:, None, :]
        + 1.0e-9 * np.eye(M)[None, :, :]
    )
    return mean, covariance


def negative_log_likelihood(
    theta: np.ndarray,
    ln_im: np.ndarray,
    log_demands: np.ndarray,
) -> float:
    """Return mean trivariate Gaussian negative log likelihood."""
    x = np.atleast_1d(np.asarray(ln_im, dtype=float))
    observed = np.asarray(log_demands, dtype=float)
    if observed.shape != (len(x), M):
        raise ValueError(f"Expected response shape {(len(x), M)}, got {observed.shape}")
    try:
        mean, covariance = predict(theta, x)
        residual = observed - mean
        total = 0.0
        constant = M * np.log(2.0 * np.pi)
        for row in range(len(x)):
            factor = np.linalg.cholesky(covariance[row])
            solved = np.linalg.solve(factor, residual[row])
            log_determinant = 2.0 * np.log(np.diag(factor)).sum()
            total += 0.5 * (constant + log_determinant + solved @ solved)
    except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
        return float("inf")
    return float(total / len(x))


def default_bounds(
    ln_im: np.ndarray,
    reference_theta: np.ndarray | None = None,
) -> list[tuple[float | None, float | None]]:
    """Construct conservative bounds for the canonical coordinates."""
    x = np.atleast_1d(np.asarray(ln_im, dtype=float))
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("ln_im must contain finite values")
    lower = float(x.min()) - 0.20
    upper = float(x.max()) + 0.20
    if lower == upper:
        lower -= 1.0
        upper += 1.0
    reference = None if reference_theta is None else np.asarray(reference_theta, dtype=float)
    if reference is not None:
        if reference.shape != (PARAMETER_COUNT,):
            raise ValueError(f"Expected {PARAMETER_COUNT} reference parameters, got {reference.shape}")
        lower = min(lower, float(reference[[9, 19, 26]].min()))
        upper = max(upper, float(reference[[9, 19, 26]].max()))
    return (
        [(None, None)] * 6
        + [(-3.0, 3.0)] * 3
        + [(lower, upper)]
        + [(-12.0, 3.0)] * 3
        + [(-12.0, 8.0)] * 3
        + [(-12.0, 8.0)] * 3
        + [(lower, upper)]
        + [(None, None)] * 3
        + [(None, None)] * 3
        + [(lower, upper)]
    )


def fit(
    ln_im: np.ndarray,
    log_demands: np.ndarray,
    initial_theta: np.ndarray | None = None,
    *,
    parameter_set: str = "all_noncollapse",
    maxiter: int = 2000,
) -> OptimizeResult:
    """Fit the model with L-BFGS-B from an explicit or stored starting point."""
    x = np.atleast_1d(np.asarray(ln_im, dtype=float))
    observed = np.asarray(log_demands, dtype=float)
    start = load_parameters(parameter_set) if initial_theta is None else np.asarray(initial_theta, dtype=float)
    if start.shape != (PARAMETER_COUNT,):
        raise ValueError(f"Expected {PARAMETER_COUNT} initial parameters, got {start.shape}")
    return minimize(
        negative_log_likelihood,
        start,
        args=(x, observed),
        method="L-BFGS-B",
        bounds=default_bounds(x, start),
        options={"maxiter": int(maxiter), "ftol": 1.0e-10},
    )
