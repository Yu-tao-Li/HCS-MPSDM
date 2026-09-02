
"""Shared utilities for the Exp1-Exp5 synthetic studies."""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal

try:
    import autograd.numpy as anp
    from autograd import value_and_grad
    # The current M4 objective contains many small matrix operations; autograd's
    # overhead is slower than SciPy finite differences for this script.
    HAS_AUTOGRAD = False
except Exception:  # pragma: no cover - optional acceleration fallback
    anp = None
    value_and_grad = None
    HAS_AUTOGRAD = False

# -----------------------------
# Global configuration
# -----------------------------
SEED = 20260408
rng = np.random.default_rng(SEED)

BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "outputs"

# Demand vector:
# z = [ln(IDR), ln(PFA), ln(RD)]
M = 3

# -----------------------------
# Model parameter containers
# -----------------------------
@dataclass
class FullModelParams:
    # mean parameters
    alpha: np.ndarray   # shape (m,)
    beta: np.ndarray    # shape (m,)
    gamma: np.ndarray   # shape (m,)
    kappa_mu: float     # shared-breakpoint form

    # variance parameters
    sigma_e2: np.ndarray    # shape (m,)
    delta_sigma2: np.ndarray
    eta: np.ndarray
    kappa_var: float        # shared-breakpoint form

    # correlation / factor parameters
    b0: np.ndarray      # shape (m,)
    b1: np.ndarray      # shape (m,)
    kappa_R: float
    psi: np.ndarray     # shape (m,), positive diagonal idiosyncratic variance


@dataclass
class LossParams:
    # three component groups:
    # c1 drift-sensitive
    # c2 accel-sensitive
    # c3 residual-sensitive / repairability
    q: np.ndarray  # quantities, shape (3,)
    # state means for DS=0,1,2 : shape (3,3)
    m_state: np.ndarray
    # state variances for DS=0,1,2 : shape (3,3)
    s2_state: np.ndarray


# -----------------------------
# Fixed synthetic benchmark parameters
# -----------------------------
def get_true_params() -> FullModelParams:
    return FullModelParams(
        alpha=np.array([-4.60, -0.25, -5.00]),
        beta=np.array([0.90, 0.70, 0.80]),
        gamma=np.array([0.55, 0.20, 0.85]),
        kappa_mu=-0.20,

        sigma_e2=np.array([0.08**2, 0.12**2, 0.10**2]),
        delta_sigma2=np.array([0.12**2, 0.08**2, 0.14**2]),
        eta=np.array([2.0, 1.6, 2.4]),
        kappa_var=-0.20,

        b0=np.array([0.45, 0.30, 0.25]),
        b1=np.array([0.18, -0.10, 0.22]),
        kappa_R=-0.10,
        psi=np.array([0.85, 0.95, 0.90]),
    )


def get_loss_params() -> LossParams:
    # rows = component classes, cols = DS0, DS1, DS2
    # units can be normalized repair cost
    return LossParams(
        q=np.array([30, 20, 1], dtype=float),
        m_state=np.array([
            [0.0, 1.5, 6.0],   # drift-sensitive
            [0.0, 1.0, 4.0],   # accel-sensitive
            [0.0, 3.0, 10.0],  # residual-sensitive / repairability
        ], dtype=float),
        s2_state=np.array([
            [0.0, 0.40**2, 1.20**2],
            [0.0, 0.30**2, 0.90**2],
            [0.0, 0.70**2, 2.00**2],
        ], dtype=float),
    )


# -----------------------------
# Core model functions
# -----------------------------
def pos_part(x: np.ndarray | float) -> np.ndarray | float:
    return np.maximum(x, 0.0)


def mu_fun(x: np.ndarray, p: FullModelParams) -> np.ndarray:
    """
    x: shape (n,)
    returns mu: shape (n, m)
    """
    x = np.asarray(x)
    return (
        p.alpha[None, :]
        + p.beta[None, :] * x[:, None]
        + p.gamma[None, :] * pos_part(x[:, None] - p.kappa_mu)
    )


def sigma2_fun(x: np.ndarray, p: FullModelParams) -> np.ndarray:
    """
    returns shape (n, m)
    """
    x = np.asarray(x)
    h = pos_part(x[:, None] - p.kappa_var)
    return p.sigma_e2[None, :] + p.delta_sigma2[None, :] * (1.0 - np.exp(-p.eta[None, :] * h))


def B_fun(x: np.ndarray, p: FullModelParams) -> np.ndarray:
    """
    rank-1 factor loadings, shape (n, m)
    """
    x = np.asarray(x)
    s = pos_part(x - p.kappa_R)
    return p.b0[None, :] + p.b1[None, :] * s[:, None]


def cov_fun(x: np.ndarray, p: FullModelParams) -> np.ndarray:
    """
    returns Sigma(x), shape (n, m, m)
    rank-1 factor model:
      Omega(x) = b(x)b(x)^T + Psi
      R(x) = Delta(x)^(-1/2) Omega(x) Delta(x)^(-1/2)
      Sigma(x) = D(x) R(x) D(x)
    Therefore the diagonal of Sigma(x) equals sigma2_fun(x, p).
    """
    x = np.asarray(x)
    n = x.shape[0]
    sig2 = sigma2_fun(x, p)
    sig = np.sqrt(sig2)
    b = B_fun(x, p)  # (n,m)

    Sigma = np.zeros((n, M, M), dtype=float)
    for i in range(n):
        Omega = np.outer(b[i], b[i]) + np.diag(p.psi)
        delta = np.diag(Omega)
        R = Omega / np.sqrt(np.outer(delta, delta))
        D = np.diag(sig[i])
        Sigma[i] = D @ R @ D
    return Sigma


def covariance_definition_check() -> Dict[str, float]:
    """Self-check for the covariance convention Sigma = D R D."""
    p = get_true_params()
    x = np.linspace(-2.0, 1.0, 9)
    S = cov_fun(x, p)
    target = sigma2_fun(x, p)
    diag = np.stack([np.diag(Si) for Si in S], axis=0)
    max_diag_error = float(np.max(np.abs(diag - target)))
    min_eig = float(min(np.linalg.eigvalsh(Si).min() for Si in S))
    if max_diag_error > 1.0e-10:
        raise AssertionError(f"Covariance diagonal mismatch: {max_diag_error}")
    if min_eig <= 0.0:
        raise AssertionError(f"Covariance is not positive definite: {min_eig}")
    return {"max_diag_error": max_diag_error, "min_eigenvalue": min_eig}


def corr_from_cov(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


# -----------------------------
# Simulation
# -----------------------------
def simulate_dataset(n: int, p: FullModelParams, x_low: float = -2.3, x_high: float = 1.2,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rg = np.random.default_rng(seed)
    x = rg.uniform(x_low, x_high, size=n)
    x.sort()
    mu = mu_fun(x, p)
    Sigma = cov_fun(x, p)
    z = np.zeros((n, M), dtype=float)
    for i in range(n):
        z[i] = rg.multivariate_normal(mean=mu[i], cov=Sigma[i])
    return x, z


# -----------------------------
# Fragility definitions (three component classes, DS>=1 and DS>=2)
# -----------------------------
@dataclass
class FragilityComponent:
    name: str
    # two limit states k=1,2 sharing one class-level demand combination vector
    # and one class-level capacity variance used by the reference formulation
    A: np.ndarray        # class-level demand combination vector a_c
    b: List[float]       # ordered thresholds b_{c,1} < b_{c,2}
    zeta2: float         # class-level capacity variance zeta_c^2


def get_fragility_components() -> List[FragilityComponent]:
    return [
        FragilityComponent(
            name="drift_sensitive",
            A=np.array([1.00, 0.15, 0.00]),
            b=[-2.30, -1.50],
            zeta2=0.18**2,
        ),
        FragilityComponent(
            name="accel_sensitive",
            A=np.array([0.10, 1.00, 0.00]),
            b=[-0.25, 0.15],
            zeta2=0.16**2,
        ),
        FragilityComponent(
            name="residual_sensitive",
            A=np.array([0.20, 0.05, 1.00]),
            b=[-3.20, -2.30],
            zeta2=0.20**2,
        ),
    ]


def fragility_exceedance(x_grid: np.ndarray, p: FullModelParams,
                         comp: FragilityComponent, k: int) -> np.ndarray:
    """
    k in {0,1} corresponding to DS>=1 and DS>=2.
    Both damage states of one component class share the same demand combination
    vector a_c and capacity variance zeta_c^2, so the exceedance probabilities
    are nested by construction (b_{c,1} < b_{c,2}).
    """
    mu = mu_fun(x_grid, p)
    Sigma = cov_fun(x_grid, p)
    a = comp.A
    b = comp.b[k]
    zeta2 = comp.zeta2
    num = mu @ a - b
    den = np.sqrt(np.einsum("ni,nij,nj->n", np.tile(a, (len(x_grid), 1)), Sigma, np.tile(a, (len(x_grid), 1))) + zeta2)
    return norm.cdf(num / den)


def fragility_state_probs(x_grid: np.ndarray, p: FullModelParams, comp: FragilityComponent) -> np.ndarray:
    """
    returns shape (n, 3): P(DS=0), P(DS=1), P(DS=2)
    """
    p1 = fragility_exceedance(x_grid, p, comp, 0)
    p2 = fragility_exceedance(x_grid, p, comp, 1)
    # No numerical ordering clip is needed: both states share the same a_c and
    # zeta_c^2 with ordered thresholds, so P(DS>=2) <= P(DS>=1) by construction.
    ds0 = 1.0 - p1
    ds1 = p1 - p2
    ds2 = p2
    P = np.column_stack([ds0, ds1, ds2])
    P = np.clip(P, 1e-12, 1.0)
    P /= P.sum(axis=1, keepdims=True)
    return P


# -----------------------------
# Closed-form loss moments
# -----------------------------
def loss_moments_from_muS(x_grid: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                          comps: List[FragilityComponent],
                          loss_p: LossParams) -> Tuple[np.ndarray, np.ndarray]:
    """Returns E[L|x], Var[L|x] for any fitted demand model given its
    (mu, Sigma) trajectories, using the mutually exclusive ordered
    ordered damage-state loss model used by this release:
      - exact component-level mean/var from the state probabilities P(DS_c=k);
      - exact cross-component covariance through the full joint state
        probabilities P(DS_c=k, DS_d=l) evaluated as Gaussian rectangle
        probabilities from the shared-capacity affine margins.

    The same operator is used for every model in the Exp5 loss comparison, so
    model differences reflect demand-model differences only, never loss
    propagation differences.
    """
    n = len(x_grid)
    mean_total = np.zeros(n)
    var_total = np.zeros(n)

    comp_state_probs = []
    for c, comp in enumerate(comps):
        p1 = fragility_exceedance_from_muS(x_grid, mu, Sigma, comp, 0)
        p2 = fragility_exceedance_from_muS(x_grid, mu, Sigma, comp, 1)
        P = state_probs_from_exceed(p1, p2)  # n x 3
        comp_state_probs.append(P)

        mk = loss_p.m_state[c]
        sk2 = loss_p.s2_state[c]
        q = loss_p.q[c]

        mean_c = q * np.sum(P * mk[None, :], axis=1)
        second_c = q**2 * np.sum(P * (sk2 + mk**2)[None, :], axis=1)
        var_c = second_c - mean_c**2

        mean_total += mean_c
        var_total += var_c

    # Exact cross-component covariance: Eq. (43) with the full (k, l) joint
    # state probabilities, not a DS>=1 Bernoulli approximation.
    for c in range(len(comps)):
        for d in range(c + 1, len(comps)):
            joint_states = joint_state_probabilities_from_muS(
                x_grid, mu, Sigma, comps[c], comps[d]
            )  # n x Kc x Kd
            p_c = comp_state_probs[c]  # n x 3 (DS=0,1,2)
            p_d = comp_state_probs[d]
            m_c = loss_p.m_state[c]
            m_d = loss_p.m_state[d]
            q_c, q_d = loss_p.q[c], loss_p.q[d]
            cov_cd = np.zeros(n)
            for k in range(1, 3):
                for l in range(1, 3):
                    cov_cd += (
                        q_c * q_d * m_c[k] * m_d[l]
                        * (joint_states[:, k - 1, l - 1] - p_c[:, k] * p_d[:, l])
                    )
            var_total += 2.0 * cov_cd

    return mean_total, var_total


def loss_moments_closed_form(x_grid: np.ndarray, p: FullModelParams,
                             comps: List[FragilityComponent],
                             loss_p: LossParams) -> Tuple[np.ndarray, np.ndarray]:
    """E[L|x], Var[L|x] for the full M4 model (Eqs. 38--43).

    Thin wrapper over :func:`loss_moments_from_muS` using the M4 conditional
    mean and covariance trajectories.
    """
    mu = mu_fun(x_grid, p)
    Sigma = cov_fun(x_grid, p)
    return loss_moments_from_muS(x_grid, mu, Sigma, comps, loss_p)


def joint_state_probabilities(x_grid: np.ndarray, p: FullModelParams,
                              comp_c: FragilityComponent,
                              comp_d: FragilityComponent) -> np.ndarray:
    """P(DS_c=k, DS_d=l | x) for k,l in {1,2}, shape (n, 2, 2) for the full M4 model.

    Thin wrapper over :func:`joint_state_probabilities_from_muS` using the M4
    conditional mean and covariance trajectories.
    """
    mu = mu_fun(x_grid, p)
    Sigma = cov_fun(x_grid, p)
    return joint_state_probabilities_from_muS(x_grid, mu, Sigma, comp_c, comp_d)


def joint_state_probabilities_from_muS(x_grid: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                                       comp_c: FragilityComponent,
                                       comp_d: FragilityComponent) -> np.ndarray:
    """P(DS_c=k, DS_d=l | x) for k,l in {1,2}, shape (n, 2, 2), for any fitted
    demand model given its (mu, Sigma) trajectories.

    Each class margin is an affine demand combination plus an independent
    class-level capacity term (shared within a class, independent across
    classes), so the pair (W_c, W_d) is bivariate Gaussian and every joint
    exceedance event is a Gaussian rectangle probability.  The mutually
    exclusive state probabilities follow from the rectangle differences
    P(k,l) = P(>=k,>=l) - P(>=k+1,>=l) - P(>=k,>=l+1) + P(>=k+1,>=l+1).
    """
    a1 = comp_c.A
    a2 = comp_d.A
    zeta1 = comp_c.zeta2
    zeta2 = comp_d.zeta2

    m1 = mu @ a1
    m2 = mu @ a2
    v1 = np.einsum("ni,nij,nj->n", np.tile(a1, (len(x_grid), 1)), Sigma, np.tile(a1, (len(x_grid), 1))) + zeta1
    v2 = np.einsum("ni,nij,nj->n", np.tile(a2, (len(x_grid), 1)), Sigma, np.tile(a2, (len(x_grid), 1))) + zeta2
    cov12 = np.einsum("ni,nij,nj->n", np.tile(a1, (len(x_grid), 1)), Sigma, np.tile(a2, (len(x_grid), 1)))

    out = np.zeros((len(x_grid), 2, 2))

    def joint_exceed(k: int, l: int) -> np.ndarray:
        # P(W_c >= b_{c,k}, W_d >= b_{d,l}); k,l in {1,2,3} with 3 meaning infeasible
        if k > 2 or l > 2:
            return np.zeros(len(x_grid))
        b1 = comp_c.b[k - 1]
        b2 = comp_d.b[l - 1]
        s1 = np.sqrt(np.maximum(v1, 1.0e-300))
        s2 = np.sqrt(np.maximum(v2, 1.0e-300))
        rho = np.clip(cov12 / (s1 * s2), -0.9999, 0.9999)
        result = np.zeros(len(x_grid))
        for i in range(len(x_grid)):
            mvn = multivariate_normal(mean=[m1[i], m2[i]], cov=[[v1[i], cov12[i]], [cov12[i], v2[i]]])
            result[i] = (
                1.0
                - norm.cdf((b1 - m1[i]) / s1[i])
                - norm.cdf((b2 - m2[i]) / s2[i])
                + mvn.cdf([b1, b2])
            )
        return np.clip(result, 0.0, 1.0)

    for k in (1, 2):
        for l in (1, 2):
            out[:, k - 1, l - 1] = np.clip(
                joint_exceed(k, l) - joint_exceed(k + 1, l) - joint_exceed(k, l + 1) + joint_exceed(k + 1, l + 1),
                1e-12,
                1.0,
            )
    return out


def joint_exceedance_ds1(x_grid: np.ndarray, p: FullModelParams,
                         comp_c: FragilityComponent,
                         comp_d: FragilityComponent) -> np.ndarray:
    """
    P(DS_c >=1, DS_d >=1 | x) using bivariate normal CDF for affine projections.
    """
    mu = mu_fun(x_grid, p)
    Sigma = cov_fun(x_grid, p)

    a1 = comp_c.A
    b1 = comp_c.b[0]
    zeta1 = comp_c.zeta2

    a2 = comp_d.A
    b2 = comp_d.b[0]
    zeta2 = comp_d.zeta2

    out = np.zeros(len(x_grid))
    for i in range(len(x_grid)):
        m1 = a1 @ mu[i]
        m2 = a2 @ mu[i]
        v1 = a1 @ Sigma[i] @ a1 + zeta1
        v2 = a2 @ Sigma[i] @ a2 + zeta2
        cov12 = a1 @ Sigma[i] @ a2
        s1 = math.sqrt(v1)
        s2 = math.sqrt(v2)
        rho = cov12 / (s1 * s2)
        rho = np.clip(rho, -0.9999, 0.9999)

        # P(W1>=b1, W2>=b2)
        mean = np.array([m1, m2])
        cov = np.array([[v1, cov12], [cov12, v2]])
        mvn = multivariate_normal(mean=mean, cov=cov)
        # use complement:
        p_both = 1.0 - mvn.cdf([b1, np.inf]) - mvn.cdf([np.inf, b2]) + mvn.cdf([b1, b2])
        # robust identity:
        p_both = 1.0 - norm.cdf((b1 - m1) / s1) - norm.cdf((b2 - m2) / s2) + mvn.cdf([b1, b2])
        out[i] = np.clip(p_both, 1e-12, 1.0)
    return out


# -----------------------------
# Monte Carlo consistency check
# -----------------------------
def mc_check(x_grid: np.ndarray, p: FullModelParams, comp: FragilityComponent,
             loss_p: LossParams, n_mc: int = 200000, seed: int = 11) -> Dict[str, np.ndarray]:
    rg = np.random.default_rng(seed)
    p1_cf = fragility_exceedance(x_grid, p, comp, 0)
    p2_cf = fragility_exceedance(x_grid, p, comp, 1)

    # Build one-component loss for consistency check
    q = loss_p.q[0]
    mk = loss_p.m_state[0]
    sk = np.sqrt(loss_p.s2_state[0])

    frag_mc = np.zeros(len(x_grid))
    frag2_mc = np.zeros(len(x_grid))
    mean_loss_mc = np.zeros(len(x_grid))
    var_loss_mc = np.zeros(len(x_grid))

    for i, x in enumerate(x_grid):
        mu = mu_fun(np.array([x]), p)[0]
        S = cov_fun(np.array([x]), p)[0]
        z = rg.multivariate_normal(mu, S, size=n_mc)

        # One shared class-level capacity error drives both ordered thresholds.
        a = comp.A
        eps = rg.normal(0.0, math.sqrt(comp.zeta2), size=n_mc)
        w = z @ a + eps
        ds1 = (w >= comp.b[0]).astype(int)
        ds2 = (w >= comp.b[1]).astype(int)
        state = ds1 + ds2  # 0, 1, 2 with DS>=2 nested in DS>=1 by construction

        # two-state loss: state 1 and state 2 draw from their own consequence
        # distributions
        loss = np.zeros(n_mc)
        loss[state == 1] = q * rg.normal(mk[1], sk[1], size=(state == 1).sum())
        loss[state == 2] = q * rg.normal(mk[2], sk[2], size=(state == 2).sum())

        frag_mc[i] = ds1.mean()
        frag2_mc[i] = ds2.mean()
        mean_loss_mc[i] = loss.mean()
        var_loss_mc[i] = loss.var(ddof=0)

    # closed-form counterpart
    frag_cf = p1_cf
    frag2_cf = p2_cf
    p0 = 1.0 - p1_cf
    p1 = p1_cf - p2_cf
    p2 = p2_cf
    mean_cf = q * (mk[1] * p1 + mk[2] * p2)
    var_cf = q**2 * (
        (loss_p.s2_state[0, 1] + mk[1] ** 2) * p1
        + (loss_p.s2_state[0, 2] + mk[2] ** 2) * p2
    ) - mean_cf ** 2

    return {
        "frag_cf": frag_cf,
        "frag_mc": frag_mc,
        "frag2_cf": frag2_cf,
        "frag2_mc": frag2_mc,
        "mean_cf": mean_cf,
        "mean_mc": mean_loss_mc,
        "var_cf": var_cf,
        "var_mc": var_loss_mc,
    }


# -----------------------------
# Estimation models
# -----------------------------
# The estimation routines use compact, stable parameterizations.
# M1: three independent scalar homoscedastic Gaussian models
# M2: multivariate homoscedastic Gaussian with linear mean
# M3: multivariate heteroscedastic Gaussian with linear mean, log-linear
#     marginal variances, and constant correlation fitted by joint NLL
# M4: full benchmark model (shared breakpoint, rank-1 factor, diagonal heteroscedastic)

def fit_M1_independent_homoscedastic(x: np.ndarray, z: np.ndarray) -> Dict[str, np.ndarray]:
    X = np.column_stack([np.ones_like(x), x])
    Bhat = np.linalg.lstsq(X, z, rcond=None)[0]
    resid = z - X @ Bhat
    sigma2 = np.maximum(np.mean(resid**2, axis=0), 1.0e-10)
    return {"Bhat": Bhat, "sigma2": sigma2}


def predict_M1_independent_homoscedastic(x: np.ndarray, fit: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones_like(x), x])
    mu = X @ fit["Bhat"]
    n = len(x)
    S = np.zeros((n, M, M))
    diag = np.diag(fit["sigma2"])
    for i in range(n):
        S[i] = diag
    return mu, S


def fit_M2_multivar_homoscedastic(x: np.ndarray, z: np.ndarray) -> Dict[str, np.ndarray]:
    X = np.column_stack([np.ones_like(x), x])
    Bhat = np.linalg.lstsq(X, z, rcond=None)[0]  # 2 x m
    resid = z - X @ Bhat
    S = resid.T @ resid / len(x)
    S = S + 1.0e-8 * np.eye(M)
    return {"Bhat": Bhat, "S": S}


def corr_from_raw_rhos(raw: np.ndarray) -> np.ndarray:
    rho12, rho13, rho23 = np.tanh(raw)
    R = np.array(
        [
            [1.0, rho12, rho13],
            [rho12, 1.0, rho23],
            [rho13, rho23, 1.0],
        ],
        dtype=float,
    )
    return R


def pack_M3(theta_dict: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(theta_dict["Bhat"]).ravel(),
            np.asarray(theta_dict["U"]),
            np.asarray(theta_dict["V"]),
            np.asarray(theta_dict["raw_rhos"]),
        ]
    )


def unpack_M3(theta: np.ndarray) -> Dict[str, np.ndarray]:
    idx = 0
    Bhat = theta[idx:idx + 2 * M].reshape(2, M); idx += 2 * M
    U = theta[idx:idx + M]; idx += M
    V = theta[idx:idx + M]; idx += M
    raw_rhos = theta[idx:idx + 3]; idx += 3
    return {"Bhat": Bhat, "U": U, "V": V, "raw_rhos": raw_rhos, "R": corr_from_raw_rhos(raw_rhos)}


def negloglik_M3(theta: np.ndarray, x: np.ndarray, z: np.ndarray, l2: float = 1.0e-4) -> float:
    fit = unpack_M3(theta)
    R = fit["R"]
    eig = np.linalg.eigvalsh(R)
    if eig.min() <= 1.0e-6:
        return 1.0e12 + 1.0e8 * (1.0e-6 - eig.min()) ** 2
    mu, S = predict_M2_M3(x, fit, "M3")
    nll = gaussian_nll(x, z, mu, S) * len(x)
    nll += l2 * (np.sum(fit["V"] ** 2) + np.sum(fit["raw_rhos"] ** 2))
    return float(nll)


def initial_M3_fit(x: np.ndarray, z: np.ndarray) -> Dict[str, np.ndarray]:
    X = np.column_stack([np.ones_like(x), x])
    Bhat = np.linalg.lstsq(X, z, rcond=None)[0]
    resid = z - X @ Bhat
    U = np.zeros(M)
    V = np.zeros(M)
    for j in range(M):
        y = np.log(np.maximum(resid[:, j] ** 2, 1e-8))
        H = np.column_stack([np.ones_like(x), x])
        coef = np.linalg.lstsq(H, y, rcond=None)[0]
        U[j], V[j] = coef
    std = np.sqrt(np.exp(U[None, :] + V[None, :] * x[:, None]))
    eps = resid / std
    R = np.corrcoef(eps.T)
    R = 0.98 * R + 0.02 * np.eye(M)
    raw_rhos = np.arctanh(np.clip([R[0, 1], R[0, 2], R[1, 2]], -0.95, 0.95))
    return {"Bhat": Bhat, "U": U, "V": V, "raw_rhos": raw_rhos, "R": corr_from_raw_rhos(raw_rhos)}


def fit_M3_multivar_hetero_constcorr(x: np.ndarray, z: np.ndarray) -> Dict[str, np.ndarray]:
    init = initial_M3_fit(x, z)
    theta0 = pack_M3(init)
    bounds: list[tuple[float | None, float | None]] = []
    for _ in range(2 * M):
        bounds.append((None, None))
    for _ in range(M):
        bounds.append((np.log(1.0e-5), np.log(5.0)))
    for _ in range(M):
        bounds.append((-3.0, 3.0))
    for _ in range(3):
        bounds.append((-3.0, 3.0))

    res = minimize(
        negloglik_M3,
        theta0,
        args=(x, z),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 600, "ftol": 1.0e-8},
    )
    if not res.success:
        print("WARNING: M3 optimization message:", res.message)
    return unpack_M3(res.x)


def predict_M2_M3(x: np.ndarray, fit: Dict[str, np.ndarray], model: str) -> Tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones_like(x), x])
    mu = X @ fit["Bhat"]
    n = len(x)
    S = np.zeros((n, M, M))

    if model == "M2":
        for i in range(n):
            S[i] = fit["S"]
    elif model == "M3":
        R = fit["R"]
        for i in range(n):
            var = np.exp(fit["U"] + fit["V"] * x[i])
            D = np.diag(np.sqrt(var))
            S[i] = D @ R @ D + 1.0e-10 * np.eye(M)
    else:
        raise ValueError("model must be M2 or M3")
    return mu, S


# Full model M4 with shared breakpoints and rank-1 factor
def pack_theta(theta_dict: Dict[str, np.ndarray | float]) -> np.ndarray:
    parts = [
        np.asarray(theta_dict["alpha"]),
        np.asarray(theta_dict["beta"]),
        np.asarray(theta_dict["gamma"]),
        np.array([theta_dict["kappa_mu"]]),
        np.log(np.asarray(theta_dict["sigma_e2"])),
        np.log(np.asarray(theta_dict["delta_sigma2"])),
        np.log(np.asarray(theta_dict["eta"])),
        np.array([theta_dict["kappa_var"]]),
        np.asarray(theta_dict["b0"]),
        np.asarray(theta_dict["b1"]),
        np.array([theta_dict["kappa_R"]]),
        np.log(np.asarray(theta_dict["psi"])),
    ]
    return np.concatenate(parts)


def unpack_theta(theta: np.ndarray) -> FullModelParams:
    idx = 0
    alpha = theta[idx:idx+M]; idx += M
    beta = theta[idx:idx+M]; idx += M
    gamma = theta[idx:idx+M]; idx += M
    kappa_mu = theta[idx]; idx += 1
    sigma_e2 = np.exp(theta[idx:idx+M]); idx += M
    delta_sigma2 = np.exp(theta[idx:idx+M]); idx += M
    eta = np.exp(theta[idx:idx+M]); idx += M
    kappa_var = theta[idx]; idx += 1
    b0 = theta[idx:idx+M]; idx += M
    b1 = theta[idx:idx+M]; idx += M
    kappa_R = theta[idx]; idx += 1
    psi = np.exp(theta[idx:idx+M]); idx += M
    return FullModelParams(alpha, beta, gamma, kappa_mu, sigma_e2, delta_sigma2, eta, kappa_var, b0, b1, kappa_R, psi)


def negloglik_M4(theta: np.ndarray, x: np.ndarray, z: np.ndarray, l2: float = 1e-3) -> float:
    p = unpack_theta(theta)
    mu = mu_fun(x, p)
    S = cov_fun(x, p)
    nll = 0.0
    for i in range(len(x)):
        diff = z[i] - mu[i]
        try:
            sign, logdet = np.linalg.slogdet(S[i])
            if sign <= 0:
                return 1e12
            Sinv = np.linalg.inv(S[i])
        except np.linalg.LinAlgError:
            return 1e12
        nll += 0.5 * (logdet + diff @ Sinv @ diff + M * np.log(2 * np.pi))
    # mild regularization on correlation evolution
    p_reg = unpack_theta(theta)
    nll += l2 * (np.sum(p_reg.b0**2) + np.sum(p_reg.b1**2) + np.sum(p_reg.gamma**2))
    return float(nll)


def fit_piecewise_mean_init(x: np.ndarray, z: np.ndarray) -> Dict[str, np.ndarray | float]:
    candidates = np.quantile(x, np.linspace(0.20, 0.80, 25))
    best_sse = np.inf
    best_coef = None
    best_kappa = float(np.median(x))
    for kappa in candidates:
        X = np.column_stack([np.ones_like(x), x, np.maximum(x - kappa, 0.0)])
        coef = np.linalg.lstsq(X, z, rcond=None)[0]
        resid = z - X @ coef
        sse = float(np.sum(resid**2))
        if sse < best_sse:
            best_sse = sse
            best_coef = coef
            best_kappa = float(kappa)
    assert best_coef is not None
    return {
        "alpha": best_coef[0],
        "beta": best_coef[1],
        "gamma": np.clip(best_coef[2], -2.0, 2.0),
        "kappa_mu": best_kappa,
    }


def variance_init_from_residuals(x: np.ndarray, resid: np.ndarray, kappa: float) -> Dict[str, np.ndarray | float]:
    low = x <= kappa
    high = x > kappa
    if low.sum() < 20 or high.sum() < 20:
        low = x <= np.median(x)
        high = ~low
    low_var = np.maximum(np.var(resid[low], axis=0, ddof=1), 1.0e-4)
    high_var = np.maximum(np.var(resid[high], axis=0, ddof=1), low_var * 1.05)
    return {
        "sigma_e2": np.maximum(low_var * 0.8, 1.0e-4),
        "delta_sigma2": np.maximum(high_var - low_var * 0.5, 1.0e-5),
        "eta": np.full(M, 1.5),
        "kappa_var": float(kappa),
    }


def negloglik_M4_autograd(theta: np.ndarray, x: np.ndarray, z: np.ndarray, l2: float = 1e-3):
    idx = 0
    alpha = theta[idx:idx+M]; idx += M
    beta = theta[idx:idx+M]; idx += M
    gamma = theta[idx:idx+M]; idx += M
    kappa_mu = theta[idx]; idx += 1
    sigma_e2 = anp.exp(theta[idx:idx+M]); idx += M
    delta_sigma2 = anp.exp(theta[idx:idx+M]); idx += M
    eta = anp.exp(theta[idx:idx+M]); idx += M
    kappa_var = theta[idx]; idx += 1
    b0 = theta[idx:idx+M]; idx += M
    b1 = theta[idx:idx+M]; idx += M
    kappa_R = theta[idx]; idx += 1
    psi = anp.exp(theta[idx:idx+M]); idx += M

    x_ag = anp.asarray(x)
    z_ag = anp.asarray(z)
    mu = alpha[None, :] + beta[None, :] * x_ag[:, None] + gamma[None, :] * anp.maximum(x_ag[:, None] - kappa_mu, 0.0)
    h = anp.maximum(x_ag[:, None] - kappa_var, 0.0)
    sig2 = sigma_e2[None, :] + delta_sigma2[None, :] * (1.0 - anp.exp(-eta[None, :] * h))
    sig = anp.sqrt(sig2)
    b = b0[None, :] + b1[None, :] * anp.maximum(x_ag - kappa_R, 0.0)[:, None]

    nll = 0.0
    for i in range(x_ag.shape[0]):
        omega = anp.outer(b[i], b[i]) + anp.diag(psi)
        delta = anp.diag(omega)
        r = omega / anp.sqrt(anp.outer(delta, delta))
        s = anp.diag(sig[i]) @ r @ anp.diag(sig[i]) + 1.0e-10 * anp.eye(M)
        diff = z_ag[i] - mu[i]
        sign, logdet = anp.linalg.slogdet(s)
        sol = anp.linalg.solve(s, diff)
        nll = nll + 0.5 * (logdet + anp.dot(diff, sol) + M * anp.log(2.0 * anp.pi))
    nll = nll + l2 * (anp.sum(b0**2) + anp.sum(b1**2) + anp.sum(gamma**2))
    return nll


if HAS_AUTOGRAD:
    m4_value_and_grad = value_and_grad(negloglik_M4_autograd)
else:
    m4_value_and_grad = None


def initial_theta_and_bounds_M4(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, list[tuple[float | None, float | None]]]:
    # initialization from a shared-breakpoint bilinear mean and residual moments
    mean_init = fit_piecewise_mean_init(x, z)
    X_init = np.column_stack(
        [np.ones_like(x), x, np.maximum(x - float(mean_init["kappa_mu"]), 0.0)]
    )
    coef_init = np.vstack(
        [mean_init["alpha"], mean_init["beta"], mean_init["gamma"]]
    )
    resid = z - X_init @ coef_init
    var_init = variance_init_from_residuals(x, resid, float(mean_init["kappa_mu"]))
    init = {
        "alpha": mean_init["alpha"],
        "beta": mean_init["beta"],
        "gamma": mean_init["gamma"],
        "kappa_mu": mean_init["kappa_mu"],
        "sigma_e2": var_init["sigma_e2"],
        "delta_sigma2": var_init["delta_sigma2"],
        "eta": var_init["eta"],
        "kappa_var": var_init["kappa_var"],
        "b0": np.array([0.30, 0.20, 0.20]),
        "b1": np.array([0.05, 0.00, 0.05]),
        "kappa_R": mean_init["kappa_mu"],
        "psi": np.array([0.90, 0.90, 0.90]),
    }
    theta0 = pack_theta(init)

    bounds = []
    for _ in range(M): bounds.append((None, None))  # alpha
    for _ in range(M): bounds.append((None, None))  # beta
    for _ in range(M): bounds.append((-2.0, 2.0))   # gamma
    bounds.append((x.min() - 0.5, x.max() + 0.5))   # kappa_mu
    for _ in range(M): bounds.append((np.log(1e-4), np.log(5.0)))  # sigma_e2
    for _ in range(M): bounds.append((np.log(1e-6), np.log(5.0)))  # delta_sigma2
    for _ in range(M): bounds.append((np.log(0.1), np.log(8.0)))   # eta
    bounds.append((x.min() - 0.5, x.max() + 0.5))   # kappa_var
    for _ in range(M): bounds.append((-2.0, 2.0))   # b0
    for _ in range(M): bounds.append((-2.0, 2.0))   # b1
    bounds.append((x.min() - 0.5, x.max() + 0.5))   # kappa_R
    for _ in range(M): bounds.append((np.log(1e-3), np.log(5.0)))  # psi

    return theta0, bounds


def _torch_m4_value_grad_factory(x: np.ndarray, z: np.ndarray, l2: float = 1e-3):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    x_t = torch.as_tensor(np.asarray(x, dtype=float), dtype=dtype, device=device)
    z_t = torch.as_tensor(np.asarray(z, dtype=float), dtype=dtype, device=device)
    eye = torch.eye(M, dtype=dtype, device=device).unsqueeze(0)
    two_pi = torch.tensor(2.0 * np.pi, dtype=dtype, device=device)

    def objective(theta_np: np.ndarray) -> tuple[float, np.ndarray]:
        theta = torch.as_tensor(theta_np, dtype=dtype, device=device).clone().detach().requires_grad_(True)
        idx = 0
        alpha = theta[idx:idx+M]; idx += M
        beta = theta[idx:idx+M]; idx += M
        gamma = theta[idx:idx+M]; idx += M
        kappa_mu = theta[idx]; idx += 1
        sigma_e2 = torch.exp(theta[idx:idx+M]); idx += M
        delta_sigma2 = torch.exp(theta[idx:idx+M]); idx += M
        eta = torch.exp(theta[idx:idx+M]); idx += M
        kappa_var = theta[idx]; idx += 1
        b0 = theta[idx:idx+M]; idx += M
        b1 = theta[idx:idx+M]; idx += M
        kappa_R = theta[idx]; idx += 1
        psi = torch.exp(theta[idx:idx+M])

        mu = alpha[None, :] + beta[None, :] * x_t[:, None] + gamma[None, :] * torch.clamp(x_t[:, None] - kappa_mu, min=0.0)
        h = torch.clamp(x_t[:, None] - kappa_var, min=0.0)
        sig2 = sigma_e2[None, :] + delta_sigma2[None, :] * (1.0 - torch.exp(-eta[None, :] * h))
        sig = torch.sqrt(sig2)
        b = b0[None, :] + b1[None, :] * torch.clamp(x_t - kappa_R, min=0.0)[:, None]

        omega = b[:, :, None] * b[:, None, :] + torch.diag_embed(psi.expand(x_t.shape[0], M))
        delta = torch.diagonal(omega, dim1=1, dim2=2)
        r = omega / torch.sqrt(delta[:, :, None] * delta[:, None, :])
        s = sig[:, :, None] * r * sig[:, None, :] + 1.0e-10 * eye

        diff = (z_t - mu).unsqueeze(-1)
        chol = torch.linalg.cholesky(s)
        solved = torch.cholesky_solve(diff, chol)
        maha = torch.matmul(diff.transpose(1, 2), solved).squeeze(-1).squeeze(-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(dim=1)
        nll = 0.5 * (logdet + maha + M * torch.log(two_pi)).sum()
        nll = nll + l2 * (torch.sum(b0**2) + torch.sum(b1**2) + torch.sum(gamma**2))
        nll.backward()
        return float(nll.detach().cpu().item()), theta.grad.detach().cpu().numpy().astype(float)

    return objective, str(device)


def fit_M4_torch(x: np.ndarray, z: np.ndarray, return_info: bool = False):
    theta0, bounds = initial_theta_and_bounds_M4(x, z)
    objective, device = _torch_m4_value_grad_factory(x, z)
    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 1000, "maxfun": 20000, "ftol": 1e-8},
    )
    if not res.success:
        print("WARNING: M4 torch optimization message:", res.message)
    params = unpack_theta(res.x)
    info = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "fun": float(res.fun),
        "nit": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)),
        "njev": int(getattr(res, "njev", -1)),
        "optimizer": f"L-BFGS-B-torch-autograd-{device}",
    }
    if return_info:
        return params, info
    return params


def fit_M4(x: np.ndarray, z: np.ndarray, return_info: bool = False, backend: str = "scipy"):
    if backend not in {"scipy", "torch", "auto"}:
        raise ValueError("backend must be 'scipy', 'torch', or 'auto'")
    if backend in {"torch", "auto"}:
        try:
            return fit_M4_torch(x, z, return_info=return_info)
        except ImportError:
            if backend == "torch":
                raise

    theta0, bounds = initial_theta_and_bounds_M4(x, z)

    if HAS_AUTOGRAD:
        def objective(theta):
            value, gradient = m4_value_and_grad(theta, x, z)
            return float(value), np.asarray(gradient, dtype=float)

        res = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 1000, "maxfun": 20000, "ftol": 1e-8},
        )
        optimizer = "L-BFGS-B-autograd"
    else:
        res = minimize(
            negloglik_M4,
            theta0,
            args=(x, z),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "maxfun": 20000, "ftol": 1e-8},
        )
        optimizer = "L-BFGS-B-finite-difference"
    if not res.success:
        print("WARNING: M4 optimization message:", res.message)
    params = unpack_theta(res.x)
    info = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "fun": float(res.fun),
        "nit": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)),
        "njev": int(getattr(res, "njev", -1)),
        "optimizer": optimizer,
    }
    if return_info:
        return params, info
    return params


# -----------------------------
# Model predictions for fragility/loss
# -----------------------------
def fragility_exceedance_from_muS(x_grid: np.ndarray, mu: np.ndarray, S: np.ndarray,
                                  comp: FragilityComponent, k: int) -> np.ndarray:
    a = comp.A
    b = comp.b[k]
    zeta2 = comp.zeta2
    num = mu @ a - b
    den = np.sqrt(np.einsum("ni,nij,nj->n", np.tile(a, (len(x_grid), 1)), S, np.tile(a, (len(x_grid), 1))) + zeta2)
    return norm.cdf(num / den)


def state_probs_from_exceed(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    # Shared class-level capacity with ordered thresholds nests the events by
    # construction, so no numerical ordering clip is needed here.
    P = np.column_stack([1-p1, p1-p2, p2])
    P = np.clip(P, 1e-12, 1.0)
    P /= P.sum(axis=1, keepdims=True)
    return P


def loss_curves_from_model(x_grid: np.ndarray, model_name: str, model_fit,
                           comps: List[FragilityComponent], loss_p: LossParams) -> Tuple[np.ndarray, np.ndarray]:
    """E[L|x], Var[L|x] for a fitted model.

    Every model, including M1--M3, uses the same mutually exclusive ordered
    damage-state loss operator of Eqs. (38)--(43) with the full cross-component
    covariance: different classes share demand coordinates through their
    affine limit states even when the fitted covariance is diagonal, so the
    cross-component state covariance is nonzero for every model in the
    comparison.  Model differences in the loss curves therefore reflect
    demand-model differences only.
    """
    if model_name == "M1":
        mu, S = predict_M1_independent_homoscedastic(x_grid, model_fit)
        return loss_moments_from_muS(x_grid, mu, S, comps, loss_p)

    if model_name in {"M2", "M3"}:
        mu, S = predict_M2_M3(x_grid, model_fit, model_name)
        return loss_moments_from_muS(x_grid, mu, S, comps, loss_p)

    if model_name == "M4":
        return loss_moments_closed_form(x_grid, model_fit, comps, loss_p)

    raise ValueError("unknown model_name")


# -----------------------------
# Metrics
# -----------------------------
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rel_frob(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B) / max(np.linalg.norm(A), 1e-12))


def gaussian_nll(x: np.ndarray, z: np.ndarray, mu: np.ndarray, S: np.ndarray) -> float:
    ll = 0.0
    for i in range(len(x)):
        ll += multivariate_normal.logpdf(z[i], mean=mu[i], cov=S[i], allow_singular=False)
    return float(-ll / len(x))


def test_nll_M1(x: np.ndarray, z: np.ndarray, fit: Dict[str, np.ndarray]) -> float:
    mu, S = predict_M1_independent_homoscedastic(x, fit)
    return gaussian_nll(x, z, mu, S)


def test_nll_M2M3(x: np.ndarray, z: np.ndarray, fit: Dict[str, np.ndarray], model: str) -> float:
    mu, S = predict_M2_M3(x, fit, model)
    return gaussian_nll(x, z, mu, S)


def test_nll_M4(x: np.ndarray, z: np.ndarray, p: FullModelParams) -> float:
    mu = mu_fun(x, p)
    S = cov_fun(x, p)
    return gaussian_nll(x, z, mu, S)


# -----------------------------
# Plotting helpers
# -----------------------------
def savefig(name: str):
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close()


# -----------------------------
# Experiment 1
# -----------------------------
def experiment_1_closed_form_consistency():
    p = get_true_params()
    loss_p = get_loss_params()
    comps = get_fragility_components()

    x_grid = np.linspace(-1.8, 1.0, 21)
    result = mc_check(x_grid, p, comps[0], loss_p, n_mc=120000, seed=101)

    df = pd.DataFrame({
        "x": x_grid,
        "frag_cf": result["frag_cf"],
        "frag_mc": result["frag_mc"],
        "frag_abs_err": np.abs(result["frag_cf"] - result["frag_mc"]),
        "mean_cf": result["mean_cf"],
        "mean_mc": result["mean_mc"],
        "mean_rel_err": np.abs(result["mean_cf"] - result["mean_mc"]) / np.maximum(np.abs(result["mean_mc"]), 1e-8),
        "var_cf": result["var_cf"],
        "var_mc": result["var_mc"],
        "var_rel_err": np.abs(result["var_cf"] - result["var_mc"]) / np.maximum(np.abs(result["var_mc"]), 1e-8),
    })
    df.to_csv(OUTDIR / "exp1_closed_form_consistency.csv", index=False)

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, result["frag_cf"], label="Closed-form")
    plt.scatter(x_grid, result["frag_mc"], s=20, label="Monte Carlo")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Fragility")
    plt.legend()
    plt.title("E1 Fragility: closed-form vs Monte Carlo")
    savefig("exp1_fragility_consistency")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, result["mean_cf"], label="Closed-form")
    plt.scatter(x_grid, result["mean_mc"], s=20, label="Monte Carlo")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("E[L|x]")
    plt.legend()
    plt.title("E1 Mean loss: closed-form vs Monte Carlo")
    savefig("exp1_mean_loss_consistency")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, result["var_cf"], label="Closed-form")
    plt.scatter(x_grid, result["var_mc"], s=20, label="Monte Carlo")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Var[L|x]")
    plt.legend()
    plt.title("E1 Loss variance: closed-form vs Monte Carlo")
    savefig("exp1_var_loss_consistency")

    summary = {
        "max_frag_abs_err": float(df["frag_abs_err"].max()),
        "max_mean_rel_err": float(df["mean_rel_err"].max()),
        "max_var_rel_err": float(df["var_rel_err"].max()),
    }
    with open(OUTDIR / "exp1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# -----------------------------
# Experiment 2
# -----------------------------
def experiment_2_parameter_recovery():
    p_true = get_true_params()
    comps = get_fragility_components()
    loss_p = get_loss_params()

    x_train, z_train = simulate_dataset(1200, p_true, seed=202)
    x_test, z_test = simulate_dataset(400, p_true, seed=203)

    p_hat = fit_M4(x_train, z_train)

    x_grid = np.linspace(-2.0, 1.0, 120)
    mu_true = mu_fun(x_grid, p_true)
    mu_hat = mu_fun(x_grid, p_hat)

    S_true = cov_fun(x_grid, p_true)
    S_hat = cov_fun(x_grid, p_hat)

    # Metrics
    mu_rmse = [rmse(mu_true[:, j], mu_hat[:, j]) for j in range(M)]
    cov_err = np.mean([rel_frob(S_true[i], S_hat[i]) for i in range(len(x_grid))])

    # Fragility & loss recovery
    p1_true = fragility_exceedance(x_grid, p_true, comps[0], 0)
    p1_hat = fragility_exceedance(x_grid, p_hat, comps[0], 0)
    frag_rmse = rmse(p1_true, p1_hat)

    mean_true, var_true = loss_moments_closed_form(x_grid, p_true, comps, loss_p)
    mean_hat, var_hat = loss_moments_closed_form(x_grid, p_hat, comps, loss_p)
    mean_rel = float(np.mean(np.abs(mean_true - mean_hat) / np.maximum(np.abs(mean_true), 1e-8)))
    var_rel = float(np.mean(np.abs(var_true - var_hat) / np.maximum(np.abs(var_true), 1e-8)))

    # Plots
    names = ["ln(IDR)", "ln(PFA)", "ln(RD)"]
    for j in range(M):
        plt.figure(figsize=(6.2, 4.2))
        plt.plot(x_grid, mu_true[:, j], label="True")
        plt.plot(x_grid, mu_hat[:, j], "--", label="Estimated")
        plt.xlabel("x = ln(IM)")
        plt.ylabel(names[j] + " mean")
        plt.legend()
        plt.title(f"E2 Mean recovery: {names[j]}")
        savefig(f"exp2_mu_recovery_{j+1}")

    for j in range(M):
        plt.figure(figsize=(6.2, 4.2))
        plt.plot(x_grid, np.diagonal(S_true, axis1=1, axis2=2)[:, j], label="True")
        plt.plot(x_grid, np.diagonal(S_hat, axis1=1, axis2=2)[:, j], "--", label="Estimated")
        plt.xlabel("x = ln(IM)")
        plt.ylabel(names[j] + " variance")
        plt.legend()
        plt.title(f"E2 Variance recovery: {names[j]}")
        savefig(f"exp2_var_recovery_{j+1}")

    # Correlation heatmaps at low / mid / high x
    idxs = [10, 60, 110]
    labels = ["low", "mid", "high"]
    for idx, lab in zip(idxs, labels):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
        axes[0].imshow(corr_from_cov(S_true[idx]), vmin=-1, vmax=1)
        axes[0].set_title(f"True corr ({lab})")
        axes[1].imshow(corr_from_cov(S_hat[idx]), vmin=-1, vmax=1)
        axes[1].set_title(f"Estimated corr ({lab})")
        savefig(f"exp2_corr_heatmap_{lab}")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, p1_true, label="True")
    plt.plot(x_grid, p1_hat, "--", label="Estimated")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Fragility")
    plt.legend()
    plt.title("E2 Fragility recovery")
    savefig("exp2_fragility_recovery")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, mean_true, label="True E[L|x]")
    plt.plot(x_grid, mean_hat, "--", label="Estimated E[L|x]")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Mean loss")
    plt.legend()
    plt.title("E2 Mean loss recovery")
    savefig("exp2_mean_loss_recovery")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, var_true, label="True Var[L|x]")
    plt.plot(x_grid, var_hat, "--", label="Estimated Var[L|x]")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Loss variance")
    plt.legend()
    plt.title("E2 Loss variance recovery")
    savefig("exp2_var_loss_recovery")

    summary = {
        "mu_rmse": mu_rmse,
        "avg_cov_rel_frob_error": cov_err,
        "fragility_rmse": frag_rmse,
        "mean_loss_avg_rel_error": mean_rel,
        "var_loss_avg_rel_error": var_rel,
        "test_nll_true_model": test_nll_M4(x_test, z_test, p_hat),
    }
    with open(OUTDIR / "exp2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# -----------------------------
# Experiment 3
# -----------------------------
def experiment_3_correlation_bias():
    # 2D analytical bias visualization
    x_grid = np.linspace(-2.0, 1.0, 160)
    mu1 = -3.8 + 1.2 * x_grid
    mu2 = -0.4 + 0.8 * x_grid
    s1 = 0.35
    s2 = 0.30
    b = -1.5

    def scenario(rho_true: float, loss_dependence: float, label: str) -> tuple[pd.DataFrame, dict[str, float | str]]:
        mx = mu1 + mu2 - b
        p_true = norm.cdf(mx / np.sqrt(s1**2 + s2**2 + 2 * rho_true * s1 * s2))
        p_ind = norm.cdf(mx / np.sqrt(s1**2 + s2**2))
        dp = p_true - p_ind

        # two-component Bernoulli-loss illustration
        m1, m2 = 8.0, 12.0
        p1 = norm.cdf((mu1 + 0.2 - (-2.0)) / 0.45)
        p2 = norm.cdf((mu2 + 0.1 - (-0.1)) / 0.42)
        # simple positive dependence proxy for visualization
        joint = np.minimum(
            np.minimum(p1, p2),
            p1 * p2 + loss_dependence * np.sqrt(p1 * (1 - p1) * p2 * (1 - p2)),
        )
        var_true = m1**2 * p1 * (1 - p1) + m2**2 * p2 * (1 - p2) + 2 * m1 * m2 * (joint - p1 * p2)
        var_ind = m1**2 * p1 * (1 - p1) + m2**2 * p2 * (1 - p2)
        df = pd.DataFrame({
            "scenario": label,
            "x": x_grid,
            "rho_true": rho_true,
            "frag_true": p_true,
            "frag_independent": p_ind,
            "frag_bias": dp,
            "var_true": var_true,
            "var_independent": var_ind,
            "var_bias": var_true - var_ind,
        })
        summary = {
            "scenario": label,
            "rho_true": float(rho_true),
            "loss_dependence_proxy": float(loss_dependence),
            "max_abs_fragility_bias": float(np.max(np.abs(dp))),
            "max_rel_lossvar_bias": float(np.max(np.abs((var_true - var_ind) / np.maximum(np.abs(var_true), 1e-8)))),
        }
        return df, summary

    moderate_df, moderate_summary = scenario(0.30, 0.10, "moderate")
    strong_df, strong_summary = scenario(0.55, 0.18, "strong")
    df = pd.concat([moderate_df, strong_df], ignore_index=True)
    p_true = strong_df["frag_true"].to_numpy()
    p_ind = strong_df["frag_independent"].to_numpy()
    dp = strong_df["frag_bias"].to_numpy()
    var_true = strong_df["var_true"].to_numpy()
    var_ind = strong_df["var_independent"].to_numpy()

    df.to_csv(OUTDIR / "exp3_correlation_bias.csv", index=False)
    pd.DataFrame([moderate_summary, strong_summary]).to_csv(OUTDIR / "exp3_correlation_bias_scenarios.csv", index=False)

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, p_true, label="True correlated")
    plt.plot(x_grid, p_ind, "--", label="Incorrect independence")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Fragility")
    plt.legend()
    plt.title("E3 Correlation misspecification on fragility")
    savefig("exp3_fragility_corr_bias")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, dp)
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Fragility bias")
    plt.title("E3 Fragility bias = true - independent")
    savefig("exp3_fragility_corr_bias_delta")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, var_true, label="True correlated")
    plt.plot(x_grid, var_ind, "--", label="Incorrect independence")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Var[L|x]")
    plt.legend()
    plt.title("E3 Correlation misspecification on loss variance")
    savefig("exp3_lossvar_corr_bias")

    summary = {
        "moderate": moderate_summary,
        "strong": strong_summary,
        "primary_plotted_scenario": "strong",
    }
    with open(OUTDIR / "exp3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# -----------------------------
# Experiment 4
# -----------------------------
def experiment_4_hetero_bias():
    x_grid = np.linspace(-2.0, 1.2, 200)
    mu = -3.8 + 1.15 * x_grid + 0.55 * np.maximum(x_grid + 0.25, 0.0)
    sigma_e2 = 0.10**2
    eta = 2.0
    kappa = -0.20
    b = -1.7

    def scenario(delta_sigma: float, label: str) -> tuple[pd.DataFrame, dict[str, float | str]]:
        sigma2 = sigma_e2 + delta_sigma**2 * (1.0 - np.exp(-eta * np.maximum(x_grid - kappa, 0.0)))
        sigma = np.sqrt(sigma2)
        sigma_bar = float(np.mean(sigma))
        p_het = norm.cdf((mu - b) / sigma)
        p_hom = norm.cdf((mu - b) / sigma_bar)

        # two-state loss
        m = 10.0
        mean_het = m * p_het
        mean_hom = m * p_hom
        var_het = m**2 * p_het * (1 - p_het)
        var_hom = m**2 * p_hom * (1 - p_hom)
        high = x_grid > kappa
        df = pd.DataFrame({
            "scenario": label,
            "x": x_grid,
            "delta_sigma": delta_sigma,
            "sigma_true": sigma,
            "sigma_const": sigma_bar,
            "frag_het": p_het,
            "frag_hom": p_hom,
            "mean_het": mean_het,
            "mean_hom": mean_hom,
            "var_het": var_het,
            "var_hom": var_hom,
        })
        active = high & (var_het > 0.05 * np.max(var_het))
        summary = {
            "scenario": label,
            "delta_sigma": float(delta_sigma),
            "max_abs_fragility_bias_high_region": float(np.max(np.abs(p_het[high] - p_hom[high]))),
            "max_abs_lossvar_bias_high_region": float(np.max(np.abs(var_het[high] - var_hom[high]))),
            "max_rel_lossvar_bias_high_region": float(
                np.max(np.abs((var_het[high] - var_hom[high]) / np.maximum(np.abs(var_het[high]), 1e-8)))
            ),
            "max_rel_lossvar_bias_active_region": float(
                np.max(np.abs((var_het[active] - var_hom[active]) / np.maximum(np.abs(var_het[active]), 1e-8)))
            ),
        }
        return df, summary

    moderate_df, moderate_summary = scenario(0.05, "moderate")
    strong_df, strong_summary = scenario(0.16, "strong")
    df = pd.concat([moderate_df, strong_df], ignore_index=True)
    sigma = strong_df["sigma_true"].to_numpy()
    sigma_bar = float(strong_df["sigma_const"].iloc[0])
    p_het = strong_df["frag_het"].to_numpy()
    p_hom = strong_df["frag_hom"].to_numpy()
    var_het = strong_df["var_het"].to_numpy()
    var_hom = strong_df["var_hom"].to_numpy()

    df.to_csv(OUTDIR / "exp4_hetero_bias.csv", index=False)
    pd.DataFrame([moderate_summary, strong_summary]).to_csv(OUTDIR / "exp4_hetero_bias_scenarios.csv", index=False)

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, sigma, label="True heteroscedastic")
    plt.plot(x_grid, np.full_like(x_grid, sigma_bar), "--", label="Constant variance")
    plt.axvline(kappa, linestyle="--", linewidth=0.8)
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Std. dev.")
    plt.legend()
    plt.title("E4 Variance misspecification")
    savefig("exp4_sigma_bias")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, p_het, label="True heteroscedastic")
    plt.plot(x_grid, p_hom, "--", label="Incorrect homoscedastic")
    plt.axvline(kappa, linestyle="--", linewidth=0.8)
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Fragility")
    plt.legend()
    plt.title("E4 Heteroscedasticity misspecification on fragility")
    savefig("exp4_fragility_hetero_bias")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_grid, var_het, label="True heteroscedastic")
    plt.plot(x_grid, var_hom, "--", label="Incorrect homoscedastic")
    plt.axvline(kappa, linestyle="--", linewidth=0.8)
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Var[L|x]")
    plt.legend()
    plt.title("E4 Heteroscedasticity misspecification on loss variance")
    savefig("exp4_lossvar_hetero_bias")

    summary = {
        "moderate": moderate_summary,
        "strong": strong_summary,
        "primary_plotted_scenario": "strong",
    }
    with open(OUTDIR / "exp4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# -----------------------------
# Experiment 5
# -----------------------------
def experiment_5_loss_comparison(m4_backend: str = "scipy", n_train: int = 1500, n_test: int = 500,
                                 train_seed: int = 304, test_seed: int = 305):
    p_true = get_true_params()
    comps = get_fragility_components()
    loss_p = get_loss_params()

    # Use one synthetic "external benchmark" split
    x_train, z_train = simulate_dataset(n_train, p_true, seed=train_seed)
    x_test, z_test = simulate_dataset(n_test, p_true, seed=test_seed)

    m1 = fit_M1_independent_homoscedastic(x_train, z_train)
    m2 = fit_M2_multivar_homoscedastic(x_train, z_train)
    m3 = fit_M3_multivar_hetero_constcorr(x_train, z_train)
    m4, m4_info = fit_M4(x_train, z_train, return_info=True, backend=m4_backend)

    x_grid = np.linspace(-2.0, 1.0, 140)

    # Demand-layer out-of-sample NLL
    nlls = {
        "M1": test_nll_M1(x_test, z_test, m1),
        "M2": test_nll_M2M3(x_test, z_test, m2, "M2"),
        "M3": test_nll_M2M3(x_test, z_test, m3, "M3"),
        "M4": test_nll_M4(x_test, z_test, m4),
    }
    nll_per_dim = {key: value / M for key, value in nlls.items()}
    nll_ranking = sorted(nlls, key=nlls.get)

    # Loss curves
    mean_M1, var_M1 = loss_curves_from_model(x_grid, "M1", m1, comps, loss_p)
    mean_M2, var_M2 = loss_curves_from_model(x_grid, "M2", m2, comps, loss_p)
    mean_M3, var_M3 = loss_curves_from_model(x_grid, "M3", m3, comps, loss_p)
    mean_M4, var_M4 = loss_curves_from_model(x_grid, "M4", m4, comps, loss_p)

    # Normal-moment high-loss proxy: mean + 1.645*sqrt(var).
    # This is not an empirical 95% loss quantile.
    q95_M1 = mean_M1 + 1.645 * np.sqrt(np.maximum(var_M1, 0))
    q95_M2 = mean_M2 + 1.645 * np.sqrt(np.maximum(var_M2, 0))
    q95_M3 = mean_M3 + 1.645 * np.sqrt(np.maximum(var_M3, 0))
    q95_M4 = mean_M4 + 1.645 * np.sqrt(np.maximum(var_M4, 0))

    df = pd.DataFrame({
        "x": x_grid,
        "mean_M1": mean_M1, "mean_M2": mean_M2, "mean_M3": mean_M3, "mean_M4": mean_M4,
        "var_M1": var_M1, "var_M2": var_M2, "var_M3": var_M3, "var_M4": var_M4,
        "q95_M1": q95_M1, "q95_M2": q95_M2, "q95_M3": q95_M3, "q95_M4": q95_M4,
    })
    df.to_csv(OUTDIR / "exp5_loss_comparison.csv", index=False)

    plt.figure(figsize=(6.6, 4.4))
    plt.plot(x_grid, mean_M1, label="M1 independent homoscedastic")
    plt.plot(x_grid, mean_M2, label="M2 multivar homoscedastic")
    plt.plot(x_grid, mean_M3, label="M3 multivar hetero")
    plt.plot(x_grid, mean_M4, label="M4 full model")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("E[L|x]")
    plt.legend(fontsize=8)
    plt.title("E5 Mean loss comparison")
    savefig("exp5_mean_loss_comparison")

    plt.figure(figsize=(6.6, 4.4))
    plt.plot(x_grid, var_M1, label="M1")
    plt.plot(x_grid, var_M2, label="M2")
    plt.plot(x_grid, var_M3, label="M3")
    plt.plot(x_grid, var_M4, label="M4")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Var[L|x]")
    plt.legend(fontsize=8)
    plt.title("E5 Loss variance comparison")
    savefig("exp5_var_loss_comparison")

    plt.figure(figsize=(6.6, 4.4))
    plt.plot(x_grid, q95_M1, label="M1")
    plt.plot(x_grid, q95_M2, label="M2")
    plt.plot(x_grid, q95_M3, label="M3")
    plt.plot(x_grid, q95_M4, label="M4")
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Normal-moment loss proxy")
    plt.legend(fontsize=8)
    plt.title("E5 High-quantile loss comparison")
    savefig("exp5_q95_loss_comparison")

    plt.figure(figsize=(6.6, 4.4))
    plt.plot(x_grid, (var_M1 - var_M4) / np.maximum(np.abs(var_M4), 1e-8), label="M1 vs M4")
    plt.plot(x_grid, (var_M2 - var_M4) / np.maximum(np.abs(var_M4), 1e-8), label="M2 vs M4")
    plt.plot(x_grid, (var_M3 - var_M4) / np.maximum(np.abs(var_M4), 1e-8), label="M3 vs M4")
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.xlabel("x = ln(IM)")
    plt.ylabel("Relative bias in Var[L|x]")
    plt.legend(fontsize=8)
    plt.title("E5 Relative loss-variance bias against M4")
    savefig("exp5_var_bias_against_M4")

    summary = {
        "test_nll_per_observation_joint_3d": nlls,
        "test_nll_per_observation_per_dimension": nll_per_dim,
        "test_nll_ranking_best_to_worst": nll_ranking,
        "n_train": n_train,
        "n_test": n_test,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "m4_backend_requested": m4_backend,
        "m4_optimizer_info": m4_info,
        "all_models_evaluated_on_same_3d_response_vector": True,
        "m1_definition": "three independent scalar homoscedastic Gaussian models; joint density is product of three marginals",
        "m3_definition": "linear mean, log-linear marginal variance, constant correlation fitted by joint Gaussian NLL",
        "loss_proxy_note": "mean + 1.645*sqrt(variance) is reported as a normal-moment proxy, not an empirical 95% quantile",
        "max_rel_bias_var_M1_vs_M4": float(np.max(np.abs((var_M1 - var_M4) / np.maximum(np.abs(var_M4), 1e-8)))),
        "max_rel_bias_var_M2_vs_M4": float(np.max(np.abs((var_M2 - var_M4) / np.maximum(np.abs(var_M4), 1e-8)))),
        "max_rel_bias_var_M3_vs_M4": float(np.max(np.abs((var_M3 - var_M4) / np.maximum(np.abs(var_M4), 1e-8)))),
        "max_rel_bias_q95_M1_vs_M4": float(np.max(np.abs((q95_M1 - q95_M4) / np.maximum(np.abs(q95_M4), 1e-8)))),
        "max_rel_bias_q95_M2_vs_M4": float(np.max(np.abs((q95_M2 - q95_M4) / np.maximum(np.abs(q95_M4), 1e-8)))),
        "max_rel_bias_q95_M3_vs_M4": float(np.max(np.abs((q95_M3 - q95_M4) / np.maximum(np.abs(q95_M4), 1e-8)))),
    }
    with open(OUTDIR / "exp5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# -----------------------------
# Main
# -----------------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summaries = {}

    print("Checking covariance definition...")
    summaries["covariance_definition_check"] = covariance_definition_check()
    print("Running Exp1...")
    summaries["Exp1"] = experiment_1_closed_form_consistency()
    print("Running Exp2...")
    summaries["Exp2"] = experiment_2_parameter_recovery()
    print("Running Exp3...")
    summaries["Exp3"] = experiment_3_correlation_bias()
    print("Running Exp4...")
    summaries["Exp4"] = experiment_4_hetero_bias()
    print("Running Exp5...")
    summaries["Exp5"] = experiment_5_loss_comparison()

    with open(OUTDIR / "all_summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("\nDone. Results written to:", OUTDIR.resolve())
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
