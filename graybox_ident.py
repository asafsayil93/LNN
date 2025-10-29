#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gray-box-ish identification utilities (importable).

What this module provides
-------------------------
• Recomputes alpha_dot from alpha via a Savitzky–Golay differentiating filter.
  Optionally pre-filters alpha with a Butterworth low-pass.
• Unwraps angles before filtering/differentiation (optional).
• Builds a one-step dataset (X, U) → Y and fits (Ad, Bd) with ridge-regularized
  least-squares in discrete-time.

Typical usages
--------------
From training code:
    from graybox_ident import estimate_graybox_from_h5

Standalone CLI:
    python -m graybox_ident \
        --data combined_all_steps.h5 \
        --indices "0:1000" \
        --sg_window 31 --sg_poly 3 \
        --butter_cutoff_hz 40 \
        --unwrap_angles \
        --ridge 1e-6 \
        --save_npz runs/plant_graybox2.npz
"""
import argparse, json, os
from typing import Tuple, List, Optional
import numpy as np
import h5py

# Optional SciPy filters (nice-to-have)
try:
    from scipy.signal import savgol_filter, butter, filtfilt
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


def _parse_indices(s: Optional[str], N: int) -> np.ndarray:
    """Parse a comma-separated list of integers or 'a:b[:step]' ranges into a unique array of indices."""
    if not s:
        return np.arange(N, dtype=np.int64)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        if ":" in p:
            toks = p.split(":")
            if len(toks) == 2:
                a, b = int(toks[0]), int(toks[1]); step = 1
            else:
                a, b, step = int(toks[0]), int(toks[1]), int(toks[2])
            out.extend(list(range(a, b, step)))
        else:
            out.append(int(p))
    out = np.asarray([i for i in out if 0 <= i < N], dtype=np.int64)
    if out.size == 0:
        out = np.arange(N, dtype=np.int64)
    return np.unique(out)


def _central_diff_uniform(y: np.ndarray, dt: float) -> np.ndarray:
    """Simple central-difference derivative with forward/backward at the ends."""
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    if len(y) < 2:
        return np.zeros_like(y)
    out[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    out[0]    = (y[1] - y[0]) / dt
    out[-1]   = (y[-1] - y[-2]) / dt
    return out


def _butter_lpf(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter (filtfilt)."""
    nyq = 0.5 * fs
    wn = min(max(cutoff_hz / nyq, 1e-6), 0.999999)
    b, a = butter(order, wn, btype="low", analog=False)
    return filtfilt(b, a, x, axis=0)


def _robust_alpha_dot(alpha: np.ndarray,
                      dt: float,
                      unwrap: bool = True,
                      sg_window: int = 31,
                      sg_poly: int = 3,
                      butter_cutoff_hz: Optional[float] = None) -> np.ndarray:
    """
    Robust derivative of alpha:
    - optional unwrap
    - optional low-pass
    - Savitzky–Golay differentiating filter (falls back to central difference if SciPy unavailable or window too short)
    """
    if unwrap:
        a = np.unwrap(alpha.astype(np.float64))
    else:
        a = alpha.astype(np.float64)

    fs = 1.0 / float(dt)
    if butter_cutoff_hz is not None and _SCIPY_OK:
        a = _butter_lpf(a, fs=fs, cutoff_hz=float(butter_cutoff_hz), order=4)

    if _SCIPY_OK:
        w = int(sg_window) if int(sg_window) % 2 == 1 else int(sg_window) + 1
        w = max(w, (sg_poly + 2) | 1)
        if len(a) >= w:
            adot = savgol_filter(a, window_length=w, polyorder=int(sg_poly),
                                 deriv=1, delta=dt, mode="interp")
        else:
            adot = _central_diff_uniform(a, dt)
    else:
        adot = _central_diff_uniform(a, dt)
    return adot.astype(np.float64)


def estimate_graybox_from_h5(h5_path: str,
                             indices: Optional[str] = None,
                             use_existing_alpha_dot: bool = False,
                             unwrap_angles: bool = True,
                             sg_window: int = 31,
                             sg_poly: int = 3,
                             butter_cutoff_hz: Optional[float] = None,
                             ridge: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit a first-order linear grey-box model in discrete time.

    Returns
    -------
    Ad : (4,4) np.ndarray
    Bd : (4,1) np.ndarray
    rmse_per_state : (4,) np.ndarray
    dt : float
    """
    with h5py.File(h5_path, "r") as f:
        g = f["step"]
        dt = float(f.attrs["dt"])
        # read states_keys
        states_keys_attr = f.attrs.get("states_keys", None)
        if isinstance(states_keys_attr, bytes):
            states_keys_attr = states_keys_attr.decode("utf-8")
        states_keys = json.loads(states_keys_attr) if states_keys_attr is not None else ["theta","alpha","theta_dot","alpha_dot"]
        N = int(g["states"].shape[0])

        idx = _parse_indices(indices, N)

        def _ix(name, default=None):
            return states_keys.index(name) if name in states_keys else default

        i_theta     = _ix("theta", 0)
        i_alpha     = _ix("alpha", 1)
        i_theta_dot = _ix("theta_dot", 2)
        i_alpha_dot = _ix("alpha_dot", 3)

        X_list, Y_list, U_list = [], [], []
        for k in idx:
            S = g["states"][int(k)].astype(np.float64)
            U = g["control"][int(k)].astype(np.float64)

            theta = S[:, i_theta]
            alpha = S[:, i_alpha]

            theta_dot = S[:, i_theta_dot] if i_theta_dot is not None else _central_diff_uniform(theta, dt)
            if (i_alpha_dot is not None) and use_existing_alpha_dot:
                alpha_dot = S[:, i_alpha_dot]
            else:
                alpha_dot = _robust_alpha_dot(alpha, dt,
                                              unwrap=unwrap_angles,
                                              sg_window=sg_window,
                                              sg_poly=sg_poly,
                                              butter_cutoff_hz=butter_cutoff_hz)
            theta_u = np.unwrap(theta) if unwrap_angles else theta
            alpha_u = np.unwrap(alpha) if unwrap_angles else alpha

            X_seq = np.column_stack([theta_u, alpha_u, theta_dot, alpha_dot])
            Xk, Yk1 = X_seq[:-1, :], X_seq[1:, :]
            Uk = U[:-1, None]

            X_list.append(Xk); Y_list.append(Yk1); U_list.append(Uk)

        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        U = np.concatenate(U_list, axis=0)

        Z = np.concatenate([X, U], axis=1)  # (N,5)
        I = np.eye(Z.shape[1])
        W = np.linalg.solve(Z.T @ Z + ridge * I, Z.T @ Y)  # (5,4)
        Ad = W[:4, :].T
        Bd = W[4:, :].reshape(4, 1)

        Y_hat = X @ Ad.T + U @ Bd.T
        rmse = np.sqrt(np.mean((Y_hat - Y)**2, axis=0))

    return Ad.astype(np.float32), Bd.astype(np.float32), rmse.astype(np.float32), dt


# ---------- CLI (optional) ----------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--indices", default=None)
    ap.add_argument("--save_npz", default=None)
    ap.add_argument("--use_existing_alpha_dot", action="store_true")
    ap.add_argument("--unwrap_angles", action="store_true")
    ap.add_argument("--sg_window", type=int, default=31)
    ap.add_argument("--sg_poly", type=int, default=3)
    ap.add_argument("--butter_cutoff_hz", type=float, default=None)
    ap.add_argument("--ridge", type=float, default=1e-6)
    args = ap.parse_args()

    Ad, Bd, rmse, dt = estimate_graybox_from_h5(
        h5_path=args.data,
        indices=args.indices,
        use_existing_alpha_dot=args.use_existing_alpha_dot,
        unwrap_angles=args.unwrap_angles,
        sg_window=args.sg_window,
        sg_poly=args.sg_poly,
        butter_cutoff_hz=args.butter_cutoff_hz,
        ridge=args.ridge,
    )

    np.set_printoptions(precision=6, suppress=True)
    print("\n=== Gray-box (discrete) fit ===")
    print("Ad =\n", Ad)
    print("\nBd =\n", Bd)
    print("\nrmse per state =", rmse)
    eig = np.linalg.eigvals(Ad); rho = float(np.max(np.abs(eig)))
    print("eig(Ad) =", eig)
    print("rho(Ad) =", rho)

    if args.save_npz:
        os.makedirs(os.path.dirname(args.save_npz), exist_ok=True)
        np.savez(args.save_npz, Ad=Ad, Bd=Bd, rmse=rmse, dt=dt)
        print(f"[OK] saving to: {args.save_npz}")

if __name__ == "__main__":
    _cli()
