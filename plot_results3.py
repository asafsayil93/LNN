#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare LQR (from H5 by scenario index) vs CfC (from CSV) on shared axes,
and print RMSEs for:
  (a) theta vs theta_ref
  (b) alpha vs 0
  (c) command vs 0

Features
--------
• Reads LQR states/ref/u from an H5 dataset at idx=SCEN_IDX.
• Reads CfC logs from a CSV and maps header aliases (deg→rad conversion supported).
• Optional reference alignment via cross-correlation (handles small time shifts).
• Optional masking to focus RMSE on “active” reference segments.

Author: you :)
"""

import os, json
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# ------------------- Config -------------------
DATASET_H5_PATH = "Datasets/combined_all_steps.h5"      # LQR H5
SCEN_IDX        = 250                                    # scenario index to compare
CFC_CSV_PATH    = "results/office_v1/csv/Trajectory_step_cfc_pretrain_1step_robust_office_idx_250.csv"

DT = 0.002   # seconds per sample (kept explicit for consistency with RT)
SAVE_FIG = True
FIG_PATH = "results/office_v1/figures/cfc_pretrain_1step_robust_office/lqr_vs_cfc_idx250.png"

# Align by reference if slightly off (cross-correlation)
ALIGN_BY_REF = True
MAX_LAG_SAMPLES = 400  # search in ± this many samples

# Optionally compute RMSE on active-ref regions only
FOCUS_ON_ACTIVE = False
REF_ABS_THR = 1e-3
REF_DER_THR = 1e-3

# ------------------- Canonical names -------------------
CANON = [
    "Theta (rad)",
    "Alpha (rad)",
    "Theta_dot (rad/s)",
    "Alpha_dot (rad/s)",
    "Reference (rad)",
    "Command (V)",
]

ALIASES = {
    "Theta (rad)": ["theta (rad)", "theta", "theta(rad)", "Theta(rad)", "Theta (deg)", "theta (deg)", "theta(deg)"],
    "Alpha (rad)": ["alpha (rad)", "alpha", "alpha(rad)", "Alpha(rad)", "Alpha (deg)", "alpha (deg)", "alpha(deg)"],
    "Theta_dot (rad/s)": [
        "theta_dot (rad/s)", "theta_dot", "theta vel (rad/s)",
        "Theta_dot(rad/s)", "Theta_dota (rad/s)", "Theta_dota(rad/s)",
        "theta_dot (deg/s)", "Theta_dot (deg/s)"
    ],
    "Alpha_dot (rad/s)": [
        "alpha_dot (rad/s)", "alpha_dot", "alpha vel (rad/s)",
        "Alpha_dot(rad/s)", "Alpha_dota (rad/s)", "Alpha_dota(rad/s)",
        "alpha_dot (deg/s)", "Alpha_dot (deg/s)"
    ],
    "Reference (rad)": [
        "ref (rad)", "ref", "reference", "reference(rad)",
        "Trajectory (rad)", "trajectory (rad)", "Trajectory(rad)",
        "Reference (deg)", "ref (deg)", "trajectory (deg)"
    ],
    "Command (V)": ["u (v)", "u", "command(v)", "command", "U (V)"],
}

def _norm(s: str) -> str:
    return (s.strip().lower()
            .replace("_", " ").replace("( ", "(").replace(" )", ")")
            .replace("volt", "v").replace("  ", " "))

def convert_deg_to_rad_inplace(df: pd.DataFrame) -> None:
    """Convert any column containing 'deg' in the name from degrees to radians (best effort)."""
    for col in df.columns:
        lc = col.lower()
        if "deg" in lc:
            try:
                df[col] = df[col].astype(float) * (np.pi / 180.0)
            except Exception:
                pass

def remap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map many possible header spellings into CANON and return only those columns."""
    df = df.copy()
    existing_norm = {_norm(c): c for c in df.columns}
    for canon in CANON:
        if canon in df.columns:
            continue
        for alias in ALIASES.get(canon, []):
            alias_norm = _norm(alias)
            if alias_norm in existing_norm:
                df.rename(columns={existing_norm[alias_norm]: canon}, inplace=True)
                break
    missing = [c for c in CANON if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns after alias mapping: {missing}\nFound: {list(df.columns)}")
    return df[CANON].copy()

def smart_read_csv(path: str):
    """Read CSV from path or path+'.csv', apply deg→rad and alias mapping."""
    p = path
    if not os.path.exists(p) and not p.lower().endswith(".csv"):
        if os.path.exists(p + ".csv"):
            p = p + ".csv"
    if not os.path.exists(p):
        raise FileNotFoundError(f"CSV not found: {path} (also tried '{path}.csv')")
    raw = pd.read_csv(p)
    convert_deg_to_rad_inplace(raw)
    df = remap_columns(raw)
    return df, p

def series(df, name):
    return df[name].astype(float).to_numpy()

def rmse(x, y=None):
    if y is None:
        return float(np.sqrt(np.nanmean(np.square(x))))
    return float(np.sqrt(np.nanmean(np.square(x - y))))

def best_lag_by_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """Return lag (samples) that maximizes normalized cross-correlation between a and b."""
    n = min(len(a), len(b))
    a0 = a[:n] - np.nanmean(a[:n])
    b0 = b[:n] - np.nanmean(b[:n])
    lags = np.arange(-max_lag, max_lag+1)
    best_lag = 0; best_val = -np.inf
    for L in lags:
        if L >= 0:
            aa = a0[L:n]; bb = b0[:n-L]
        else:
            aa = a0[:n+L]; bb = b0[-L:n]
        if len(aa) < 4:  # skip too-short windows
            continue
        val = np.nan_to_num(np.dot(aa, bb) / (np.sqrt(np.dot(aa, aa)) * np.sqrt(np.dot(bb, bb)) + 1e-12))
        if val > best_val:
            best_val = val; best_lag = L
    return int(best_lag)

# ------------------- H5: LQR loader -------------------
def read_lqr_from_h5_as_df(h5_path: str, idx: int) -> pd.DataFrame:
    """
    From H5 '/step' group, read states/ref/control for scenario idx and
    return a DataFrame with canonical columns in (rad, rad/s, V).
    """
    with h5py.File(h5_path, "r") as f:
        g = f["step"]
        S = g["states"][int(idx)].astype(np.float64)      # (T,F)
        R = g["ref"][int(idx)].astype(np.float64).reshape(-1)     # (T,)
        U = g["control"][int(idx)].astype(np.float64).reshape(-1) # (T,)

        # resolve states_keys
        val = f.attrs.get("states_keys", None)
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        keys = json.loads(val) if val is not None else ["theta","alpha","theta_dot","alpha_dot"]

        i_th  = keys.index("theta")      if "theta" in keys      else 0
        i_al  = keys.index("alpha")      if "alpha" in keys      else 1
        i_thd = keys.index("theta_dot")  if "theta_dot" in keys  else 2
        i_ald = keys.index("alpha_dot")  if "alpha_dot" in keys  else 3

        theta     = S[:, i_th].reshape(-1)
        alpha     = S[:, i_al].reshape(-1)
        theta_dot = S[:, i_thd].reshape(-1)
        alpha_dot = S[:, i_ald].reshape(-1)

        # equalize lengths
        T = min(len(theta), len(alpha), len(theta_dot), len(alpha_dot), len(R), len(U))
        theta, alpha, theta_dot, alpha_dot, R, U = (
            theta[:T], alpha[:T], theta_dot[:T], alpha_dot[:T], R[:T], U[:T]
        )

    df = pd.DataFrame({
        "Theta (rad)": theta,
        "Alpha (rad)": alpha,
        "Theta_dot (rad/s)": theta_dot,
        "Alpha_dot (rad/s)": alpha_dot,
        "Reference (rad)": R,
        "Command (V)": U,
    })
    return df

# ------------------- Load LQR (H5) & CfC (CSV) -------------------
df_lqr = read_lqr_from_h5_as_df(DATASET_H5_PATH, SCEN_IDX)
df_cfc, cfc_used_path = smart_read_csv(CFC_CSV_PATH)

# ------------------- Align by reference (optional) -------------------
# Coarse length equalization first
N0 = min(len(df_lqr), len(df_cfc))
df_lqr = df_lqr.iloc[:N0].reset_index(drop=True)
df_cfc = df_cfc.iloc[:N0].reset_index(drop=True)

ref_lqr_full = series(df_lqr, "Reference (rad)")
ref_cfc_full = series(df_cfc, "Reference (rad)")

lag = 0
if ALIGN_BY_REF:
    lag = best_lag_by_xcorr(ref_lqr_full, ref_cfc_full, MAX_LAG_SAMPLES)
    if lag != 0:
        if lag > 0:
            # CfC is delayed w.r.t LQR -> drop first 'lag' samples from CfC
            df_lqr = df_lqr.iloc[:-lag].reset_index(drop=True)
            df_cfc = df_cfc.iloc[lag:].reset_index(drop=True)
        else:
            L = -lag
            df_lqr = df_lqr.iloc[L:].reset_index(drop=True)
            df_cfc = df_cfc.iloc[:-L].reset_index(drop=True)
        print(f"[Align] Refs aligned by lag={lag} samples ({lag*DT:.4f} s)")
    else:
        print("[Align] Refs already aligned (lag=0)")

# ------------------- Final length & time vector -------------------
N = min(len(df_lqr), len(df_cfc))
df_lqr = df_lqr.iloc[:N].reset_index(drop=True)
df_cfc = df_cfc.iloc[:N].reset_index(drop=True)
tvec = np.arange(N, dtype=np.float64) * DT

# ------------------- Extract series -------------------
th_lqr  = series(df_lqr, "Theta (rad)")
al_lqr  = series(df_lqr, "Alpha (rad)")
thd_lqr = series(df_lqr, "Theta_dot (rad/s)")
ald_lqr = series(df_lqr, "Alpha_dot (rad/s)")
u_lqr   = series(df_lqr, "Command (V)")
ref_lqr = series(df_lqr, "Reference (rad)")

th_cfc  = series(df_cfc, "Theta (rad)")
al_cfc  = series(df_cfc, "Alpha (rad)")
thd_cfc = series(df_cfc, "Theta_dot (rad/s)")
ald_cfc = series(df_cfc, "Alpha_dot (rad/s)")
u_cfc   = series(df_cfc, "Command (V)")
ref_cfc = series(df_cfc, "Reference (rad)")

# ------------------- Optional “active” mask -------------------
mask = np.ones(N, dtype=bool)
if FOCUS_ON_ACTIVE:
    dref = np.diff(ref_lqr, prepend=ref_lqr[0])
    mask = (np.abs(ref_lqr) > REF_ABS_THR) | (np.abs(dref) > REF_DER_THR)

def mrmse(x, y=None, m=None):
    """Masked RMSE helper."""
    if m is None:
        return rmse(x, y)
    if y is None:
        z = x[m]
    else:
        z = x[m] - y[m]
    return float(np.sqrt(np.nanmean(np.square(z))))

# ------------------- RMSEs (3 requested metrics) -------------------
rmse_theta_lqr = mrmse(th_lqr, ref_lqr, mask)
rmse_theta_cfc = mrmse(th_cfc, ref_cfc, mask)
rmse_alpha_lqr = mrmse(al_lqr, None,    mask)
rmse_alpha_cfc = mrmse(al_cfc, None,    mask)
rmse_u_lqr     = mrmse(u_lqr,  None,    mask)
rmse_u_cfc     = mrmse(u_cfc,  None,    mask)

print("\n=== RMSEs ===")
print(f"(a) θ vs θ_ref :  LQR={rmse_theta_lqr:.6f} rad | CfC={rmse_theta_cfc:.6f} rad")
print(f"(b) α vs 0     :  LQR={rmse_alpha_lqr:.6f} rad | CfC={rmse_alpha_cfc:.6f} rad")
print(f"(c) u vs 0     :  LQR={rmse_u_lqr:.6f} V   | CfC={rmse_u_cfc:.6f} V")

# ------------------- Plot -------------------
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# θ
axs[0].plot(tvec, th_cfc,  label="CfC θ", color='tab:blue')
axs[0].plot(tvec, th_lqr,  label="LQR θ", color='tab:red')
axs[0].plot(tvec, ref_lqr, label="Ref θ (LQR)", linestyle="--", color='black', alpha=0.7)
axs[0].plot(tvec, ref_cfc, label="Ref θ (CfC)", linestyle=":", color='gray', alpha=0.9)
axs[0].set_title("Theta (rad)")
axs[0].grid(True); axs[0].legend(loc="upper right")

# α
axs[1].plot(tvec, al_cfc,  label="CfC α", color='tab:blue')
axs[1].plot(tvec, al_lqr,  label="LQR α", color='tab:red')
axs[1].axhline(0.0, label="Ref α = 0", linestyle="--", color='black', alpha=0.7)
axs[1].set_title("Alpha (rad)")
axs[1].grid(True); axs[1].legend(loc="upper right")

# u
axs[2].plot(tvec, u_cfc,   label="CfC u (V)", color='tab:blue')
axs[2].plot(tvec, u_lqr,   label="LQR u (V)", color='tab:red')
axs[2].set_title("Command (V)")
axs[2].set_xlabel("Time (s)")
axs[2].grid(True); axs[2].legend(loc="upper right")

# Info box
txt = (
    f"θ RMSE vs own ref: LQR={rmse_theta_lqr:.4g}, CfC={rmse_theta_cfc:.4g}\n"
    f"α RMSE (rad)     : LQR={rmse_alpha_lqr:.4g}, CfC={rmse_alpha_cfc:.4g}\n"
    f"u RMSE (V)       : LQR={rmse_u_lqr:.4g}, CfC={rmse_u_cfc:.4g}\n"
    f"(ref lag used: {lag} samp = {lag*DT:.4f}s)"
)
fig.text(0.015, 0.015, txt, fontsize=10, family="monospace")

plt.tight_layout(rect=[0, 0.05, 1, 1])

if SAVE_FIG:
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {FIG_PATH}")

plt.show()
