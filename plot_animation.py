#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animate LQR vs CfC from CSV logs and export to MP4.

What this script does
---------------------
• Loads two CSVs (LQR and CfC) and normalizes their column names to a canonical set.
• Builds a 3-panel Matplotlib animation for θ, α, and u over time.
• Supports pre-roll (zeros), post-hold (freeze last frame), and exact frame decimation
  to hit a target VIDEO_FPS independent of the sample period DT.
• Automatically discovers FFmpeg on Windows (WinGet / WindowsApps), or you can
  force a specific binary via FFMPEG_PATH.

Usage tips
----------
- Ensure your CSVs have (or can be mapped to) these columns:
  ["Theta (rad)","Alpha (rad)","Theta_dot (rad/s)","Alpha_dot (rad/s)","Reference (rad)","Command (V)"].
- If your CSV paths omit ".csv", the loader tries "<path>.csv" as a fallback.

Author: you :)
"""

import os, shutil, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =================== Config ===================
LQR_CSV_PATH = "results/office_v2/csv/Trajectory_step_lqr_idx_100_mass.csv"
CFC_CSV_PATH = "results/office_v1/csv/Trajectory_step_cfc_pretrain_imitation_only_default12N_sivas_idx_100_v2.csv"

DT = 0.002           # sample period (s)
PRE_ROLL = 0.0       # seconds of zero flat lines at the start
POST_HOLD = 0.0      # seconds to hold the final frame
VIDEO_FPS = 10       # target video fps (we decimate frames to reach this)
DPI = 120            # lower -> faster render
SAVE_PATH = "results/lab_v4/figures/Trajectory_step_cfc_pretrain_imitation_only_default12N_sivas_idx_100_v2.mp4"

FIGSIZE = (11, 12)   # overall figure size

# If you want to force the ffmpeg path manually, set this (or env FFMPEG_PATH)
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "").strip()

# =================== Column mapping ===================
CANON = [
    "Theta (rad)",
    "Alpha (rad)",
    "Theta_dot (rad/s)",
    "Alpha_dot (rad/s)",
    "Reference (rad)",
    "Command (V)",
]
ALIASES = {
    "Theta (rad)": ["theta (rad)", "theta", "theta(rad)", "Theta(rad)"],
    "Alpha (rad)": ["alpha (rad)", "alpha", "alpha(rad)", "Alpha(rad)"],
    "Theta_dot (rad/s)": [
        "theta_dot (rad/s)", "theta_dot", "theta vel (rad/s)",
        "Theta_dot(rad/s)", "Theta_dota (rad/s)", "Theta_dota(rad/s)"
    ],
    "Alpha_dot (rad/s)": [
        "alpha_dot (rad/s)", "alpha_dot", "alpha vel (rad/s)",
        "Alpha_dot(rad/s)", "Alpha_dota (rad/s)", "Alpha_dota(rad/s)"
    ],
    "Reference (rad)": [
        "ref (rad)", "ref", "reference", "reference(rad)",
        "Trajectory (rad)", "trajectory (rad)", "Trajectory(rad)"
    ],
    "Command (V)": ["u (v)", "u", "command(v)", "command", "U (V)"],
}

def _norm(s: str) -> str:
    return (s.strip().lower()
            .replace("_", " ")
            .replace("( ", "(").replace(" )", ")")
            .replace("  ", " "))

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
    return df

def smart_read_csv(path: str):
    """Read CSV from path or path+'.csv', then remap and validate required columns."""
    p = path
    if not os.path.exists(p) and not p.lower().endswith(".csv"):
        if os.path.exists(p + ".csv"):
            p = p + ".csv"
    if not os.path.exists(p):
        raise FileNotFoundError(f"CSV not found: {path} (also tried '{path}.csv')")
    df = pd.read_csv(p)
    df = remap_columns(df)
    missing = [c for c in CANON if c not in df.columns]
    if missing:
        raise ValueError(f"CSV '{p}' missing columns after alias mapping: {missing}\nFound: {list(df.columns)}")
    return df[CANON].copy(), p

def series(df, name):
    return df[name].astype(float).to_numpy()

# =================== FFmpeg finder (Windows-friendly) ===================
def _extend_path_once(p):
    if p and os.path.isdir(p) and p not in os.environ.get("PATH",""):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH","")

# Add common WinGet/WindowsApps link dirs into PATH (no need to reopen shell)
_extend_path_once(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links"))
_extend_path_once(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps"))

def _search_winget_packages():
    base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if not os.path.isdir(base):
        return []
    patterns = [
        os.path.join(base, "*ffmpeg*", "**", "bin", "ffmpeg.exe"),
        os.path.join(base, "*FFmpeg*", "**", "bin", "ffmpeg.exe"),
    ]
    found = []
    for pat in patterns:
        found += glob.glob(pat, recursive=True)
    return found

def get_ffmpeg_writer(fps: int):
    """Try several locations to find ffmpeg and construct an FFMpegWriter."""
    candidates = []
    if FFMPEG_PATH:
        candidates.append(FFMPEG_PATH)
    wh = shutil.which("ffmpeg")
    if wh:
        candidates.append(wh)
    candidates += [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps\ffmpeg.exe"),
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    candidates += _search_winget_packages()

    # unique
    seen, uniq = set(), []
    for p in candidates:
        p = (p or "").strip()
        if p and p.lower() not in seen:
            seen.add(p.lower()); uniq.append(p)

    for p in uniq:
        if os.path.isfile(p):
            mpl.rcParams["animation.ffmpeg_path"] = p
            print(f"[info] Using ffmpeg at: {p}")
            return FFMpegWriter(
                fps=fps,
                metadata={"artist": "LQR vs CfC"},
                bitrate=-1,
                extra_args=["-pix_fmt", "yuv420p", "-preset", "ultrafast"]
            )

    probe = "\n".join(f" - {p}" for p in uniq[:10])
    raise RuntimeError(
        "FFmpeg not found.\n"
        "Try one of the following:\n"
        "  • Open a NEW PowerShell so PATH refreshes and rerun,\n"
        "  • Or set FFMPEG_PATH to the full path to ffmpeg.exe.\n"
        "Searched paths (first 10):\n" + probe
    )

# =================== Load & prep ===================
df_lqr, _ = smart_read_csv(LQR_CSV_PATH)
df_cfc, _ = smart_read_csv(CFC_CSV_PATH)

# Align lengths
N = min(len(df_lqr), len(df_cfc))
df_lqr = df_lqr.iloc[:N].reset_index(drop=True)
df_cfc = df_cfc.iloc[:N].reset_index(drop=True)

# Arrays
th_lqr  = series(df_lqr, "Theta (rad)")
al_lqr  = series(df_lqr, "Alpha (rad)")
u_lqr   = series(df_lqr, "Command (V)")
ref_lqr = series(df_lqr, "Reference (rad)")

th_cfc  = series(df_cfc, "Theta (rad)")
al_cfc  = series(df_cfc, "Alpha (rad)")
u_cfc   = series(df_cfc, "Command (V)")
ref_cfc = series(df_cfc, "Reference (rad)")

# Time vectors
t_data = np.arange(N) * DT
pre_frames  = int(round(PRE_ROLL / DT))
post_frames = int(round(POST_HOLD / DT))
total_frames = pre_frames + N + post_frames
t_total_end = PRE_ROLL + N*DT + POST_HOLD

# Decimate to true VIDEO_FPS
FRAME_STEP = max(1, int(round((1.0/DT) / VIDEO_FPS)))  # samples per video frame

print(f"[info] samples={N}, dt={DT}s, duration={N*DT:.3f}s, "
      f"pre={PRE_ROLL}s, post={POST_HOLD}s, total video ~{t_total_end:.3f}s")
print(f"[info] total simulation frames (1/dt)={total_frames}, FRAME_STEP={FRAME_STEP}, "
      f"actual video frames ≈ {int(np.ceil(total_frames/FRAME_STEP))}")

# Axis limits helpers
def lims(*arrs, pad=0.05):
    v = np.concatenate([np.array(a, dtype=float).ravel() for a in arrs])
    v = v[np.isfinite(v)]
    vmin, vmax = (np.min(v), np.max(v)) if v.size else (0.0, 1.0)
    if vmin == vmax:
        vmin -= 1e-3; vmax += 1e-3
    span = vmax - vmin
    return (vmin - pad*span, vmax + pad*span)

th_min, th_max = lims(th_lqr, th_cfc, ref_lqr, ref_cfc)
al_min, al_max = lims(al_lqr, al_cfc, np.array([0.0]))
u_min,  u_max  = lims(u_lqr, u_cfc)

# =================== Figure & artists ===================
fig, axs = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)

# Theta
line_th_lqr, = axs[0].plot([], [], label="LQR θ", color="red")
line_th_cfc, = axs[0].plot([], [], label="CfC θ", color="blue")
line_th_ref, = axs[0].plot([], [], linestyle="--", label="Ref θ", color="black")
axs[0].set_xlim(0, t_total_end)
axs[0].set_ylim(th_min, th_max)
axs[0].set_title("Theta (rad)")
axs[0].grid(True)
axs[0].legend()

# Alpha
line_al_lqr, = axs[1].plot([], [], label="LQR α", color="red")
line_al_cfc, = axs[1].plot([], [], label="CfC α", color="blue")
axs[1].axhline(0.0, linestyle="--", label="Ref α = 0", color="black")
axs[1].set_xlim(0, t_total_end)
axs[1].set_ylim(al_min, al_max)
axs[1].set_title("Alpha (rad)")
axs[1].grid(True)
axs[1].legend()

# Command
line_u_lqr, = axs[2].plot([], [], label="LQR", color="red")
line_u_cfc, = axs[2].plot([], [], label="CfC", color="blue")
axs[2].set_xlim(0, t_total_end)
axs[2].set_ylim(u_min, u_max)
axs[2].set_title("Command (V)")
axs[2].set_xlabel("Time (s)")
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()

# =================== Update function ===================
def update(frame):
    """
    'frame' is a raw sample index. We step by FRAME_STEP so the video FPS is stable
    even if 1/DT is much higher than VIDEO_FPS.
    """
    if frame < pre_frames:
        # pre-roll zeros
        t = np.arange(frame) * DT
        zeros = np.zeros_like(t)
        line_th_lqr.set_data(t, zeros)
        line_th_cfc.set_data(t, zeros)
        line_th_ref.set_data(t, zeros)
        line_al_lqr.set_data(t, zeros)
        line_al_cfc.set_data(t, zeros)
        line_u_lqr.set_data(t, zeros)
        line_u_cfc.set_data(t, zeros)
    elif frame < pre_frames + N:
        # draw with real data
        k = frame - pre_frames
        t = PRE_ROLL + t_data[:k]
        # theta
        line_th_lqr.set_data(t, th_lqr[:k])
        line_th_cfc.set_data(t, th_cfc[:k])
        # reference (take LQR's; typically identical)
        line_th_ref.set_data(t, ref_lqr[:k])
        # alpha
        line_al_lqr.set_data(t, al_lqr[:k])
        line_al_cfc.set_data(t, al_cfc[:k])
        # command
        line_u_lqr.set_data(t, u_lqr[:k])
        line_u_cfc.set_data(t, u_cfc[:k])
    else:
        # post-hold: frozen final frame
        t = PRE_ROLL + t_data
        line_th_lqr.set_data(t, th_lqr)
        line_th_cfc.set_data(t, th_cfc)
        line_th_ref.set_data(t, ref_lqr)
        line_al_lqr.set_data(t, al_lqr)
        line_al_cfc.set_data(t, al_cfc)
        line_u_lqr.set_data(t, u_lqr)
        line_u_cfc.set_data(t, u_cfc)

    return (line_th_lqr, line_th_cfc, line_th_ref,
            line_al_lqr, line_al_cfc,
            line_u_lqr, line_u_cfc)

# =================== Animate & save ===================
frame_indices = range(0, total_frames, FRAME_STEP)
anim = FuncAnimation(fig, update, frames=frame_indices, interval=1000/VIDEO_FPS, blit=True)

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
writer = get_ffmpeg_writer(VIDEO_FPS)
anim.save(SAVE_PATH, writer=writer, dpi=DPI)
print(f"[ok] MP4 saved to: {SAVE_PATH}")
