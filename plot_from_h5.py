#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a single scenario from an HDF5 dataset ("packed" or "groups" layout).

What it draws
-------------
1) Theta and Theta_desired vs Time
2) Alpha vs Time
3) Command vs Time

Usage (PowerShell examples)
---------------------------
python plot_from_h5.py --h5 step.h5 --idx 20 --outdir plots
python plot_from_h5.py --h5 random.h5 --idx 20 --outdir plots
python plot_from_h5.py --h5 impulse_diff.h5 --idx 20 --outdir plots

Notes
-----
• In "packed" mode, the time vector is synthesized using root attr 'dt'.
• In "groups" mode, the time vector is read from the group's 'time' dataset.
• If 'states_keys' attr exists, use it to resolve theta/alpha indices;
  otherwise assume indices 0:theta, 1:alpha.
"""

import argparse, os, json, h5py, numpy as np
import matplotlib.pyplot as plt

def plot_one(h5_path: str, idx: int, outdir: str = "plots", dpi: int = 150, fmt: str = "png"):
    """Load scenario #idx and export three figures (theta, alpha, command)."""
    os.makedirs(outdir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        if "step" in f:  # packed layout
            g = f["step"]
            N, steps = g["ref"].shape
            if not (0 <= idx < N):
                raise IndexError(f"idx {idx} invalid. Valid range: [0, {N-1}]")
            # time vector
            dt = float(f.attrs.get("dt", 0.0))
            if dt <= 0:
                raise ValueError("H5 root attrs['dt'] missing or invalid.")
            t = np.arange(steps, dtype=np.float64) * dt

            # states_keys
            states_keys = None
            if "states_keys" in f.attrs:
                try:
                    states_keys = json.loads(f.attrs["states_keys"])
                except Exception:
                    states_keys = None

            # resolve theta/alpha indices
            theta_i, alpha_i = 0, 1
            if states_keys:
                if "theta" in states_keys:
                    theta_i = states_keys.index("theta")
                if "alpha" in states_keys:
                    alpha_i = states_keys.index("alpha")

            theta = g["states"][idx, :, theta_i]
            alpha = g["states"][idx, :, alpha_i]
            theta_desired = g["ref"][idx, :]
            command = g["control"][idx, :]

            total_N = N
            title_suffix = f"[packed] idx={idx}/{total_N-1}, dt={dt:.6g}s, steps={steps}"

        elif "scenarios" in f:  # groups layout
            sg = f["scenarios"]
            # list and sort group names
            names = sorted(sg.keys())
            if idx >= len(names) or idx < 0:
                raise IndexError(f"idx {idx} invalid. Valid range: [0, {len(names)-1}]")
            g = sg[names[idx]]

            # time vector
            if "time" not in g:
                raise KeyError("In groups mode, 'time' dataset is required.")
            t = g["time"][:]

            # states_keys
            states_keys = None
            if "states_keys" in g.attrs:
                try:
                    states_keys = json.loads(g.attrs["states_keys"])
                except Exception:
                    states_keys = None

            theta_i, alpha_i = 0, 1
            if states_keys:
                if "theta" in states_keys:
                    theta_i = states_keys.index("theta")
                if "alpha" in states_keys:
                    alpha_i = states_keys.index("alpha")

            theta = g["states"][:, theta_i]
            alpha = g["states"][:, alpha_i]
            # ref or theta_desired
            if "ref" in g:
                theta_desired = g["ref"][:]
            elif "theta_desired" in g:
                theta_desired = g["theta_desired"][:]
            else:
                theta_desired = np.zeros_like(theta)
            command = g["control"][:]

            total_N = len(names)
            dt_est = np.median(np.diff(t)) if len(t) > 1 else float("nan")
            title_suffix = f"[groups] idx={idx}/{total_N-1}, dt≈{dt_est:.6g}s, steps={len(t)}"

        else:
            raise RuntimeError("Invalid H5: expected 'step' (packed) or 'scenarios' (groups).")

    # ---- Plots ----
    base = f"scenario_{idx:06d}"

    # 1) Theta vs time (+ ref)
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, theta, label="Theta")
    plt.plot(t, theta_desired, label="Theta_desired", linestyle="--")
    plt.xlabel("Time [s]"); plt.ylabel("rad")
    plt.title(f"Theta Tracking | {title_suffix}")
    plt.legend(); plt.grid(True, alpha=0.3)
    fp1 = os.path.join(outdir, f"{base}_theta.{fmt}")
    plt.tight_layout(); plt.savefig(fp1, dpi=dpi); plt.close(fig)

    # 2) Alpha vs time
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, alpha, label="Alpha")
    plt.xlabel("Time [s]"); plt.ylabel("rad")
    plt.title(f"Alpha | {title_suffix}")
    plt.legend(); plt.grid(True, alpha=0.3)
    fp2 = os.path.join(outdir, f"{base}_alpha.{fmt}")
    plt.tight_layout(); plt.savefig(fp2, dpi=dpi); plt.close(fig)

    # 3) Command vs time
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, command, label="Command")
    plt.xlabel("Time [s]"); plt.ylabel("V")
    plt.title(f"Command | {title_suffix}")
    plt.legend(); plt.grid(True, alpha=0.3)
    fp3 = os.path.join(outdir, f"{base}_command.{fmt}")
    plt.tight_layout(); plt.savefig(fp3, dpi=dpi); plt.close(fig)

    print("[✓] Kaydedildi:")
    print(" ", fp1)
    print(" ", fp2)
    print(" ", fp3)
    return fp1, fp2, fp3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="HDF5 path (packed or groups)")
    ap.add_argument("--idx", type=int, default=0, help="Scenario index (0-based)")
    ap.add_argument("--outdir", default="plots", help="Output folder")
    ap.add_argument("--dpi", type=int, default=150, help="DPI")
    ap.add_argument("--fmt", default="png", help="png/jpg/pdf/svg")
    args = ap.parse_args()
    plot_one(args.h5, args.idx, args.outdir, args.dpi, args.fmt)

if __name__ == "__main__":
    main()
