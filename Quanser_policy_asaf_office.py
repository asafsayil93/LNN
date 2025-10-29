#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quanser RFL — Real-time CfC control at ~500 Hz (PyTorch checkpoint)

I/O conventions
---------------
Inputs to CfC (exact order):
  [theta, ref, alpha, e_theta, theta_dot, alpha_dot, ref_dot]
Model output:
  tanh in [-1, 1] → scaled by 10 → Volts, clamped to [-10, +10].

What this script provides
-------------------------
• High-frequency control loop with sleep+spin scheduling to approach 500 Hz.
• CfC policy loaded from a Torch checkpoint + training normalization stats.
• Multiple reference generators (step, triangle, ramp, sinus) + H5 dataset ref loader.
• CSV logging compatible with your plotting utilities.

Safety
------
This drives real hardware. Keep clamps, speed checks, and have an E-stop within reach.
"""

from array import array
from quanser.hardware import HIL, Clock

import os, time, math, csv, atexit, ctypes, warnings
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter, sleep

import torch
# If the training used dt-aware models, import 'model_cfc_ncp_dt'; otherwise, this one:
from model_cfc_ncp import CfCNCPWrapper  # tanh output; CfC internal step (timespans=None)

# --------------------------- Paths ---------------------------
CKPT_PATH     = os.path.normpath("runs/cfc_pretrain_imitation_only_default24N/best_ckpt.pth")
NORM_NPZ_PATH = os.path.normpath("runs/cfc_pretrain_imitation_only_default24N/norm_stats.npz")

# --------------------------- Hardware / Scaling ---------------------------
ENCODER_STEP_TO_RAD = 0.0015   # encoder ticks → rad
GAUGE_TO_RAD        = 0.0606   # strain gauge → rad (per your prior calibration)

sampling_freq = 500                 # Hz
dt_nominal    = 1.0 / sampling_freq

# --------------------------- Trajectory helpers ---------------------------
def get_trajectory(duration, value):
    """
    Piecewise-constant reference builder (step list).
    duration: list of segment durations [s]
    value   : list of segment values [rad]
    Returns 1D np.ndarray, length ≈ sum(duration) * sampling_freq.
    """
    global sampling_freq
    trajectory = np.empty((1, 0))
    for i in range(len(value)):
        tmp = int(duration[i] * sampling_freq)
        vec = np.array([value[i]] * tmp).reshape(1, tmp)
        trajectory = np.append(trajectory, vec, axis=1)
    return trajectory.flatten()

def get_trajectory_triangle(durations, values, y0=None):
    """
    Piecewise linear ramps (triangle-like).
    durations: list of segment durations [s]
    values   : list of segment target values [rad]
    y0: optional starting value; if None, starts from values[0].
    """
    global sampling_freq
    values = [float(v) for v in values]
    y_prev = float(values[0] if y0 is None else y0)
    traj = np.empty((0,), dtype=float)
    for i, dur in enumerate(durations):
        y_next = float(values[i])
        n = max(1, int(round(dur * sampling_freq)))
        seg = np.linspace(y_prev, y_next, n, dtype=float)
        if i < len(durations) - 1 and n > 1:
            seg = seg[:-1]  # avoid duplicate endpoint between segments
        traj = np.concatenate([traj, seg])
        y_prev = y_next
    return traj

def get_trajectory_ramp(waypoints, ramp_s=3.0, hold_s=3.0, y0=0.0, include_last_hold=False):
    """
    Ramp-and-hold sequence: y0 → waypoints[0] → ... with a ramp of 'ramp_s' sec,
    then hold 'hold_s' sec. Optionally hold at the final waypoint.
    """
    global sampling_freq
    fs = sampling_freq
    traj = np.empty((0,), dtype=float)
    current = float(y0)
    W = [float(w) for w in waypoints]
    for i, w in enumerate(W):
        n_ramp = max(1, int(round(ramp_s * fs)))
        seg_ramp = np.linspace(current, w, n_ramp, dtype=float)
        if n_ramp > 1:
            seg_ramp = seg_ramp[:-1]
        traj = np.concatenate([traj, seg_ramp])
        is_last = (i == len(W) - 1)
        if (not is_last) or (is_last and include_last_hold):
            n_hold = max(1, int(round(hold_s * fs)))
            seg_hold = np.full(n_hold, w, dtype=float)
            if not is_last and n_hold > 1:
                seg_hold = seg_hold[:-1]
            traj = np.concatenate([traj, seg_hold])
        current = w
    return traj

def get_trajectory_sinus(durations, amps, periods, offsets=0.0, phases=0.0):
    """
    Piecewise sinusoidal reference; supports scalars or per-segment lists.
    Units: amps/offsets in rad, periods in sec, phases in rad.
    """
    global sampling_freq
    def expand(x, L):
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) != L:
                raise ValueError("All lists must match len(durations).")
            return [float(v) for v in x]
        return [float(x)] * L

    L = len(durations)
    amps    = expand(amps,    L)
    periods = expand(periods, L)
    offsets = expand(offsets, L)
    phases  = expand(phases,  L)

    traj = np.empty((0,), dtype=float)
    for i, dur in enumerate(durations):
        n = max(1, int(round(dur * sampling_freq)))
        t = np.arange(n, dtype=float) / sampling_freq
        seg = offsets[i] + amps[i] * np.sin(2.0 * np.pi * t / periods[i] + phases[i])
        if i < L - 1 and n > 1:
            seg = seg[:-1]
        traj = np.concatenate([traj, seg])
    return traj

# ---- Dataset-backed reference (reads /step/ref or reasonable fallbacks) ----
import h5py
def get_trajectory_dataset(idx: int, h5_path: str, key_hint: str | None = None) -> np.ndarray:
    """
    Return theta reference for scenario 'idx' from an H5 file as a 1D array in radians.
    Prefers /step/ref (N, T). If a time vector exists (/step/{t,time,...}), it is used
    to resample to the current 'sampling_freq'.
    """
    global sampling_freq

    def _to_1d_rad(a: np.ndarray, name: str, unit: str | None):
        a = np.asarray(a).reshape(-1).astype(np.float64)
        if unit and "deg" in str(unit).lower():
            a = np.deg2rad(a)
        elif "(deg" in name.lower() or (np.nanmax(np.abs(a)) > (np.pi * 1.5)):
            a = np.deg2rad(a)
        return a

    def _resample(arr: np.ndarray, tvec: np.ndarray | None):
        fs_out = float(sampling_freq)
        if tvec is not None:
            t = np.asarray(tvec, dtype=np.float64).reshape(-1)
            t = t - t[0]
            T = float(t[-1])
            t_out = np.arange(0.0, T + 1e-12, 1.0 / fs_out)
            return np.interp(t_out, t, arr).astype(np.float64)
        return arr.astype(np.float64)

    with h5py.File(h5_path, "r") as f:
        # Fast path: /step/ref
        if "step" in f and "ref" in f["step"]:
            ref_ds = f["step"]["ref"]
            ref_row = ref_ds[int(idx), :] if ref_ds.ndim == 2 else ref_ds[...]
            tvec = None
            for tk in ["t", "time", "Time (s)", "timestamps"]:
                if tk in f["step"]:
                    tds = f["step"][tk]
                    tvec = tds[int(idx), :] if (tds.ndim == 2) else tds[...]
                    break
            ref_1d = _to_1d_rad(ref_row, "/step/ref", ref_ds.attrs.get("unit", None))
            out = _resample(ref_1d, tvec)
            if np.any(np.isnan(out)):
                out = out[~np.isnan(out)]
                warnings.warn("NaN values removed from trajectory.")
            print(f"[H5] /step/ref[{idx}] -> len={len(out)} (time vec: {'yes' if tvec is not None else 'no'})")
            return out

        # Generic fallback: search groups for a plausible 'ref'
        def _search_any_ref(g: h5py.Group):
            for cand in [key_hint, "ref", "reference", "theta_ref", "Reference (rad)", "y_ref", "target"]:
                if cand and cand in g and isinstance(g[cand], h5py.Dataset):
                    ds = g[cand]; data = ds[...]
                    if ds.ndim == 2:
                        data = data[int(idx), :] if int(idx) < data.shape[0] else data[:, int(idx)]
                    return _to_1d_rad(data, ds.name, ds.attrs.get("unit", None))
            for k, v in g.items():
                if isinstance(v, h5py.Group):
                    r = _search_any_ref(v)
                    if r is not None: return r
            return None

        for p in [f"/episodes/{idx}", f"/traj/{idx}", f"/scenarios/{idx}", f"/runs/{idx}", f"/rollouts/{idx}", f"/{idx}"]:
            if p in f:
                arr = _search_any_ref(f[p])
                if arr is not None:
                    out = _resample(arr, None)
                    if np.any(np.isnan(out)):
                        out = out[~np.isnan(out)]
                        warnings.warn("NaN values removed from trajectory.")
                    print(f"[H5] {p} -> len={len(out)}")
                    return out

        arr = _search_any_ref(f)
        if arr is not None:
            out = _resample(arr, None)
            if np.any(np.isnan(out)):
                out = out[~np.isnan(out)]
                warnings.warn("NaN values removed from trajectory.")
            print(f"[H5] /ref-like dataset -> len={len(out)}")
            return out

        # If nothing matched, dump tree for debugging
        def _tree(gr, prefix=""):
            lines=[]
            for k, v in gr.items():
                path = f"{prefix}/{k}" if prefix else f"/{k}"
                if isinstance(v, h5py.Group):
                    lines.append(path + "/")
                    lines.extend(_tree(v, path))
                else:
                    lines.append(f"{path} shape={v.shape}")
            return lines
        raise RuntimeError("Reference not found; H5 structure:\n" + "\n".join(_tree(f)))

# --------------------------- CfC Loader ---------------------------
def load_cfc_and_norm(ckpt_path, norm_npz_path, device="cpu"):
    """
    Load CfC checkpoint and normalization stats.
    Returns (model, mean[7], std[7]).
    """
    z = np.load(norm_npz_path)
    mean = z["mean"].astype(np.float32).copy()
    std  = np.maximum(z["std"].astype(np.float32).copy(), 1e-6)

    # Torch version robust loading
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception:
        import torch.serialization as ts, numpy as _np
        ts.add_safe_globals([_np.core.multiarray._reconstruct])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    cfg = ckpt.get("config", {})
    model = CfCNCPWrapper(
        in_dim=7, out_dim=1,
        units=int(cfg.get("units", 24)),
        sparsity_level=float(cfg.get("sparsity_level", 0.5)),
        mode=str(cfg.get("mode", "default")),
        mixed_memory=bool(cfg.get("mixed_memory", True)),
        seed=42
    ).to(device)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, mean, std

# --------------------------- Utilities ---------------------------
def clamp(x, lo, hi): return lo if x < lo else (hi if x > hi else x)

# --------------------------- Realtime controller ---------------------------
def control_loop():
    """
    One control tick (~2 ms):
      • read sensors, compute derivatives;
      • index reference & ref_dot from prebuilt trajectory;
      • build CfC input vector, run model, write DAC;
      • append logs; print loop rate every ~0.4 s.
    """
    global counter, dt_real, prev_theta, prev_alpha, prev_ref, hstate
    global command_storage, alpha_storage, theta_storage, alpha_dot_storage, theta_dot_storage, e_theta_storage
    global tic, t0, last_print

    counter += 1
    card.read_encoder(encoder_channels, num_encoder_channels, encoder_buffer)
    card.read_analog(analog_channels, num_analog_channels, buffer_analog)

    alpha_rad = buffer_analog[0] * GAUGE_TO_RAD
    theta_rad = encoder_buffer[0] * ENCODER_STEP_TO_RAD

    t_now   = perf_counter()
    dt_real = max(t_now - tic, 1e-4)
    tic     = t_now

    theta_dot = (theta_rad - prev_theta) / dt_real
    alpha_dot = (alpha_rad - prev_alpha) / dt_real
    prev_theta, prev_alpha = theta_rad, alpha_rad

    # time-indexed reference
    idx = int((t_now - t0) * sampling_freq)
    if idx >= len(trajectory): idx = len(trajectory) - 1
    ref = float(trajectory[idx])
    ref_dot = (ref - prev_ref) / dt_real
    prev_ref = ref

    e_theta = theta_rad - ref

    # hard speed safety (cut output if too fast)
    if abs(theta_dot) > np.deg2rad(180.0):
        u_volts = 0.0
    else:
        x = np.array([theta_rad, ref, alpha_rad, e_theta, theta_dot, alpha_dot, ref_dot], dtype=np.float32)
        x_norm = (x - norm_mean) / norm_std
        xt = torch.from_numpy(x_norm).view(1, 1, 7).to(device)
        with torch.no_grad():
            y_scaled, h_next = cfc_model(xt, dt=dt_real, h=hstate)
        hstate = h_next
        u_volts = clamp(float(y_scaled.squeeze().cpu().numpy()) * 10.0, -10.0, 10.0)

    buffer_out[0] = u_volts
    card.write_analog(channels, num_channels, buffer_out)

    command_storage.append(u_volts)
    alpha_storage.append(alpha_rad)
    theta_storage.append(theta_rad)
    alpha_dot_storage.append(alpha_dot)
    theta_dot_storage.append(theta_dot)
    e_theta_storage.append(e_theta)

    if counter % 200 == 0:
        hz = 200.0 / max(t_now - last_print, 1e-6)
        print(f"loop ≈ {hz:.1f} Hz")
        last_print = t_now

def run_realtime(duration_s, dt):
    """Fixed-rate loop: sleep + short spin to reduce jitter."""
    next_t = perf_counter()
    end_t  = next_t + duration_s
    while True:
        now = perf_counter()
        if now >= end_t:
            break
        control_loop()
        next_t += dt
        delay = next_t - perf_counter()
        if delay > 0.0015:
            sleep(delay - 0.0005)
        while perf_counter() < next_t:
            pass  # short spin

# --------------------------- Main ---------------------------
if __name__ == "__main__":
    # CPU only; reduce overhead
    device = torch.device("cpu")
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Improve Windows timer resolution
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        atexit.register(lambda: ctypes.windll.winmm.timeEndPeriod(1))
        print("Timer resolution set to 1 ms")
    except Exception as e:
        print("timeBeginPeriod failed:", e)

    # Optional: raise process priority
    try:
        import win32api, win32process
        p = win32api.GetCurrentProcess()
        win32process.SetPriorityClass(p, win32process.HIGH_PRIORITY_CLASS)
    except Exception:
        pass

    # Load CfC and normalization
    cfc_model, norm_mean, norm_std = load_cfc_and_norm(CKPT_PATH, NORM_NPZ_PATH, device=device)

    # Quanser init
    card = HIL()
    card.open("q2_usb", "0")

    encoder_channels     = array('I', [0])
    num_encoder_channels = len(encoder_channels)
    analog_channels      = array('I', [1])
    num_analog_channels  = len(analog_channels)
    channels             = array('I', [0])   # DAC out
    num_channels         = len(channels)

    encoder_buffer = array('i', [0] * num_encoder_channels)
    buffer_analog  = array('d', [0.0] * num_analog_channels)
    buffer_out     = array('d', [0.0])

    # ===== Reference selection =====
    # Example 1: explicit step list (rad)
    value    = [ 0.0, 0.28368124, -0.28368124, 0.35432990, -0.35432990,
                 0.2, -0.4, 0.4, 0.0, 0.28368124, -0.28368124, 0.35432990, -0.35432990,
                 0.2, -0.4, 0.4, 0.0, 0.28368124, -0.28368124, 0.35432990, -0.35432990,
                 0.2, -0.4, 0.4 ]
    duration = [3]*len(value)   # seconds per step
    trajectory = get_trajectory(duration, value)

    # Example 2: from dataset (uncomment to use)
    # DATASET_PATH = os.path.normpath("combined_all_steps.h5")
    # SCEN_IDX = 1
    # trajectory = get_trajectory_dataset(SCEN_IDX, DATASET_PATH)

    t_stop = len(trajectory) / sampling_freq

    # Logs
    command_storage, alpha_storage, theta_storage = [], [], []
    alpha_dot_storage, theta_dot_storage, e_theta_storage = [], [], []

    # Realtime state
    prev_theta = 0.0
    prev_alpha = 0.0
    prev_ref   = trajectory[0] if len(trajectory) > 0 else 0.0
    hstate     = None
    counter    = -1
    dt_real    = dt_nominal
    t0  = perf_counter()
    tic = t0
    last_print = t0

    # Run
    print("Experiment running")
    run_realtime(duration_s=t_stop, dt=dt_nominal)
    print("Experiment stopped")

    # Stop motor, close card
    buffer_out[0] = 0.0
    card.write_analog(channels, num_channels, buffer_out)
    card.close()

    # Save CSV (headers compatible with plotting scripts)
    os.makedirs("results/refs_step", exist_ok=True)
    filename = "results/refs_step/cfc_pretrain_imitation_only_default24N.csv"
    time_axis = np.arange(0, t_stop, dt_nominal)

    data = list(zip(theta_storage, alpha_storage, theta_dot_storage, alpha_dot_storage,
                    trajectory[:len(theta_storage)], command_storage))
    with open(filename, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Theta (rad)", "Alpha (rad)", "Theta_dot (rad/s)", "Alpha_dot (rad/s)",
                    "Reference (rad)", "Command (V)"])
        w.writerows(data)
    print(f"[OK] Saved CSV -> {filename}")

    # Quick plots
    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    tlen = len(theta_storage)
    tvec = time_axis[:tlen]
    axs[0].plot(tvec, theta_storage, label="CfC θ");            axs[0].set_title("Theta (rad)");       axs[0].grid(True)
    axs[1].plot(tvec, alpha_storage);                            axs[1].set_title("Alpha (rad)");       axs[1].grid(True)
    axs[2].plot(tvec, theta_dot_storage);                        axs[2].set_title("Theta_dot (rad/s)"); axs[2].grid(True)
    axs[3].plot(tvec, alpha_dot_storage);                        axs[3].set_title("Alpha_dot (rad/s)"); axs[3].grid(True)
    axs[4].plot(tvec, e_theta_storage);                          axs[4].set_title("e_theta = theta - ref (rad)"); axs[4].grid(True)
    axs[5].plot(tvec, command_storage);                          axs[5].set_title("Command (V)"); axs[5].set_xlabel("Time (s)"); axs[5].grid(True)
    plt.tight_layout()
    plt.show()
