#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quanser RFL — Real-time CfC or LQR control at 500 Hz

I/O conventions
---------------
CfC inputs (exact order):
  [theta, ref, alpha, e_theta, theta_dot, alpha_dot, ref_dot]
Output:
  Volts clamped to [-10, +10].

Features
--------
• Toggle between LQR and CfC with CONTROLLER.
• EMA smoothing for derivatives (LQR) to stabilize estimates.
• Multiple reference generators + H5 dataset reference reader.
• Sleep+spin scheduler to reduce jitter; Windows 1 ms timer resolution.

Safety
------
Hardware test: keep clamps, add your own interlocks/E-stop as needed.
"""

from array import array
from quanser.hardware import HIL, Clock

import os, time, math, csv, ctypes, atexit, warnings
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter, sleep

# ====== Controller selection ======
CONTROLLER = "LQR"            # "CfC" or "LQR"
LQR_GAIN   = np.array([11.8303, -30.4544, 1.4627, -0.6952], dtype=float)  # [e_theta, -alpha, -theta_dot, -alpha_dot]
LQR_SIGN   = -1.0             # some rigs need a global sign flip; try +1.0 or -1.0

# ====== Paths (CfC) ======
CKPT_PATH     = os.path.normpath("runs/sivas/cfc_pretrain_1step_robust_office/best_ckpt.pth")
NORM_NPZ_PATH = os.path.normpath("runs/sivas/cfc_pretrain_1step_robust_office/norm_stats.npz")
DATASET_PATH  = os.path.normpath("combined_all_steps.h5")

# ====== Hardware / scaling ======
ENCODER_STEP_TO_RAD = 0.0015
GAUGE_TO_RAD        = 0.0606

sampling_freq = 500
dt_nominal    = 1.0 / sampling_freq

# ====== Trajectory helpers ======
def get_trajectory(duration, value):
    """Piecewise-constant step sequence from (duration[s], value[rad]) lists."""
    global sampling_freq
    trajectory = np.empty((1, 0))
    for i in range(len(value)):
        tmp = int(duration[i] * sampling_freq)
        vec = np.array([value[i]] * tmp).reshape(1, tmp)
        trajectory = np.append(trajectory, vec, axis=1)
    return trajectory.flatten()

def get_trajectory_triangle(durations, values, y0=None):
    """Piecewise linear ramps (triangle-like)."""
    global sampling_freq
    values = [float(v) for v in values]
    y_prev = float(values[0] if y0 is None else y0)
    traj = np.empty((0,), dtype=float)
    for i, dur in enumerate(durations):
        y_next = float(values[i])
        n = max(1, int(round(dur * sampling_freq)))
        seg = np.linspace(y_prev, y_next, n, dtype=float)
        if i < len(durations) - 1 and n > 1:
            seg = seg[:-1]
        traj = np.concatenate([traj, seg])
        y_prev = y_next
    return traj

def get_trajectory_ramp(waypoints, ramp_s=3.0, hold_s=3.0, y0=0.0, include_last_hold=False):
    """Ramp-and-hold sequence; see docstring in the 'office' variant for details."""
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
    """Piecewise sinusoidal reference (amps/offsets in rad, periods in sec, phases in rad)."""
    global sampling_freq
    def expand(x, L):
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) != L:
                raise ValueError("List lengths must match durations.")
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

# ====== CfC loader ======
import torch
from model_cfc_ncp import CfCNCPWrapper

def load_cfc_and_norm(ckpt_path, norm_npz_path, device="cpu"):
    """Load CfC checkpoint + normalization stats, return (model, mean, std)."""
    z = np.load(norm_npz_path)
    mean = z["mean"].astype(np.float32).copy()
    std  = np.maximum(z["std"].astype(np.float32).copy(), 1e-6)
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
        units=int(cfg.get("units", 48)),
        sparsity_level=float(cfg.get("sparsity_level", 0.5)),
        mode=str(cfg.get("mode", "default")),
        mixed_memory=bool(cfg.get("mixed_memory", True)),
        seed=42
    ).to(device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, mean, std

# ---- H5 dataset reference loader (same capabilities as 'office' variant) ----
import h5py
def get_trajectory_dataset(idx: int, h5_path: str, key_hint: str | None = None) -> np.ndarray:
    """
    Return theta reference for scenario 'idx' from an H5 file as a 1D array in radians.
    Prefers /step/ref (N, T). Uses available time vector for resampling to sampling_freq.
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
            t_out = np.arange(0.0, T + 1e-12, 1.0/fs_out)
            return np.interp(t_out, t, arr).astype(np.float64)
        return arr.astype(np.float64)

    with h5py.File(h5_path, "r") as f:
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

# ====== Utilities ======
def clamp(x, lo, hi): return lo if x < lo else (hi if x > hi else x)

# Simple exponential smoother for derivatives (used by LQR at 500 Hz)
EMA_TAU = 0.015  # seconds (~15 ms)
def ema_update(prev, new, dt):
    a = math.exp(-dt / max(EMA_TAU, 1e-6))
    return a*prev + (1.0 - a)*new

# ====== LQR policy ======
def run_LQR(theta, alpha, theta_dot_ema, alpha_dot_ema, reference):
    """
    Full-state feedback on error vector: [e_theta, -alpha, -theta_dot, -alpha_dot]
    u = LQR_SIGN * dot(LQR_GAIN, err_vec), clamped to ±10 V.
    """
    e_theta = reference - theta
    err_vec = np.array([e_theta, -alpha, -theta_dot_ema, -alpha_dot_ema], dtype=float)
    u = float(LQR_SIGN * np.dot(LQR_GAIN, err_vec))
    return clamp(u, -10.0, 10.0)

# ====== Realtime loop ======
def control_loop():
    """
    One 500 Hz tick:
      • read sensors → derivatives (raw + EMA),
      • index ref & ref_dot from trajectory,
      • run LQR or CfC,
      • write DAC, log, and occasionally print loop rate.
    """
    global counter, dt_real, prev_theta, prev_alpha, prev_ref, hstate
    global ema_theta_dot, ema_alpha_dot
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

    # raw derivatives
    theta_dot_raw = (theta_rad - prev_theta) / dt_real
    alpha_dot_raw = (alpha_rad - prev_alpha) / dt_real
    prev_theta, prev_alpha = theta_rad, alpha_rad

    # smoothed (for LQR only)
    ema_theta_dot = ema_update(ema_theta_dot, theta_dot_raw, dt_real)
    ema_alpha_dot = ema_update(ema_alpha_dot, alpha_dot_raw, dt_real)

    # time-indexed reference
    idx = int((t_now - t0) * sampling_freq)
    if idx >= len(trajectory): idx = len(trajectory) - 1
    ref = float(trajectory[idx])
    ref_dot = (ref - prev_ref) / dt_real
    prev_ref = ref

    e_theta = theta_rad - ref

    # safety: cut output if crazy speed
    if abs(theta_dot_raw) > np.deg2rad(180.0):
        u_volts = 0.0
    else:
        if CONTROLLER.upper() == "LQR":
            u_volts = run_LQR(theta_rad, alpha_rad, ema_theta_dot, ema_alpha_dot, ref)
        else:  # CfC
            x = np.array([theta_rad, ref, alpha_rad, e_theta, theta_dot_raw, alpha_dot_raw, ref_dot], dtype=np.float32)
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
    alpha_dot_storage.append(alpha_dot_raw)
    theta_dot_storage.append(theta_dot_raw)
    e_theta_storage.append(e_theta)

    if counter % 200 == 0:
        hz = 200.0 / max(t_now - last_print, 1e-6)
        print(f"loop ≈ {hz:.1f} Hz")
        last_print = t_now

def run_realtime(duration_s, dt):
    """Fixed-rate loop with sleep+spin scheduling to reduce jitter."""
    next_t = perf_counter()
    end_t  = next_t + duration_s
    while True:
        now = perf_counter()
        if now >= end_t: break
        control_loop()
        next_t += dt
        delay = next_t - perf_counter()
        if delay > 0.0015:
            sleep(delay - 0.0005)
        while perf_counter() < next_t:
            pass

# ====== Main ======
if __name__ == "__main__":
    # CPU only (lower overhead)
    device = torch.device("cpu")
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # 1 ms timer resolution (Windows)
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

    # Load CfC only if needed
    if CONTROLLER.upper() == "CFC":
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

    # ==== Trajectory (choose one) ====
    # 1) Explicit steps (rad):
    value    = [ 0.0, 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                 0.2, -0.4, 0.4, 0.0, 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                 0.2, -0.4, 0.4, 0.0, 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                 0.2, -0.4, 0.4 ]
    duration = [3]*len(value)
    trajectory = get_trajectory(duration, value)

    # 2) From dataset (uncomment):
    # SCEN_IDX = 250
    # trajectory = get_trajectory_dataset(SCEN_IDX, DATASET_PATH)

    # 3) MATLAB-style square-wave builder (example, keep commented if not needed):
    # dt_ref      = dt_nominal
    # durations   = [4]
    # periods     = [4]
    # A_deg_list  = list(range(30, 31))
    # trajectory  = build_ref_sequence(kind="step", A_deg_list=A_deg_list, dt=dt_ref, durations=durations, periods=periods)

    t_stop = len(trajectory) / sampling_freq

    # ==== Logs ====
    command_storage, alpha_storage, theta_storage = [], [], []
    alpha_dot_storage, theta_dot_storage, e_theta_storage = [], [], []

    # ==== Realtime state ====
    prev_theta = 0.0
    prev_alpha = 0.0
    prev_ref   = trajectory[0] if len(trajectory) > 0 else 0.0
    hstate     = None
    counter    = -1
    dt_real    = dt_nominal
    t0  = perf_counter()
    tic = t0
    last_print = t0

    # EMA init (used by LQR)
    ema_theta_dot = 0.0
    ema_alpha_dot = 0.0

    # ==== Run ====
    print(f"Experiment running [{CONTROLLER}]")
    run_realtime(duration_s=t_stop, dt=dt_nominal)
    print("Experiment stopped")

    # Stop motor and close card
    buffer_out[0] = 0.0
    card.write_analog(channels, num_channels, buffer_out)
    card.close()

    # ==== Save CSV ====
    os.makedirs("results/office_v1/csv", exist_ok=True)
    fname = f"results/office_v1/csv/Trajectory_step_{CONTROLLER.lower()}_idx_100_last_exp.csv"
    time_axis = np.arange(0, t_stop, dt_nominal)
    data = list(zip(theta_storage, alpha_storage, theta_dot_storage, alpha_dot_storage,
                    trajectory[:len(theta_storage)], command_storage))
    with open(fname, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Theta (rad)", "Alpha (rad)", "Theta_dot (rad/s)", "Alpha_dot (rad/s)",
                    "Reference (rad)", "Command (V)"])
        w.writerows(data)
    print(f"Saved CSV -> {fname}")

    # ==== Quick plots ====
    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    tlen = len(theta_storage)
    tvec = time_axis[:tlen]
    axs[0].plot(tvec, theta_storage, label=f"{CONTROLLER} θ"); axs[0].set_title("Theta (rad)");       axs[0].grid(True)
    axs[1].plot(tvec, alpha_storage);                           axs[1].set_title("Alpha (rad)");       axs[1].grid(True)
    axs[2].plot(tvec, theta_dot_storage);                       axs[2].set_title("Theta_dot (rad/s)"); axs[2].grid(True)
    axs[3].plot(tvec, alpha_dot_storage);                       axs[3].set_title("Alpha_dot (rad/s)"); axs[3].grid(True)
    axs[4].plot(tvec, e_theta_storage);                         axs[4].set_title("e_theta = theta - ref (rad)"); axs[4].grid(True)
    axs[5].plot(tvec, command_storage);                         axs[5].set_title("Command (V)"); axs[5].set_xlabel("Time (s)"); axs[5].grid(True)
    plt.tight_layout()
    plt.show()
