#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quanser RFL — Real-time CfC or LQR control at 500 Hz.

I/O conventions
---------------
Inputs to CfC (in this exact order):
  [theta, ref, alpha, e_theta, theta_dot, alpha_dot, ref_dot]
Output command:
  Volts, clamped to [-10, +10].

What this script provides
-------------------------
• A high-frequency (500 Hz) control loop with jitter mitigation (sleep+spin).
• LQR or CfC policy selection via CONTROLLER.
• Step-reference generator helpers, including a MATLAB-like square-wave builder
  implemented using the provided get_trajectory(duration, value).
• Live EMA smoothing for derivatives to stabilize LQR estimates.
• Automatic Windows timer resolution & (optional) process priority bump.

Safety note
-----------
This is real hardware. Keep command clamps, speed checks, and E-stop access!
"""

from array import array
from quanser.hardware import HIL, Clock

import os, time, math, csv, ctypes, atexit
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

# -----------------------------------------------------------------------------
# TRAJECTORY HELPERS
# -----------------------------------------------------------------------------
def get_trajectory(duration, value):
    """
    Piecewise-constant trajectory builder (provided by you).
    duration: list of segment durations in seconds
    value   : list of segment values (same length)
    Returns a 1D numpy vector of length sum(duration)*Fs at global 'sampling_freq'.
    """
    global sampling_freq
    trajectory = np.empty((1, 0))
    for i in range(len(value)):
        tmp = int(duration[i] * sampling_freq)
        vec = np.array([value[i]] * tmp).reshape(1, tmp)
        trajectory = np.append(trajectory, vec ,axis = 1)
    return trajectory.flatten()

def build_step_segment_with_get_trajectory(A_rad: float, seg_duration_s: float, period_s: float) -> np.ndarray:
    """
    Build a 50% duty square wave for one segment: +A for P/2, then -A for P/2,
    repeated until seg_duration_s is covered. Uses get_trajectory(duration, value).
    """
    fs = sampling_freq
    Ns_target = int(round(seg_duration_s * fs))
    half_samp = max(1, int(round((period_s / 2.0) * fs)))

    durations_s = []
    values = []

    sign = 1.0
    n_acc = 0
    while n_acc < Ns_target:
        n_this = min(half_samp, Ns_target - n_acc)
        durations_s.append(n_this / fs)      # seconds
        values.append(sign * A_rad)          # +A or -A
        n_acc += n_this
        sign *= -1.0

    return get_trajectory(durations_s, values)

def build_step_all_with_get_trajectory(A_deg_list, durations_seg, periods):
    """
    MATLAB spec (example):
      dt=0.002; durations=[60 60 60 60 60]; periods=[12 10 8 6 4]; A_deg=10:30; kind='step'
    For each amplitude A (deg) and each (duration, period) pair, generate square-wave segments
    and concatenate across all A values. Returns a 1D numpy array in radians.
    """
    traj_parts = []
    for A_deg in A_deg_list:
        A_rad = float(np.deg2rad(A_deg))
        for seg_dur, P in zip(durations_seg, periods):
            seg = build_step_segment_with_get_trajectory(A_rad, float(seg_dur), float(P))
            traj_parts.append(seg)
    return np.concatenate(traj_parts, axis=0) if traj_parts else np.zeros(0, dtype=float)

# ====== CfC loader (unchanged) ======
import torch
from model_cfc_ncp import CfCNCPWrapper

def load_cfc_and_norm(ckpt_path, norm_npz_path, device="cpu"):
    """Load CfC checkpoint and normalization stats, return (model, mean, std)."""
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
        units=int(cfg.get("units",48)),
        sparsity_level=float(cfg.get("sparsity_level", 0.5)),
        mode=str(cfg.get("mode", "default")),
        mixed_memory=bool(cfg.get("mixed_memory", True)),
        seed=42
    ).to(device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, mean, std

# ====== Utilities ======
def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

# Simple exponential smoother for derivatives @ 500 Hz
EMA_TAU = 0.015  # seconds ~15 ms
def ema_update(prev, new, dt):
    a = math.exp(-dt / max(EMA_TAU, 1e-6))
    return a*prev + (1.0 - a)*new

# ====== LQR policy ======
def run_LQR(theta, alpha, theta_dot_ema, alpha_dot_ema, reference):
    """
    Full-state feedback:
      err_vec = [e_theta, -alpha, -theta_dot, -alpha_dot]
      u = LQR_SIGN * dot(LQR_GAIN, err_vec)
    """
    e_theta = reference - theta
    err_vec = np.array([e_theta, -alpha, -theta_dot_ema, -alpha_dot_ema], dtype=float)
    u = float(LQR_SIGN * np.dot(LQR_GAIN, err_vec))
    return clamp(u, -10.0, 10.0)

# ====== Realtime loop ======
def control_loop():
    """
    Single 500 Hz iteration:
      • read sensors (encoder+gauge), compute derivatives and EMA,
      • compute ref & ref_dot from time,
      • run LQR or CfC,
      • write DAC output and log series.
    """
    global counter, dt_real, prev_theta, prev_alpha, prev_ref, hstate
    global ema_theta_dot, ema_alpha_dot
    global command_storage, alpha_storage, theta_storage, alpha_dot_storage, theta_dot_storage, e_theta_storage
    global time_storage
    global tic, t0, last_print

    counter += 1
    card.read_encoder(encoder_channels, num_encoder_channels, encoder_buffer)
    card.read_analog(analog_channels, num_analog_channels, buffer_analog)

    alpha_rad = buffer_analog[0] * GAUGE_TO_RAD
    theta_rad = encoder_buffer[0] * ENCODER_STEP_TO_RAD

    t_now   = perf_counter()
    dt_real = max(t_now - tic, 1e-4)
    tic     = t_now
    time_storage.append(t_now - t0)

    # raw derivatives
    theta_dot_raw = (theta_rad - prev_theta) / dt_real
    alpha_dot_raw = (alpha_rad - prev_alpha) / dt_real
    prev_theta = theta_rad
    prev_alpha = alpha_rad

    # smoothed (for LQR)
    ema_theta_dot = ema_update(ema_theta_dot, theta_dot_raw, dt_real)
    ema_alpha_dot = ema_update(ema_alpha_dot, alpha_dot_raw, dt_real)

    # time-indexed reference
    idx = int((t_now - t0) * sampling_freq)
    if idx >= len(trajectory): idx = len(trajectory) - 1
    ref = float(trajectory[idx])
    ref_dot = (ref - prev_ref) / dt_real
    prev_ref = ref

    e_theta = theta_rad - ref

    # safety: cut output if angular speed is excessive
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
            u_volts = float(y_scaled.squeeze().cpu().numpy()) * 10.0
            u_volts = clamp(u_volts, -10.0, 10.0)

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
    """Fixed-rate loop: sleep+spin scheduling to approach 500 Hz on Windows."""
    next_t = perf_counter()
    end_t  = next_t + duration_s
    while True:
        now = perf_counter()
        if now >= end_t:
            break
        control_loop()
        next_t += dt
        # sleep + spin to reduce jitter
        delay = next_t - perf_counter()
        if delay > 0.0015:
            sleep(delay - 0.0005)
        while perf_counter() < next_t:
            pass

# ====== Main ======
if __name__ == "__main__":
    # CPU only; reduce thread overhead
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

    # raise process priority (optional)
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

    # ==== Build MATLAB-equivalent step reference using get_trajectory() ====
    # Example to generate long square-wave scenarios:
    # durations_seg = [60, 60, 60, 60, 60]
    # periods      = [12, 10, 8, 6, 4]
    # A_deg_list   = list(range(10, 31))
    # trajectory = build_step_all_with_get_trajectory(A_deg_list, durations_seg, periods)

    # The following is a user-provided piecewise-constant list (already in radians):
    value    = [ 0.0, 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                        0.2, -0.4, 0.4, 0.0, 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                        0.2, -0.4, 0.4, 0.0, 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                        0.2, -0.4, 0.4, 0.0 , 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                        0.2, -0.4, 0.4, 0.0, 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                        0.2, -0.4, 0.4, 0.0, 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                        0.2, -0.4, 0.4, 0.0, 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                        0.2, -0.4, 0.4, 0.0, 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                        0.2, -0.4, 0.4, 0.0, 0.28368124,-0.28368124, 0.35432990,-0.35432990,
                        0.2, -0.4, 0.4, 0.0]
    duration = [5]*len(value)   # seconds per step
    trajectory = get_trajectory(duration, value)
    # WARNING: Above pattern can be very long (≈ hours). Shorten as needed.
    t_stop = len(trajectory) / sampling_freq

    # ==== Logs ====
    command_storage   = []
    alpha_storage     = []
    theta_storage     = []
    alpha_dot_storage = []
    theta_dot_storage = []
    e_theta_storage   = []
    time_storage      = []

    # ==== Realtime state ====
    prev_theta = 0.0
    prev_alpha = 0.0
    prev_ref   = trajectory[0] if len(trajectory)>0 else 0.0
    hstate     = None
    counter    = -1
    dt_real    = dt_nominal
    t0  = perf_counter()
    tic = t0
    last_print = t0

    # derivative EMA init
    ema_theta_dot = 0.0
    ema_alpha_dot = 0.0

    # ==== Run ====
    print(f"Experiment running [{CONTROLLER}]  |  total duration ≈ {t_stop/60:.1f} min, samples={len(trajectory)}")
    run_realtime(duration_s=t_stop, dt=dt_nominal)
    print("Experiment stopped")

    # Stop motor and close
    buffer_out[0] = 0.0
    card.write_analog(channels, num_channels, buffer_out)
    card.close()

    # ==== Save CSV (headers compatible with the plotting tools) ====
    os.makedirs("results/refs_step", exist_ok=True)
    fname = os.path.normpath("results/refs_step/REF_STEP_all_10to30deg_LQR.csv")

    n = min(len(time_storage), len(command_storage), len(theta_storage), len(alpha_storage))
    ref_used = trajectory[:n]
    alpha_desired = np.zeros_like(ref_used, dtype=float)

    with open(fname, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Time_s", "Command_V", "Theta_rad", "Alpha_rad", "Theta_desired_rad", "Alpha_desired_rad"])
        for i in range(n):
            w.writerow([time_storage[i],
                        command_storage[i],
                        theta_storage[i],
                        alpha_storage[i],
                        ref_used[i],
                        alpha_desired[i]])

    print(f"[OK] Saved CSV -> {fname}")

    # ==== Quick plots ====
    fig, axs = plt.subplots(6, 1, figsize=(11, 12), sharex=True)
    tvec = np.array(time_storage[:n])
    axs[0].plot(tvec, theta_storage[:n]);      axs[0].set_title("Theta (rad)");           axs[0].grid(True)
    axs[1].plot(tvec, alpha_storage[:n]);      axs[1].set_title("Alpha (rad)");           axs[1].grid(True)
    axs[2].plot(tvec, theta_dot_storage[:n]);  axs[2].set_title("Theta_dot (rad/s)");     axs[2].grid(True)
    axs[3].plot(tvec, alpha_dot_storage[:n]);  axs[3].set_title("Alpha_dot (rad/s)");     axs[3].grid(True)
    axs[4].plot(tvec, np.array(theta_storage[:n]) - ref_used); axs[4].set_title("e_theta = theta - ref (rad)"); axs[4].grid(True)
    axs[5].plot(tvec, command_storage[:n]);    axs[5].set_title("Command (V)");           axs[5].set_xlabel("Time (s)"); axs[5].grid(True)
    plt.tight_layout()
    plt.show()
