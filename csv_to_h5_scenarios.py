#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a raw CSV log into fixed-length scenarios and export to HDF5/CSV.

What this script does
---------------------
• Reads a CSV with columns like:
    Time_s, Command_V, Theta_rad, Alpha_rad, Tachometer_V,
    Theta_desired_rad, Alpha_desired_rad
• Slices the time series into non-overlapping windows of length T seconds
  and writes either:
    - "groups" mode: one H5 group per scenario (ragged time is okay), or
    - "packed" mode: re-sampled to uniform dt; all scenarios stacked under /step.
• Adds derived channels:
    - theta_dot (rad/s): from tachometer voltage with SRV02 gear calibration,
      or falls back to central-difference derivative if requested/unavailable.
    - alpha_dot (rad/s): optional central-difference derivative.
    - theta_desired_dot (rad/s): optional central-difference derivative.
    - e_theta = theta - theta_desired.

Tachometer calibration (SRV02)
------------------------------
Default uses HIGH gear unless overridden:
    HIGH gear (G=70):  gain ≈ -0.99733 rad/s per Volt
    LOW  gear (G=14):  gain ≈ -4.98666 rad/s per Volt
Derived from:
  Tach Cal [krpm/V] = -1 / (1.5 * G)
  1 krpm = 1000 rpm = 1000 * 2π / 60 rad/s

Modes
-----
--mode groups : Each scenario is written as a separate group (ragged length OK).
--mode packed : Re-sample uniformly (dt), then store as fixed-size datasets.

Metadata
--------
• In "groups" mode, each scenario carries its own 'states_keys' attribute.
• In "packed" mode, root attributes carry: dt, T, N, states_keys.

Examples
--------
python csv_to_h5_scenarios.py --csv LQR2_COSINUS_asaf.csv --out_h5 cos_new.h5 \
  --mode packed --resample_dt 0.002 --T 10 --gear high --theta_dot_from diff
python csv_to_h5_scenarios.py --csv LQR2_STEP_asaf.csv --out_h5 step_new30.h5 \
  --mode packed --resample_dt 0.002 --T 30 --gear high --theta_dot_from diff
python csv_to_h5_scenarios.py --csv LQI_STEP_asaf.csv --out_h5 step_lqi.h5 \
  --mode packed --resample_dt 0.002 --T 10 --gear high
"""

import argparse, os, math, json, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import h5py

# -------------------- Columns --------------------
WANTED_COLS = {
     "Time_s", "Command_V", "Theta_rad", "Alpha_rad",
    "Tachometer_V", "Theta_desired_rad", "Alpha_desired_rad",
}
SIMPLIFY_MAP = {
    "Time_s": "time",
    "Command_V": "command",
    "Theta_rad": "theta",
    "Alpha_rad": "alpha",
    "Tachometer_V": "tachometer_v",
    "Theta_desired_rad": "theta_desired",
    "Alpha_desired_rad": "alpha_desired",
}

def usecols(col: str) -> bool:
    """Filter only the expected columns when reading the CSV."""
    return col.strip() in WANTED_COLS

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names into a compact set (time, command, theta, alpha, ...).
    If no time column is obtained after renaming, raise a clear error.
    """
    cols = [c.strip() for c in df.columns]
    rename = {}
    for c in cols:
        key = c
        if key in SIMPLIFY_MAP:
            rename[c] = SIMPLIFY_MAP[key]
        else:
            # light tolerance: drop spaces and try again
            key2 = c.replace(" ", "")
            if key2 in SIMPLIFY_MAP:
                rename[c] = SIMPLIFY_MAP[key2]
    df = df.rename(columns=rename)
    if "time" not in df.columns:
        raise KeyError(
            f"time column not found after rename. Seen columns: {list(df.columns)}. "
            f"Expected one of {sorted(WANTED_COLS)}"
        )
    return df


# -------------------- Utilities --------------------
def central_diff(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Central-difference derivative; forward/backward difference at the ends. t must be monotonic."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(y, dtype=np.float64)
    if len(y) < 2:
        return np.zeros_like(y, dtype=np.float64)
    dt = np.diff(t)
    # robust: forward/backward difference at edges for nonuniform dt
    out[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    out[0] = (y[1] - y[0]) / (t[1] - t[0])
    out[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    return out

def resample_uniform(t: np.ndarray, series: Dict[str, np.ndarray], dt: float, t0: float, T: float):
    """Linear interpolation onto a uniform grid [t0, t0+T) with sample time dt."""
    steps = int(math.floor(T/dt))
    tg = (t0 + np.arange(steps) * dt).astype(np.float64)
    out = {}
    idx = np.argsort(t)
    ts = t[idx]
    for k, v in series.items():
        vs = np.asarray(v, dtype=np.float64)[idx]
        out[k] = np.interp(tg, ts, vs)
    return tg, out

def compute_dt_stats(t: np.ndarray) -> Dict[str, float]:
    """Return median/min/max of time differences for a given time vector."""
    if len(t) < 2:
        return dict(dt_median=np.nan, dt_min=np.nan, dt_max=np.nan)
    dt = np.diff(t)
    return dict(dt_median=float(np.median(dt)), dt_min=float(np.min(dt)), dt_max=float(np.max(dt)))

# ---- SRV02 tachometer gain helpers ----
def tach_gain_rad_per_s_per_V(gear: str) -> float:
    """
    SRV02 tachometer calibration:
      Tach Cal [krpm/V] = -1 / (1.5 * G)
      Convert to rad/s per V: gain = TachCal * (1000 * 2π / 60)
    HIGH gear: G=70  → ≈ -0.99733 rad/s/V
    LOW  gear: G=14  → ≈ -4.98666 rad/s/V
    """
    gear = gear.lower().strip()
    if gear not in {"high", "low"}:
        raise ValueError("gear must be 'high' or 'low'")
    G = 70 if gear == "high" else 14
    gain = -(1000.0 * 2.0 * math.pi) / (60.0 * 1.5 * G)
    return gain

# -------------------- H5 Helpers --------------------
class PackedWriter:
    """
    Writer for 'packed' layout with extendable datasets of shape (N, steps, ...).
    Keeps dynamic 'states_keys' compatibility across runs.
    """
    def __init__(self, h5: h5py.File, steps: int, state_dim: int):
        self.steps = steps
        self.n = 0
        g = h5.require_group("step")
        self.ref = g.create_dataset("ref", shape=(0, steps), maxshape=(None, steps), dtype="f4", compression="lzf")
        self.states = g.create_dataset("states", shape=(0, steps, state_dim),
                                       maxshape=(None, steps, state_dim), dtype="f4", compression="lzf")
        self.errors = g.create_dataset("errors", shape=(0, steps, 2), maxshape=(None, steps, 2), dtype="f4", compression="lzf")
        self.e_theta = g.create_dataset("e_theta", shape=(0, steps), maxshape=(None, steps), dtype="f4", compression="lzf")
        self.control = g.create_dataset("control", shape=(0, steps), maxshape=(None, steps), dtype="f4", compression="lzf")
        # Optional derivatives as separate datasets
        self.theta_dot = g.create_dataset("theta_dot", shape=(0, steps), maxshape=(None, steps), dtype="f4", compression="lzf")
        self.alpha_dot = g.create_dataset("alpha_dot", shape=(0, steps), maxshape=(None, steps), dtype="f4", compression="lzf")
        self.theta_desired_dot = g.create_dataset("theta_desired_dot", shape=(0, steps), maxshape=(None, steps), dtype="f4", compression="lzf")

    def append(self, ref, states, errors, e_theta, control, theta_dot, alpha_dot, theta_desired_dot):
        """Append one scenario row to all datasets (auto-resizes first dimension)."""
        i = self.n
        self.n += 1
        for ds_name, arr in [
            ("ref", ref.astype("f4", copy=False)),
            ("states", states.astype("f4", copy=False)),
            ("errors", errors.astype("f4", copy=False)),
            ("e_theta", e_theta.astype("f4", copy=False)),
            ("control", control.astype("f4", copy=False)),
            ("theta_dot", theta_dot.astype("f4", copy=False)),
            ("alpha_dot", alpha_dot.astype("f4", copy=False)),
            ("theta_desired_dot", theta_desired_dot.astype("f4", copy=False)),
        ]:
            ds = getattr(self, ds_name)
            new_shape = list(ds.shape)
            new_shape[0] = self.n
            ds.resize(tuple(new_shape))
            ds[i] = arr

def write_group(h5: h5py.File, idx: int, data: Dict[str, np.ndarray], meta: Dict[str, float], states_keys: List[str]):
    """Write one scenario as a distinct H5 group (ragged-friendly)."""
    g = h5.create_group(f"scenarios/{idx:06d}")
    for k, v in data.items():
        g.create_dataset(k, data=v, compression="lzf")
    for k, v in meta.items():
        g.attrs[k] = v
    g.attrs["states_keys"] = json.dumps(list(states_keys))

# -------------------- Slicer --------------------
def slice_and_write(
    csv_path: str,
    out_h5: Optional[str],
    out_csv: Optional[str],
    T: float = 10.0,
    keep_partial: bool = False,
    mode: str = "groups",
    resample_dt: Optional[float] = None,
    chunksize: int = 1_000_000,
    csvsep: str = ",",
    add_theta_dot: bool = True,
    add_alpha_dot: bool = True,
    add_theta_desired_dot: bool = True,
    gear: str = "high",
    tach_gain_manual: Optional[float] = None,
    tach_flip_sign: bool = False,
    theta_dot_from: str = "tach",          # NEW: source of theta_dot (tachometer vs diff)
):
    """
    Stream the CSV, cut it into T-second windows, and write to CSV/H5.

    Parameters
    ----------
    mode : {'groups','packed'}
        'groups' keeps original timestamps per scenario;
        'packed' resamples to uniform 'resample_dt' and stacks under /step.
    theta_dot_from : {'tach','diff'}
        Use calibrated tachometer voltage (preferred) or central difference.
    """
    assert mode in {"groups", "packed"}, "--mode must be one of {groups,packed}"

    # Tachometer conversion gain (V -> rad/s)
    if tach_gain_manual is not None:
        gain_rs_per_V = float(tach_gain_manual)
        gain_src = "manual"
    else:
        gain_rs_per_V = tach_gain_rad_per_s_per_V(gear)
        gain_src = f"SRV02_{gear}_gear"

    if tach_flip_sign:
        gain_rs_per_V = -gain_rs_per_V
        gain_src += "_flipped"

    # Prepare single CSV (optional)
    csv_f = None
    wrote_header = False
    if out_csv:
        csv_f = open(out_csv, "w", encoding="utf-8", newline="")

    # Open H5 (optional)
    h5 = None
    packed_writer = None
    if out_h5:
        h5 = h5py.File(out_h5, "w")
        h5.attrs["created"] = datetime.datetime.utcnow().isoformat() + "Z"
        h5.attrs["source_csv"] = os.path.abspath(csv_path)
        h5.attrs["desc"] = "Real data → T-second scenario slicer output"
        h5.attrs["tach_gain_rad_per_s_per_V"] = float(gain_rs_per_V)
        h5.attrs["tach_gain_source"] = gain_src

    # Streaming state
    buf = pd.DataFrame(columns=["time","command","theta","alpha","tachometer_v","theta_desired","alpha_desired"])
    cur_start = None
    cur_end = None
    scenario_idx = 0
    first_time_seen = None

    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        usecols=usecols,
        skipinitialspace=True,
        engine="c",
        on_bad_lines="skip",
        sep=csvsep,
    )

    def flush_scenario(df_win: pd.DataFrame, start_t: float):
        """Finalize one window: compute features, write CSV/H5, bump scenario_idx."""
        nonlocal scenario_idx, h5, packed_writer, wrote_header, csv_f
        if df_win.empty:
            return

        # numpy arrays
        t   = df_win["time"].to_numpy(dtype=np.float64)
        th  = df_win["theta"].to_numpy(dtype=np.float64)
        al  = df_win["alpha"].to_numpy(dtype=np.float64)
        cmd = df_win["command"].to_numpy(dtype=np.float64)

        # references (fallback to zeros if missing)
        thd = df_win["theta_desired"].to_numpy(dtype=np.float64) if "theta_desired" in df_win else np.zeros_like(th)
        ald = df_win["alpha_desired"].to_numpy(dtype=np.float64) if "alpha_desired" in df_win else np.zeros_like(al)

        # error
        e_th = th - thd

        # derivatives
        if add_theta_dot:
            if theta_dot_from == "tach":
                if "tachometer_v" in df_win and not df_win["tachometer_v"].isna().all():
                    v_tach = df_win["tachometer_v"].to_numpy(dtype=np.float64)
                    thdot = gain_rs_per_V * v_tach
                else:
                    thdot = central_diff(t, th)
            else:  # "diff"
                thdot = central_diff(t, th)
        else:
            thdot = np.zeros_like(th)

        aldot = central_diff(t, al) if add_alpha_dot else np.zeros_like(al)
        thddot = central_diff(t, thd) if add_theta_desired_dot else np.zeros_like(thd)

        # assemble states dynamically
        states_list = [th, al]
        states_keys = ["theta", "alpha"]
        if add_theta_dot:
            states_list.append(thdot); states_keys.append("theta_dot")
        if add_alpha_dot:
            states_list.append(aldot); states_keys.append("alpha_dot")
        states = np.stack(states_list, axis=-1)

        # optional CSV output: scenario_id + e_theta and derivatives
        if out_csv:
            df_to_write = df_win.copy()
            df_to_write.insert(0, "scenario_id", int(scenario_idx))
            df_to_write["e_theta"] = e_th
            if add_theta_dot: df_to_write["theta_dot"] = thdot
            if add_alpha_dot: df_to_write["alpha_dot"] = aldot
            if add_theta_desired_dot: df_to_write["theta_desired_dot"] = thddot
            if not wrote_header:
                df_to_write.to_csv(csv_f, index=False)
                wrote_header = True
            else:
                df_to_write.to_csv(csv_f, index=False, header=False)

        if h5:
            if mode == "groups":
                meta = dict(
                    t_start=float(t[0]),
                    t_end=float(t[-1]),
                    n=int(len(t)),
                    **compute_dt_stats(t),
                )
                data = {
                    "time": t.astype("f8"),
                    "ref": thd.astype("f4"),
                    "states": states.astype("f4"),
                    "errors": np.stack([e_th, al - ald], axis=-1).astype("f4"),
                    "e_theta": e_th.astype("f4"),
                    "control": cmd.astype("f4"),
                    "theta_dot": thdot.astype("f4"),
                    "alpha_dot": aldot.astype("f4"),
                    "theta_desired_dot": thddot.astype("f4"),
                }
                write_group(h5, scenario_idx, data, meta, states_keys)
            else:  # packed
                assert resample_dt is not None and resample_dt > 0, "--mode packed requires --resample_dt"
                series = {
                    "th": th, "al": al, "cmd": cmd, "thd": thd, "ald": ald,
                    "thdot": thdot, "aldot": aldot, "thddot": thddot, "e_th": e_th,
                }
                tg, rs = resample_uniform(t, series, resample_dt, start_t, T)
                ref = rs["thd"].astype(np.float32)
                # states (dynamic)
                states_keys = ["theta", "alpha"]
                states_list = [rs["th"], rs["al"]]
                if add_theta_dot:
                    states_keys.append("theta_dot"); states_list.append(rs["thdot"])
                if add_alpha_dot:
                    states_keys.append("alpha_dot"); states_list.append(rs["aldot"])
                states_rs = np.stack(states_list, axis=-1).astype(np.float32)
                errors_rs = np.stack([rs["e_th"], rs["al"] - rs["ald"]], axis=-1).astype(np.float32)
                e_theta_rs = rs["e_th"].astype(np.float32)
                control_rs = rs["cmd"].astype(np.float32)
                theta_dot_rs = rs["thdot"].astype(np.float32)
                alpha_dot_rs = rs["aldot"].astype(np.float32)
                theta_desired_dot_rs = rs["thddot"].astype(np.float32)

                steps = len(tg)
                if packed_writer is None:
                    packed_writer = PackedWriter(h5, steps=steps, state_dim=states_rs.shape[-1])
                    if h5 is not None:
                        h5.attrs["dt"] = float(resample_dt)
                        h5.attrs["T"] = float(T)
                        h5.attrs["states_keys"] = json.dumps(states_keys)
                packed_writer.append(ref, states_rs, errors_rs, e_theta_rs, control_rs,
                                     theta_dot_rs, alpha_dot_rs, theta_desired_dot_rs)

        scenario_idx += 1

    # streaming loop
    for chunk in reader:
        chunk = norm_cols(chunk).sort_values("time")
        if len(chunk) == 0:
            continue

        if first_time_seen is None:
            first_time_seen = float(chunk["time"].iloc[0])
            cur_start = first_time_seen
            cur_end = cur_start + T

        # append into buffer (keep sorted by time)
        buf = pd.concat([buf, chunk], ignore_index=True)
        buf = buf.sort_values("time").reset_index(drop=True)

        # whenever the current window fills, flush & advance
        while True:
            m = (buf["time"] >= cur_start) & (buf["time"] < cur_end)
            if not m.any():
                break
            df_win = buf.loc[m].copy()
            flush_scenario(df_win, cur_start)
            # drop processed rows
            buf = buf.loc[~m].reset_index(drop=True)
            # next window
            cur_start = cur_end
            cur_end = cur_start + T

    # optionally flush remaining (partial) window
    if keep_partial and cur_start is not None:
        m = buf["time"] >= cur_start
        if m.any():
            df_win = buf.loc[m].copy()
            flush_scenario(df_win, cur_start)

    # close resources
    if csv_f:
        csv_f.close()
    if h5:
        if mode == "packed" and "step" in h5:
            h5.attrs["N"] = int(h5["step"]["ref"].shape[0])
        h5.close()

    return scenario_idx

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--out_h5", default=None, help="Output HDF5 path (optional)")
    ap.add_argument("--out_csv", default=None, help="Single CSV with scenario_id (optional)")
    ap.add_argument("--T", type=float, default=10.0, help="Scenario length [s] (default: 10)")
    ap.add_argument("--keep_partial", action="store_true", help="Also write the last partial window")
    ap.add_argument("--mode", choices=["groups","packed"], default="groups", help="HDF5 writing mode")
    ap.add_argument("--resample_dt", type=float, default=None, help="Uniform dt for 'packed' mode")
    ap.add_argument("--chunksize", type=int, default=1_000_000, help="read_csv chunksize")
    ap.add_argument("--csvsep", default=",", help="CSV separator (default: ,)")
    ap.add_argument("--no_theta_dot", action="store_true", help="Do not compute/export theta_dot")
    ap.add_argument("--no_alpha_dot", action="store_true", help="Do not compute/export alpha_dot")
    ap.add_argument("--no_theta_desired_dot", action="store_true", help="Do not compute/export theta_desired_dot")
    ap.add_argument("--theta_dot_from", choices=["tach","diff"], default="tach",
                    help="theta_dot source: 'tach' (tachometer V→rad/s) or 'diff' (central diff). Default: tach")

    # Tachometer conversion options
    ap.add_argument("--gear", choices=["high","low"], default="high",
                    help="SRV02 gear for tachometer calibration (default: high)")
    ap.add_argument("--tach_gain", type=float, default=None,
                    help="Manual tachometer gain (rad/s per Volt). Overrides --gear if set.")
    ap.add_argument("--tach_flip_sign", action="store_true",
                    help="Flip sign after conversion (multiply by -1).")

    args = ap.parse_args()

    scenarios = slice_and_write(
        csv_path=args.csv,
        out_h5=args.out_h5,
        out_csv=args.out_csv,
        T=args.T,
        keep_partial=args.keep_partial,
        mode=args.mode,
        resample_dt=args.resample_dt,
        chunksize=args.chunksize,
        csvsep=args.csvsep,
        add_theta_dot=(not args.no_theta_dot),
        add_alpha_dot=(not args.no_alpha_dot),
        add_theta_desired_dot=(not args.no_theta_desired_dot),
        gear=args.gear,
        tach_gain_manual=args.tach_gain,
        tach_flip_sign=args.tach_flip_sign,
        theta_dot_from=args.theta_dot_from,
    )
    print(f"[OK] Toplam senaryo: {scenarios}")

if __name__ == "__main__":
    main()

"""
Example:
python csv_to_h5_scenarios.py --csv LQR_RAMP_NoSlewRate.csv --out_h5 ramp.h5 \
  --mode packed --resample_dt 0.002 --T 10 --gear high
"""
