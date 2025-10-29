#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Concatenate multiple 'packed' H5 datasets (with /step group) into one file.

What it enforces
----------------
• All inputs must have identical:
  - dt (root attr),
  - steps (second dimension of /step/ref),
  - states_keys (root attr; JSON list).
• If any input lacks /step/theta_desired_dot, it is computed by central difference.

Outputs
-------
• /step/{states, errors, ref, control, theta_desired_dot}
• Root attrs: dt, T, N, states_keys, sources (list of input files & N).
"""

import os
import json
import argparse
import h5py
import numpy as np

def finite_diff(ref: np.ndarray, dt: float) -> np.ndarray:
    """Compute central-difference derivative for each row of ref (N, T)."""
    r = ref.astype(np.float64)
    N, T = r.shape
    rdot = np.empty((N, T), dtype=np.float32)
    # interior
    rdot[:, 1:-1] = (r[:, 2:] - r[:, :-2]) / (2 * dt)
    # edges
    rdot[:, 0] = (r[:, 1] - r[:, 0]) / dt
    rdot[:, -1] = (r[:, -1] - r[:, -2]) / dt
    return rdot

def append_block(dst, start, states, errors, ref, control, rdot):
    """Append a contiguous block (Ni rows) to resizable target datasets, return new cursor."""
    n = states.shape[0]
    end = start + n
    # resize and write
    for name, arr in [
        ("states",  states),
        ("errors",  errors),
        ("ref",     ref),
        ("control", control),
        ("theta_desired_dot", rdot),
    ]:
        ds = dst[name]
        new_shape = list(ds.shape)
        new_shape[0] = end
        ds.resize(new_shape)
        ds[start:end] = arr
    return end

def merge_h5(out_path: str, in_paths: list[str]):
    """Concatenate inputs in the given order into a single 'packed' H5."""
    assert len(in_paths) > 0, "No input files provided."

    # --- Read metadata from first file ---
    with h5py.File(in_paths[0], "r") as f0:
        g0 = f0["step"]
        dt = float(f0.attrs["dt"])
        T = float(f0.attrs["T"])
        steps = g0["ref"].shape[1]
        states_keys_attr = f0.attrs.get("states_keys", None)
        if isinstance(states_keys_attr, bytes):
            states_keys_attr = states_keys_attr.decode("utf-8")
        states_keys = json.loads(states_keys_attr) if states_keys_attr else ['theta','alpha','theta_dot','alpha_dot']

    # --- Sanity-check others ---
    total_N = 0
    source_info = []
    for p in in_paths:
        with h5py.File(p, "r") as f:
            g = f["step"]
            N = g["ref"].shape[0]
            dt_i = float(f.attrs["dt"]); T_i = float(f.attrs["T"]); steps_i = g["ref"].shape[1]
            sk_attr = f.attrs.get("states_keys", None)
            if isinstance(sk_attr, bytes):
                sk_attr = sk_attr.decode("utf-8")
            sk = json.loads(sk_attr) if sk_attr else states_keys
            if abs(dt_i - dt) > 1e-12 or steps_i != steps or sk != states_keys:
                raise ValueError(
                    f"Metadata mismatch in {p}.\n"
                    f"Expected dt={dt}, steps={steps}, states_keys={states_keys}\n"
                    f"Found    dt={dt_i}, steps={steps_i}, states_keys={sk}"
                )
            total_N += N
            source_info.append({"file": os.path.basename(p), "N": int(N)})

    # --- Create destination file with resizable datasets ---
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as fout:
        fout.attrs["dt"] = dt
        fout.attrs["T"] = T
        fout.attrs["N"] = int(total_N)
        fout.attrs["states_keys"] = json.dumps(states_keys)
        fout.attrs["sources"] = json.dumps(source_info)

        gdst = fout.create_group("step")

        # Pre-create resizable datasets
        gdst.create_dataset(
            "states", shape=(0, steps, 4), maxshape=(None, steps, 4),
            dtype="f4", compression="gzip", compression_opts=4, shuffle=True, chunks=(1, steps, 4)
        )
        gdst.create_dataset(
            "errors", shape=(0, steps, 2), maxshape=(None, steps, 2),
            dtype="f4", compression="gzip", compression_opts=4, shuffle=True, chunks=(1, steps, 2)
        )
        gdst.create_dataset(
            "ref", shape=(0, steps), maxshape=(None, steps),
            dtype="f4", compression="gzip", compression_opts=4, shuffle=True, chunks=(1, steps)
        )
        gdst.create_dataset(
            "control", shape=(0, steps), maxshape=(None, steps),
            dtype="f4", compression="gzip", compression_opts=4, shuffle=True, chunks=(1, steps)
        )
        gdst.create_dataset(
            "theta_desired_dot", shape=(0, steps), maxshape=(None, steps),
            dtype="f4", compression="gzip", compression_opts=4, shuffle=True, chunks=(1, steps)
        )

        # --- Append each file in order ---
        cursor = 0
        for p in in_paths:
            with h5py.File(p, "r") as f:
                g = f["step"]
                states = g["states"][:].astype(np.float32)    # (Ni, steps, 4)
                errors = g["errors"][:].astype(np.float32)    # (Ni, steps, 2)
                ref    = g["ref"][:].astype(np.float32)       # (Ni, steps)
                control= g["control"][:].astype(np.float32)   # (Ni, steps)

                # Prefer file rdot; compute if missing
                if "theta_desired_dot" in g:
                    rdot = g["theta_desired_dot"][:].astype(np.float32)
                else:
                    rdot = finite_diff(ref, dt)

                cursor = append_block(gdst, cursor, states, errors, ref, control, rdot)
            print(f"[OK] appended {os.path.basename(p)} (running N={cursor})")

        # Final N (redundant but kept in sync)
        fout.attrs["N"] = int(cursor)

    print(f"\n[MERGE DONE] -> {out_path}\n  Total N = {total_N}  | steps = {steps}  | dt = {dt}")
    print(f"Sources: {', '.join(os.path.basename(p) for p in in_paths)}")

def main():
    ap = argparse.ArgumentParser("Concatenate multiple /step H5 datasets into one file.")
    ap.add_argument("--out", required=True, help="Output H5 file path")
    ap.add_argument("files", nargs="+", help="Input H5 files (order is preserved)")
    args = ap.parse_args()
    merge_h5(args.out, args.files)

if __name__ == "__main__":
    main()
