#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter (remove) specific scenario rows from a packed H5 (/step/...) and write a new file.

Usage
-----
python h5_filter_scenarios.py --in ramp.h5 --out ramp_filtered.h5 --drop "0:10"
python h5_filter_scenarios.py --in sin.h5 --out sinus_filtered.h5 --drop "632:640"
python h5_filter_scenarios.py --in triangle.h5 --out triangle_filtered.h5 --drop "0:10"
python h5_filter_scenarios.py --in cosinus_diff.h5 --out cosinus_diff_filtered.h5 --drop "0:10"
python h5_filter_scenarios.py --in sinus_diff.h5 --out sinus_diff_filtered.h5 --drop "100:10000"
"""

import argparse, h5py, numpy as np, json

def parse_drop(s: str, N: int):
    """
    Parse a list of indices or inclusive ranges into a set of indices to drop.
    Examples: "0:10" (inclusive), "0,1,2", "0:20:2".
    """
    s = s.strip()
    out = set()
    if not s: return out
    for part in s.split(","):
        part = part.strip()
        if not part: continue
        if ":" in part:
            # "start:end" is INCLUSIVE here (0:10 -> 0..10)
            toks = [t for t in part.split(":") if t != ""]
            if len(toks) not in (2,3): raise ValueError(f"Invalid range: {part}")
            a = int(toks[0]); b = int(toks[1])
            step = int(toks[2]) if len(toks)==3 else 1
            rng = range(a, b+1, step)  # inclusive end
            out.update([i for i in rng if 0 <= i < N])
        else:
            i = int(part)
            if 0 <= i < N: out.add(i)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", required=True, help='Indices to delete, e.g. "0:10" or "0,1,2".')
    args = ap.parse_args()

    with h5py.File(args.inp, "r") as fi:
        if "step" not in fi: raise RuntimeError("Input H5 must contain 'step' group (packed layout expected).")
        gi = fi["step"]
        if "ref" not in gi: raise RuntimeError("'step/ref' not found.")
        N, steps = gi["ref"].shape
        keep = np.array(sorted(set(range(N)) - parse_drop(args.drop, N)), dtype=np.int64)
        print(f"[INFO] N={N}, steps={steps} -> drop={N-len(keep)} keep={len(keep)}")

        with h5py.File(args.out, "w") as fo:
            # copy root attrs (dt, T, states_keys, etc.)
            for k, v in fi.attrs.items():
                fo.attrs[k] = v
            fo.attrs["N"] = int(len(keep))

            go = fo.create_group("step")
            # subset every dataset under /step using the kept indices
            for name, ds in gi.items():
                arr = ds[keep, ...]
                go.create_dataset(name, data=arr, compression="lzf")
                print(f"  - wrote /step/{name} {arr.shape} {arr.dtype}")

    print(f"[OK] Kaydedildi -> {args.out}")

if __name__ == "__main__":
    main()
