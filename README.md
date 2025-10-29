# CfC vs LQR for Quanser Rotary Flexible Link (RFL)

End-to-end pipeline to **prepare datasets**, **train CfC models**, **analyze/plot**, and **deploy in real time** on Quanser RFL hardware — with LQR/CfC baselines.

> **Units and I/O (very important)**
>
> - **Angles**: radians; **rates**: rad/s; **command**: Volts.
> - **Model input order (7 features)**:  
>   `[theta, theta_desired, alpha, e_theta, theta_dot, alpha_dot, theta_desired_dot]`,  
>    where `e_theta = theta - theta_desired`.
> - **Model output**: `tanh` in `[-1,1]`, scaled by `*10 → Volts`, then clamped to `[-10,+10]`.

---

## Contents

- [1) Environment setup](#1-environment-setup)
- [2) Repository layout](#2-repository-layout)
- [3) Data preparation (CSV → HDF5)](#3-data-preparation-csv--hdf5)
- [4) Training](#4-training)
- [5) Evaluation & visualization](#5-evaluation--visualization)
- [6) Real-time deployment](#6-real-time-deployment)
- [7) Gray-box identification (optional)](#7-gray-box-identification-optional)
- [8) Reproducibility & conventions](#8-reproducibility--conventions)
- [9) Troubleshooting](#9-troubleshooting)
- [10) License / acknowledgments](#10-license--acknowledgments)

---

## 1) Environment setup

### Python & OS

- **Python**: 3.10+ recommended (3.11 works too).  
- **OS**:  
  - **Windows 10/11** for Quanser hardware (tested real-time @ 500 Hz).  
  - Linux/macOS fine for offline training/plots.

### Create a fresh environment

**Conda**
```bash
conda create -n rfl-cfc python=3.10 -y
conda activate rfl-cfc
Pip (system/venv)


python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
Core dependencies

pip install numpy pandas matplotlib h5py tqdm scipy
pip install ncps               # CfC/NCP library (provides ncps.torch.CfC)
PyTorch
CPU (cross-platform):


pip install torch --index-url https://download.pytorch.org/whl/cpu
GPU: install the appropriate wheel from PyTorch per your CUDA version.

Quanser Python API (for hardware)
The scripts use from quanser.hardware import HIL. Install the Quanser Python API/SDK provided by your lab/vendor (not always on PyPI). After installing drivers and the Python package, verify:


python -c "from quanser.hardware import HIL; print('OK')"
Windows extras (recommended for RT)

pip install pywin32
FFmpeg (for animations)
Install FFmpeg and make sure ffmpeg is on PATH.

Windows (PowerShell):


winget install --id Gyan.FFmpeg -e
Ubuntu:


sudo apt-get update && sudo apt-get install -y ffmpeg
macOS (Homebrew):


brew install ffmpeg
If it isn’t discovered automatically, set:


$env:FFMPEG_PATH="C:\path\to\ffmpeg.exe"
2) Repository layout

.
├── training/
│   ├── cfc_pretrain_imitation_only.py
│   ├── cfc_pretrain_imitation_1step_office.py
│   └── cfc_pretrain_imitation_1stepKstep.py
├── models/
│   ├── model_cfc_ncp.py          # CfC+AutoNCP, dt-agnostic (timespans=None)
│   └── model_cfc_ncp_dt.py       # CfC+AutoNCP with explicit dt (timespans)
├── data_tools/
│   ├── csv_to_h5_scenarios.py    # CSV → (groups/packed) HDF5 slicer
│   ├── h5_filter_scenarios.py    # drop selected scenarios from packed H5
│   ├── merge_step_h5.py          # concatenate multiple packed H5 files
│   └── plot_from_h5.py           # quick plots per-scenario from H5
├── eval/
│   ├── plot_results3.py          # LQR(H5 idx) vs CfC(CSV) on same axes + RMSE
│   └── plot_animation.py         # LQR vs CfC animation (MP4)
├── realtime/
│   ├── Quanser_policy_asaf_office.py   # Deploy CfC on hardware (500 Hz)
│   ├── Quanser_policy_BU-M_asaf.py     # LQR or CfC toggle; typically LQR baseline
│   └── Quanser_policy_asaf_dataset.py  # Run controller and log CSV for dataset
├── graybox/
│   └── graybox_ident.py          # SavGol/Butterworth derivative + (Ad,Bd) fit
├── notebooks/
│   └── CfC_test_simulation_fixed.ipynb
└── docs/
    └── Flexible Joint Workbook _Instructor_.pdf
Folder names are illustrative; place the files accordingly or run from repo root using the paths you use today.

3) Data preparation (CSV → HDF5)
3.1 Slice raw CSV logs into scenarios
data_tools/csv_to_h5_scenarios.py splits a long CSV into T-second scenarios and writes either:

groups mode: each scenario is its own group (ragged length OK),

packed mode: resampled to uniform dt, stacked under /step.

It also computes:

theta_dot (from tachometer or central diff),

alpha_dot (central diff),

theta_desired_dot (central diff),

e_theta = theta - theta_desired.

Examples (PowerShell/Bash):


# STEP logs → packed H5 at 500 Hz (dt=0.002), 10 s per scenario, use tachometer for theta_dot
python data_tools/csv_to_h5_scenarios.py \
  --csv Datasets/raw/LQR_STEP_log.csv \
  --out_h5 Datasets/step.h5 \
  --mode packed --resample_dt 0.002 --T 10 --gear high --theta_dot_from tach

# COSINUS logs → packed H5 using central difference for theta_dot
python data_tools/csv_to_h5_scenarios.py \
  --csv Datasets/raw/LQR_COSINUS_log.csv \
  --out_h5 Datasets/cosinus_diff.h5 \
  --mode packed --resample_dt 0.002 --T 10 --gear high --theta_dot_from diff
3.2 Filter/remove bad scenarios (packed H5)

python data_tools/h5_filter_scenarios.py \
  --in Datasets/step.h5 --out Datasets/step_filtered.h5 --drop "0:10,100,205"
Ranges are inclusive (0:10 = 0…10). The script rewrites all /step/* datasets with the kept rows.

3.3 Merge multiple packed H5 files
All inputs must share the same (dt, steps, states_keys).

python data_tools/merge_step_h5.py \
  --out Datasets/combined_all_steps.h5 \
  Datasets/step_filtered.h5 Datasets/sinus_filtered.h5 Datasets/cosinus_diff.h5
3.4 Quick preview plots from H5

python data_tools/plot_from_h5.py --h5 Datasets/combined_all_steps.h5 --idx 20 --outdir plots
4) Training
All training assumes the input feature order and targets described at the top.

4.1 Imitation-only pretrain
training/cfc_pretrain_imitation_only.py
Trains CfC to imitate the teacher command (u_teacher / 10), with z-score normalization computed on the train split only.


# Default CfC mode
python training/cfc_pretrain_imitation_only.py \
  --data Datasets/combined_all_steps.h5 \
  --save_dir runs/cfc_pretrain_imitation_only_default12N \
  --epochs 120 --patience 15 --batch_size 128 \
  --lr 7e-4 --u_imitation_w 1.0 \
  --cfc_mode default --log_csv

# PURE mode (no gating), 12 units
python training/cfc_pretrain_imitation_only.py \
  --data Datasets/combined_all_steps.h5 \
  --save_dir runs/cfc_pretrain_imitation_only_pure_12N \
  --epochs 120 --patience 15 --batch_size 128 \
  --lr 7e-4 --u_imitation_w 5.0 \
  --cfc_mode pure --log_csv
Artifacts:

best_ckpt.pth (saves model_state, dt, and config)

norm_stats.npz (mean/std used at deployment)

optional metrics.csv

4.2 Imitation + 1-step rollout tracking
training/cfc_pretrain_imitation_1step_office.py
Adds a 1-step rollout loss on a plant/ensemble (if configured in the script). Typical usage mirrors imitation-only; see in-file CLI.


python training/cfc_pretrain_imitation_1step_office.py \
  --data Datasets/combined_all_steps.h5 \
  --save_dir runs/cfc_pretrain_1step_default12N \
  --epochs 120 --patience 15 --batch_size 128 \
  --lr 7e-4 --u_imitation_w 1.0 --log_csv
4.3 Imitation + CLF (1-step or K-step)
training/cfc_pretrain_imitation_1stepKstep.py
Imitation + Control Lyapunov Function term (V = β_eθ * ½ e_θ² + β_θ * ½ θ̇² + β_α * ½ α²), computed at 1-step or K-step horizon.


python training/cfc_pretrain_imitation_1stepKstep.py \
  --data Datasets/combined_all_steps.h5 \
  --save_dir runs/cfc_pretrain_imitation_1stepCLF_12N \
  --epochs 120 --patience 15 --batch_size 128 \
  --lr 7e-4 --u_imitation_w 5.0 \
  --clf_mode 1step --beta_etheta 1.0 --beta_theta 0.1 --beta_alpha 0.1 --log_csv
Check the script’s CLI for exact flag names; the above matches the annotated version used in this repo.

5) Evaluation & visualization
5.1 LQR(H5) vs CfC(CSV) on shared axes + RMSE
eval/plot_results3.py reads LQR directly from the H5 dataset at a given scenario index and CfC from a CSV log (e.g., real-time run). It computes RMSEs for:

(a) theta vs theta_ref,

(b) alpha vs 0,

(c) command vs 0.

It can auto-align the references via cross-correlation to absorb small time shifts.


python eval/plot_results3.py \
  -- (edit inside the file, or pass via env) \
  # Typical inline edits:
  # DATASET_H5_PATH = "Datasets/combined_all_steps.h5"
  # SCEN_IDX        = 250
  # CFC_CSV_PATH    = "results/.../Trajectory_step_cfc_pretrain_1step_robust_office_idx_250.csv"
5.2 Animation (MP4): LQR vs CfC
eval/plot_animation.py builds a 3-panel animation (θ, α, u) from two CSV logs (LQR and CfC). It resamples to a target VIDEO_FPS and uses FFmpeg to export MP4.


python eval/plot_animation.py
# Edit at the top:
# LQR_CSV_PATH, CFC_CSV_PATH, DT, VIDEO_FPS, SAVE_PATH
If FFmpeg isn’t found, set FFMPEG_PATH or install FFmpeg (see Env section).

6) Real-time deployment
⚠️ Hardware safety

Keep command clamps (±10 V), speed limits, and an accessible E-stop.

Start with small amplitudes; verify direction/signs before longer tests.

6.1 Deploy CfC @ 500 Hz
realtime/Quanser_policy_asaf_office.py uses your trained CfC checkpoint:

Loads best_ckpt.pth and norm_stats.npz.

Builds a reference trajectory (several helper generators) or reads from H5.

Runs a fixed-rate sleep+spin loop with 1 ms Windows timer resolution.


python realtime/Quanser_policy_asaf_office.py
# Edit at the top:
# CKPT_PATH, NORM_NPZ_PATH
# Choose a reference (explicit steps, triangle/ramp/sine, or dataset idx)
6.2 LQR baseline (and CfC toggle)
realtime/Quanser_policy_BU-M_asaf.py supports CONTROLLER="LQR" or "CfC". For LQR:


# In file:
CONTROLLER = "LQR"
LQR_GAIN   = [11.8303, -30.4544, 1.4627, -0.6952]  # [e_theta, -alpha, -theta_dot, -alpha_dot]
LQR_SIGN   = -1.0                                   # flip if needed to match plant polarity

python realtime/Quanser_policy_BU-M_asaf.py
6.3 Generate datasets on hardware
realtime/Quanser_policy_asaf_dataset.py runs a controller (e.g., LQR) and logs CSV with canonical headers used by the data tools.


python realtime/Quanser_policy_asaf_dataset.py
# Edit 'value' and 'duration' (or use dataset-based reference)
# Output CSV is compatible with csv_to_h5_scenarios.py
7) Gray-box identification (optional)
graybox/graybox_ident.py can:

unwrap angles, low-pass (Butterworth), differentiate with Savitzky–Golay,

build a one-step dataset and fit discrete (Ad, Bd) with ridge regularization.


python graybox/graybox_ident.py \
  --data Datasets/combined_all_steps.h5 \
  --indices "0:1000" \
  --sg_window 31 --sg_poly 3 \
  --butter_cutoff_hz 40 \
  --unwrap_angles \
  --ridge 1e-6 \
  --save_npz runs/plant_graybox2.npz
8) Reproducibility & conventions
Normalization: mean/std computed on train split only and saved to norm_stats.npz. The identical stats must be used at deployment.

Feature order: strictly [theta, theta_desired, alpha, e_theta, theta_dot, alpha_dot, theta_desired_dot].

Targets: imitation targets are u_teacher/10 to match tanh output in [-1,1].

Time step:

offline H5 uses root attr dt (e.g., 0.002),

real-time loop passes the measured dt_real each tick to CfC (for *_dt.py models).

Modes: model_cfc_ncp.py for dt-agnostic CfC; model_cfc_ncp_dt.py when you want explicit timespans (mode ∈ {default,no_gate,pure}).

9) Troubleshooting
FFmpeg not found → install and/or set FFMPEG_PATH. The animation script also probes common Windows locations.

CSV header mismatch → plotting/animation scripts auto-map many aliases and convert deg→rad when headers include (deg). Check ALIASES maps at the top of those scripts.

Quanser import error → install the Quanser Python API/SDK for your device and ensure drivers are installed. Verify with python -c "from quanser.hardware import HIL".

Loop rate < 500 Hz → close browsers/IDE, switch to CPU only (torch.set_num_threads(1) already set), keep plotting off during runs, raise process priority (built in), and ensure USB bandwidth is OK.

LQR sign/gain → If response is inverted, swap LQR_SIGN between +1.0 and -1.0. Confirm the state vector definition matches your gain design.

RMSE looks worse despite better plots → enable reference alignment in plot_results3.py (ALIGN_BY_REF = True) and/or use the active mask to focus on periods with nonzero reference.

10) License / acknowledgments
Code in this repo is for research/education on the Quanser RFL.

Quanser names and APIs belong to Quanser, Inc.

CfC/NCP implementation via the ncps Python package.

Please review safety procedures before running experiments.

Quick command cheat-sheet
Prep


# CSV → packed H5 @ 500 Hz
python data_tools/csv_to_h5_scenarios.py \
  --csv Datasets/raw/LQR_STEP_log.csv \
  --out_h5 Datasets/step.h5 \
  --mode packed --resample_dt 0.002 --T 10 --gear high

# Filter indices
python data_tools/h5_filter_scenarios.py --in Datasets/step.h5 --out Datasets/step_f.h5 --drop "0:10"

# Merge
python data_tools/merge_step_h5.py --out Datasets/combined_all_steps.h5 Datasets/step_f.h5 Datasets/sinus.h5
Train (imitation-only)


python training/cfc_pretrain_imitation_only.py \
  --data Datasets/combined_all_steps.h5 \
  --save_dir runs/cfc_pretrain_imitation_only_default12N \
  --epochs 120 --patience 15 --batch_size 128 \
  --lr 7e-4 --u_imitation_w 1.0 \
  --cfc_mode default --log_csv
Evaluate


# LQR(H5 idx) vs CfC(CSV)
python eval/plot_results3.py
# (edit DATASET_H5_PATH, SCEN_IDX, CFC_CSV_PATH at top)
Animate


python eval/plot_animation.py
# (edit LQR_CSV_PATH, CFC_CSV_PATH, DT, VIDEO_FPS, SAVE_PATH)
Deploy (CfC)


python realtime/Quanser_policy_asaf_office.py
# (set CKPT_PATH, NORM_NPZ_PATH; choose a reference)
Deploy (LQR)

python realtime/Quanser_policy_BU-M_asaf.py
# ensure CONTROLLER="LQR"
Record dataset

bash

python realtime/Quanser_policy_asaf_dataset.py
# edit value/duration or dataset-based reference
