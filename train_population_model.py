"""
train_population_model.py  (PKL dataset version)

Trains the population-level NN state-space model T1DSim_NN^P
(CGMOHSUSimStateSpaceModel_V2) on the simulated .pkl dataset produced by
CodeForGeneratingDataset/datasetPop.py, following Section 2.2.2 of
Roquemen-Echeverri et al. (arXiv:2508.05705).

EXPECTED INPUT (produced by datasetPop.py):
  <SIMULATION_RESULTS_DIR>/
    SIM002/POPULATION/TRAIN/                    *.pkl   TRAIN split
    SIM002/POPULATION/VALIDATION/               *.pkl   VAL  split
    SIM004/boluslate_*/TRAIN/                   *.pkl   delay augmentation (TRAIN)
    SIM005/bolusCorrectionFactor_*/TRAIN/       *.pkl   correction bolus   (TRAIN)

Each .pkl is a Pandas DataFrame with 11 dosing-problem groups x 2016
timesteps (T = 2016 at Dt = 5 min).  Key columns used:

  output_glucose    mg/dL   plant ground-truth glucose (Q1 target for L_fit)
  input_insulin     U/h     total insulin administered
  input_meal_carbs  g       carbohydrate intake
  state_Q1 .. state_Ug      9 ODE compartment states (mmol/kg or model units)
  dosing_problem_id          trajectory group within a file (0 = normal, 1-10 = errors)

STATE ORDER in the 10-channel training array:
  ch 0  Q1   (state_Q1,  mmol/kg)
  ch 1  Q2   (state_Q2)
  ch 2  S1   (state_S1)
  ch 3  S2   (state_S2)
  ch 4  I    (state_I)
  ch 5  X1   (state_X1)
  ch 6  X2   (state_X2)
  ch 7  X3   (state_X3)
  ch 8  C2   (state_Ug  -- gut absorption, paper notation C2)
  ch 9  C1   (zero-padded -- carbs on board, introduced by the NN; Section 2.2.2)

The 10th compartment C1 has no ground truth in the PKL data.  It is
initialized to 0 at every timestep and its MSE is masked out of L_consistency
(per Section 2.2.2).  The soft non-negativity floor (beta term) still applies.

WHAT CHANGED FROM THE OLD .mat VERSION
---------------------------------------
1. Data source: merged .mat -> individual .pkl files, split by directory.
2. State order: 9 PKL states + C1 = 0 (channel 9).
3. STATE_WEIGHTS: corrected to match paper Sec. 2.2.3 (sum = 1).
4. Train/val split: pre-assigned by directory (TRAIN / VALIDATION sub-folders).
5. Validation window overlap: 0 % (paper says "no overlap").
6. Glucose thresholds (70, 250 mg/dL) computed locally -- no t1dsim_ai.utils.preprocess.
"""

import os
import glob
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pickle import dump

from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Compatibility patch: allow unpickling files saved with older pandas (<2.0)
# ---------------------------------------------------------------------------
if "pandas.core.indexes.numeric" not in sys.modules:
    try:
        import pandas.core.indexes.numeric  # noqa: F401
    except ModuleNotFoundError:
        import pandas.core.indexes.base as _pib
        _nm = types.ModuleType("pandas.core.indexes.numeric")
        _nm.Int64Index   = pd.Index  # type: ignore[attr-defined]
        _nm.Float64Index = pd.Index  # type: ignore[attr-defined]
        _nm.UInt64Index  = pd.Index  # type: ignore[attr-defined]
        sys.modules["pandas.core.indexes.numeric"] = _nm
        setattr(_pib, "Int64Index",   pd.Index)
        setattr(_pib, "Float64Index", pd.Index)
        setattr(_pib, "UInt64Index",  pd.Index)

# t1dsim_ai submodule -- populated separately by the user
from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop


# ==============================================================================
# CONFIG  -  EDIT THESE
# ==============================================================================
SIMULATION_RESULTS_DIR = r"C:\Users\pc\Downloads\train\CodeForGeneratingDataset\simulation_results"
OUTPUT_DIR    = "models/PopulationModel/"
MODEL_FILENAME = "population_model_trained.pt"
WARM_START_CHECKPOINT = None   # path to a .pt file for warm-starting, or None

# Sub-directory patterns scanned for each split (relative to SIMULATION_RESULTS_DIR).
# Glob wildcards (*) are supported at the directory level.
TRAIN_SCAN_PATTERNS = [
    os.path.join("SIM002", "POPULATION", "TRAIN"),
    os.path.join("SIM004", "boluslate_*", "TRAIN"),
    os.path.join("SIM005", "bolusCorrectionFactor_*", "TRAIN"),
]
VAL_SCAN_PATTERNS = [
    os.path.join("SIM002", "POPULATION", "VALIDATION"),
]

# --- Hyperparameters (Sec. 2.2.3 of the paper) ---
SEQ_LEN    = 61     # M = 5 h at Dt=5 min -> 60 intervals + 1 initial state = 61 points
OVERLAP    = 0.75   # 75 % overlap between consecutive TRAINING windows
BATCH_SIZE = 128
MAX_EPOCHS = 150
PATIENCE   = 20     # early-stopping patience

MAX_TRAIN_BATCHES_PER_EPOCH = 3000   # None = all windows every epoch
MAX_VAL_BATCHES_PER_EPOCH   = 1000   # None = all windows every epoch

LR                       = 1e-3
LR_DECAY_PER_EPOCH       = np.exp(-0.1)   # gamma = e^{-0.1} per FULL data pass
SCALE_LR_DECAY_TO_FULL_EPOCH = True       # rescale gamma if epoch is capped
WEIGHT_DECAY             = 0.0

ALPHA = 0.7    # balance L_fit vs. L_consistency (Eq. 5)
BETA  = 0.08   # soft non-negativity constraint weight (Eq. 9)

SEED = 0

# --- Memory caps ---
MAX_TRAIN_SCENARIOS = 40_000   # max trajectories for TRAIN (None = unlimited)
MAX_VAL_SCENARIOS   =  8_000   # max trajectories for VAL   (None = unlimited)
SCALER_FIT_ROWS     = 4_000_000
CHUNK_ROWS          = 500_000

# --- Physical constants ---
_VDG           = 0.16    # glucose volume of distribution (L/kg) [Resalat 2019]
_MGDL_PER_MMOL = 18.0   # 1 mmol/L glucose = 18 mg/dL

# --- Dataset structure ---
T_SIM    = 2016   # 7 days x 288 samples/day at Dt=5 min
N_STATES = 10
N_INPUTS = 2
C1_INDEX = 9      # C1 channel -- no ground truth; MSE masked, soft floor active

# State channel order: [Q1, Q2, S1, S2, I, X1, X2, X3, C2(=Ug), C1]
STATE_ORDER = ["Q1", "Q2", "S1", "S2", "I", "X1", "X2", "X3", "C2", "C1"]

# PKL column -> channel mapping (channels 0-8); channel 9 (C1) is zero-filled
PKL_STATE_COLS = [
    "state_Q1", "state_Q2", "state_S1", "state_S2",
    "state_I",  "state_X1", "state_X2", "state_X3", "state_Ug",
]
PKL_INPUT_COLS = ["input_insulin", "input_meal_carbs"]   # [u_I (U/h), u_carbs (g)]

# State weights w_i (paper Sec. 2.2.3).  1-indexed paper notation -> 0-indexed channels:
#   w_{1..6, 10} = 1/12   channels 0,1,2,3,4,5,9  (Q1,Q2,S1,S2,I,X1,C1)
#   w_7          = 5/24   channel 6                 (X2)
#   w_8          = 1/6    channel 7                 (X3) [paper "0.16" = 1/6 exactly]
#   w_9          = 1/24   channel 8                 (C2 / Ug)
# Sum = 7*(1/12) + 5/24 + 1/6 + 1/24 = 14/24 + 5/24 + 4/24 + 1/24 = 24/24 = 1.0
STATE_WEIGHTS = np.array(
    [1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 5/24, 1/6, 1/24, 1/12],
    dtype=np.float32,
)
assert abs(STATE_WEIGHTS.sum() - 1.0) < 1e-5, "STATE_WEIGHTS must sum to 1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# PKL Data Loading
# ==============================================================================

def _discover_pkl_files(base_dir, patterns):
    """Return a sorted, deduplicated list of .pkl paths matching all patterns."""
    files: set = set()
    for pat in patterns:
        full_glob = os.path.join(base_dir, pat, "*.pkl")
        files.update(glob.glob(full_glob))
    return sorted(files)


def _read_pkl_trajectories(pkl_path, t_sim=T_SIM):
    """
    Read one .pkl file and extract one trajectory per dosing_problem_id.

    Returns
    -------
    states : np.ndarray  shape (n_dp, T, 10)  float32
                         channels [Q1..X3, Ug, C1=0]
    inputs : np.ndarray  shape (n_dp, T,  2)  float32
                         columns [u_I (U/h), u_carbs (g)]
    Both are None if no valid trajectories were found.
    """
    df = pd.read_pickle(pkl_path)

    traj_states, traj_inputs = [], []

    for _, grp in df.groupby("dosing_problem_id", sort=True):
        if len(grp) != t_sim:
            # Malformed or partial trajectory -- skip silently
            continue
        grp = grp.reset_index(drop=True)

        # Channels 0-8: nine ODE states from the PKL
        st9 = grp[PKL_STATE_COLS].values.astype(np.float32)      # (T, 9)

        # Channel 9: C1 -- no ground truth, zero-initialized (Section 2.2.2)
        c1 = np.zeros((t_sim, 1), dtype=np.float32)
        st = np.concatenate([st9, c1], axis=1)                    # (T, 10)

        # Inputs: [u_I (U/h), u_carbs (g)]
        u = grp[PKL_INPUT_COLS].values.astype(np.float32)         # (T, 2)

        traj_states.append(st)
        traj_inputs.append(u)

    if not traj_states:
        return None, None

    return np.stack(traj_states), np.stack(traj_inputs)           # (n_dp, T, 10/2)


def build_split_arrays(base_dir, patterns, max_scenarios=None, seed=0, t_sim=T_SIM):
    """
    Discover, optionally cap, and load all PKL trajectories for a split.

    Parameters
    ----------
    base_dir      : root of simulation_results/
    patterns      : list of sub-directory glob patterns (see TRAIN/VAL_SCAN_PATTERNS)
    max_scenarios : maximum number of trajectories to keep (None = unlimited)
    seed          : RNG seed for file shuffle -- ensures repeatable capping

    Returns
    -------
    x : np.ndarray  (N, T, 10)  float32  -- states
    u : np.ndarray  (N, T,  2)  float32  -- inputs
    """
    all_files = _discover_pkl_files(base_dir, patterns)
    if not all_files:
        raise FileNotFoundError(
            f"No .pkl files found under {base_dir!r} with patterns {patterns}.\n"
            "Have you run datasetPop.py to generate the dataset?"
        )
    print(f"  {len(all_files)} .pkl files discovered.")

    # Shuffle file order so the cap draws a representative subsample
    rng = np.random.RandomState(seed)
    file_order = rng.permutation(len(all_files))

    x_chunks, u_chunks, n_loaded = [], [], 0
    for fi in file_order:
        fpath = all_files[fi]
        st, u = _read_pkl_trajectories(fpath, t_sim)
        if st is None:
            continue
        x_chunks.append(st)
        u_chunks.append(u)
        n_loaded += st.shape[0]
        if max_scenarios is not None and n_loaded >= max_scenarios:
            break

    if not x_chunks:
        raise ValueError("No valid trajectories could be loaded from any .pkl file.")

    x_all = np.concatenate(x_chunks, axis=0)   # (N, T, 10)
    u_all = np.concatenate(u_chunks, axis=0)   # (N, T,  2)

    if max_scenarios is not None and x_all.shape[0] > max_scenarios:
        x_all = x_all[:max_scenarios]
        u_all = u_all[:max_scenarios]

    n = x_all.shape[0]
    x_gb = x_all.nbytes / 1e9
    u_gb = u_all.nbytes / 1e9
    print(f"  Loaded {n} trajectories x T={t_sim} -> states {x_gb:.2f} GB, inputs {u_gb:.2f} GB")
    return x_all, u_all


def _transform_in_place(scaler, flat_array, chunk_rows=CHUNK_ROWS):
    n = flat_array.shape[0]
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        flat_array[start:end] = scaler.transform(flat_array[start:end]).astype(np.float32)


# ==============================================================================
# Scale helpers (replaces t1dsim_ai.utils.preprocess which depends on saved files)
# ==============================================================================

def _glucose_mgdl_to_q1_scaled(glucose_mgdl, scaler_states):
    """Convert glucose (mg/dL) -> Q1 (mmol/kg) -> scaled channel 0."""
    q1_mmolkg = glucose_mgdl / _MGDL_PER_MMOL * _VDG
    return float((q1_mmolkg - scaler_states.center_[0]) / scaler_states.scale_[0])


def _q1_scaled_to_mgdl(q1_scaled_arr, scaler_states):
    """Inverse-scale channel 0 from scaled space -> Q1 (mmol/kg) -> mg/dL."""
    q1_mmolkg = q1_scaled_arr * scaler_states.scale_[0] + scaler_states.center_[0]
    return q1_mmolkg * _MGDL_PER_MMOL / _VDG


# ==============================================================================
# Framed window sampler
# ==============================================================================

class FramedWindowSampler:
    """
    Generates batches of fixed-length overlapping windows from (N, T, C) arrays.

    Training  : overlap = 0.75  (75 % overlap, Sec. 2.2.3)
    Validation: overlap = 0.0   ("sequences of size M with no overlap", Sec. 2.2.3)
    """

    def __init__(self, x_data, u_data, seq_len, overlap, batch_size, device, seed=0):
        self.x_data     = x_data
        self.u_data     = u_data
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.device     = device
        self.N, self.T, _ = x_data.shape

        self.hop = max(1, int(round((1.0 - overlap) * seq_len)))
        starts   = np.arange(0, self.T - seq_len + 1, self.hop)

        sc_grid, st_grid = np.meshgrid(np.arange(self.N), starts, indexing="ij")
        self.pairs = np.stack([sc_grid.ravel(), st_grid.ravel()], axis=1).astype(np.int32)
        del sc_grid, st_grid

        self.rng = np.random.RandomState(seed)
        self._reshuffle()

    def _reshuffle(self):
        self.epoch_order = self.rng.permutation(len(self.pairs))
        self.pos = 0

    def n_batches_per_epoch(self):
        return len(self.pairs) // self.batch_size

    def next_batch(self):
        if self.pos + self.batch_size > len(self.pairs):
            self._reshuffle()
        chosen    = self.epoch_order[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return self.batch_from_pairs(self.pairs[chosen])

    def get_fixed_subset(self, n_batches):
        """Fixed val subset, chosen once and reused every epoch for fair comparison."""
        n_needed = min(n_batches * self.batch_size, len(self.pairs))
        chosen   = self.rng.permutation(len(self.pairs))[:n_needed]
        return [self.pairs[chosen[s:s + self.batch_size]]
                for s in range(0, len(chosen), self.batch_size)]

    def batch_from_pairs(self, pair_batch):
        sc_idx    = pair_batch[:, 0]
        st_idx    = pair_batch[:, 1]
        idx_range = st_idx[:, None] + np.arange(self.seq_len)[None, :]

        x_batch = self.x_data[sc_idx[:, None], idx_range]   # (B, M, 10)
        u_batch = self.u_data[sc_idx[:, None], idx_range]   # (B, M,  2)

        # Rearrange to (M, B, C) for the simulator
        x_batch = np.transpose(x_batch, (1, 0, 2))
        u_batch = np.transpose(u_batch, (1, 0, 2))
        x0      = x_batch[0]                                 # (B, 10) initial state

        return (
            torch.tensor(x0,      dtype=torch.float32, device=self.device),
            torch.tensor(u_batch, dtype=torch.float32, device=self.device),
            torch.tensor(x_batch, dtype=torch.float32, device=self.device),
        )


# ==============================================================================
# Forward Euler simulator (Algorithm 1)
# ==============================================================================

class ForwardEulerSimulatorPop(nn.Module):
    """
    Integrates the NN state-space model using Euler's method (Algorithm 1).

        x_sim(m+Dm) = x_sim(m) + Dm * N(x_sim(m), u(m); Theta_P)

    Q1 (channel 0) is clamped to physiologically plausible glucose range
    after every step.
    """

    def __init__(self, ss_model, cgm_min_scaled, cgm_max_scaled, ts=5.0):
        super().__init__()
        self.ss_model  = ss_model
        self.ts        = ts
        self.cgm_min   = cgm_min_scaled
        self.cgm_max   = cgm_max_scaled

    def _clamp_q1(self, x):
        return torch.cat(
            [torch.clamp(x[:, [0]], min=self.cgm_min, max=self.cgm_max),
             x[:, 1:]],
            dim=1,
        )

    def forward(self, x0_batch, u_batch):
        """
        Parameters
        ----------
        x0_batch : (B, 10)   initial states (scaled)
        u_batch  : (M, B, 2) inputs over sequence of length M

        Returns
        -------
        x_list   : (M, B, 10) simulated states x_sim(0) .. x_sim(M-1)
        """
        x_step = x0_batch
        x_list = []
        for m in range(u_batch.shape[0]):
            x_list.append(x_step)
            dx     = self.ss_model(x_step, u_batch[m])
            x_step = self._clamp_q1(x_step + self.ts * dx)
        return torch.stack(x_list, dim=0)   # (M, B, 10)


# ==============================================================================
# Loss  (Equations 5 - 9 of the paper)
# ==============================================================================

class PopulationLoss:
    """
    L_total = L_fit + alpha * L_consistency                           (Eq. 5)

    L_fit   = (1/B) * (1/M) * sum_{b,m} (y_sim - y)^2 * P(y_sim, y) (Eq. 6)
    P(y_sim, y):  1 normally, 2 for hypo under-prediction,
                  6 for extreme hyper under-prediction                 (Eq. 7)

    L_consistency = sum_i  w_i * L^i_consistency                       (Eq. 8)
    L^i_consist.  = MSE_i  +  beta * sum_{b,m} max[0, -(x_sim_i - min(x_i))]
                                                                       (Eq. 9)
    MSE for C1 (channel 9) is masked out -- no ground truth.
    """

    def __init__(self, state_weights, state_min, c1_index, alpha=0.7, beta=0.08):
        self.w         = torch.tensor(state_weights, dtype=torch.float32)
        self.state_min = torch.tensor(state_min,     dtype=torch.float32)
        self.c1_index  = c1_index
        self.alpha     = alpha
        self.beta      = beta

    def to(self, device):
        self.w         = self.w.to(device)
        self.state_min = self.state_min.to(device)
        return self

    @staticmethod
    def _penalty(y_sim, y_true, lim_inf, lim_sup):
        """Eq. 7 -- P(y_sim, y) in {1, 2, 6}."""
        p = torch.ones_like(y_true)
        # Hypo: true <= 70 and model over-predicts (y_sim > y_true)
        p[torch.logical_and(y_true <= lim_inf, y_sim >  y_true)] = 2.0
        # Extreme hyper: true >= 250 and model under-predicts
        p[torch.logical_and(y_true >= lim_sup, y_sim <  y_true)] = 6.0
        return p

    def __call__(self, x_sim, x_true, lim_inf_scaled, lim_sup_scaled):
        """
        Parameters
        ----------
        x_sim, x_true : (M, B, 10)  --  step 0 is initial condition (ground truth IC),
                                         loss is computed over steps 1 .. M-1.

        Returns
        -------
        L_total         : torch scalar (differentiable)
        L_fit_val       : float (for logging)
        L_consistency_val: float (for logging)
        """
        # ---- L_fit: glucose compartment Q1 (channel 0) only ----
        y_sim  = x_sim [1:, :, [0]]   # (M-1, B, 1)
        y_true = x_true[1:, :, [0]]
        P      = self._penalty(y_sim, y_true, lim_inf_scaled, lim_sup_scaled)
        L_fit  = torch.mean((y_sim - y_true) ** 2 * P)

        # ---- L_consistency: all 10 compartments ----
        err           = x_sim[1:] - x_true[1:]                       # (M-1, B, 10)
        mse_per_state = torch.mean(err ** 2, dim=(0, 1))              # (10,)

        # Mask C1 MSE to 0 (no ground truth; Section 2.2.2)
        mse_mask = torch.ones(N_STATES, device=mse_per_state.device)
        mse_mask[self.c1_index] = 0.0
        mse_per_state = mse_per_state * mse_mask

        # Soft non-negativity floor: penalize x_sim_i < min(x_i)   (second term, Eq. 9)
        floor          = self.state_min.view(1, 1, N_STATES)
        violation      = torch.clamp(-(x_sim[1:] - floor), min=0.0)
        soft_per_state = torch.sum(violation, dim=(0, 1))             # (10,)

        per_state_loss = mse_per_state + self.beta * soft_per_state
        L_consistency  = torch.sum(self.w * per_state_loss)

        L_total = L_fit + self.alpha * L_consistency
        return L_total, L_fit.item(), L_consistency.item()


# ==============================================================================
# Main training routine
# ==============================================================================

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load TRAIN trajectories from PKL tree
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Loading TRAIN from: {SIMULATION_RESULTS_DIR}")
    print(f"Patterns: {TRAIN_SCAN_PATTERNS}")
    t0 = time.time()
    x_train_raw, u_train_raw = build_split_arrays(
        SIMULATION_RESULTS_DIR, TRAIN_SCAN_PATTERNS,
        max_scenarios=MAX_TRAIN_SCENARIOS, seed=SEED,
    )
    n_train = x_train_raw.shape[0]
    print(f"TRAIN: {n_train} trajectories  [{time.time()-t0:.1f}s]")

    # -----------------------------------------------------------------------
    # 2. Load VAL trajectories  (SIM002 / POPULATION / VALIDATION only)
    # -----------------------------------------------------------------------
    print(f"\nLoading VAL from: {SIMULATION_RESULTS_DIR}")
    print(f"Patterns: {VAL_SCAN_PATTERNS}")
    t0 = time.time()
    x_val_raw, u_val_raw = build_split_arrays(
        SIMULATION_RESULTS_DIR, VAL_SCAN_PATTERNS,
        max_scenarios=MAX_VAL_SCENARIOS, seed=SEED + 1,
    )
    n_val = x_val_raw.shape[0]
    print(f"VAL: {n_val} trajectories  [{time.time()-t0:.1f}s]")

    # -----------------------------------------------------------------------
    # 3. Fit RobustScaler on a random subsample of TRAIN rows
    # -----------------------------------------------------------------------
    print(f"\nFitting RobustScaler on up to {SCALER_FIT_ROWS:,} TRAIN rows ...")
    n_train_flat = n_train * T_SIM
    x_train_flat = x_train_raw.reshape(n_train_flat, N_STATES)
    u_train_flat = u_train_raw.reshape(n_train_flat, N_INPUTS)

    rng_fit = np.random.RandomState(SEED)
    if n_train_flat > SCALER_FIT_ROWS:
        fit_rows = np.sort(rng_fit.choice(n_train_flat, SCALER_FIT_ROWS, replace=False))
    else:
        fit_rows = np.arange(n_train_flat)

    scaler_states = RobustScaler().fit(x_train_flat[fit_rows])
    scaler_inputs = RobustScaler().fit(u_train_flat[fit_rows])
    del fit_rows

    with open(os.path.join(OUTPUT_DIR, "scaler_states.pkl"), "wb") as fh:
        dump(scaler_states, fh)
    with open(os.path.join(OUTPUT_DIR, "scaler_inputs.pkl"), "wb") as fh:
        dump(scaler_inputs, fh)
    print(f"  Scalers saved to {OUTPUT_DIR}")

    # -----------------------------------------------------------------------
    # 4. Scale TRAIN in-place  (avoids duplicating the large arrays)
    # -----------------------------------------------------------------------
    print("Scaling TRAIN ...")
    _transform_in_place(scaler_states, x_train_flat)
    _transform_in_place(scaler_inputs, u_train_flat)
    x_train = x_train_flat.reshape(n_train, T_SIM, N_STATES)
    u_train = u_train_flat.reshape(n_train, T_SIM, N_INPUTS)
    del x_train_raw, u_train_raw, x_train_flat, u_train_flat

    # -----------------------------------------------------------------------
    # 5. Scale VAL in-place
    # -----------------------------------------------------------------------
    print("Scaling VAL ...")
    n_val_flat   = n_val * T_SIM
    x_val_flat   = x_val_raw.reshape(n_val_flat, N_STATES)
    u_val_flat   = u_val_raw.reshape(n_val_flat, N_INPUTS)
    _transform_in_place(scaler_states, x_val_flat)
    _transform_in_place(scaler_inputs, u_val_flat)
    x_val = x_val_flat.reshape(n_val, T_SIM, N_STATES)
    u_val = u_val_flat.reshape(n_val, T_SIM, N_INPUTS)
    del x_val_raw, u_val_raw, x_val_flat, u_val_flat

    # -----------------------------------------------------------------------
    # 6. Per-state floor for the soft non-negativity constraint (Eq. 9)
    # -----------------------------------------------------------------------
    # floor_i = min observed scaled value across TRAIN, except for C1:
    # C1 has no ground truth (all zeros) -> physical floor is raw 0.
    # In scaled space: (0 - center_i) / scale_i  (NOT scaled 0, which = median).
    state_min = x_train.reshape(-1, N_STATES).min(axis=0)   # (10,)
    state_min[C1_INDEX] = (
        (0.0 - scaler_states.center_[C1_INDEX]) / scaler_states.scale_[C1_INDEX]
    )

    # -----------------------------------------------------------------------
    # 7. Glucose penalty thresholds in scaled Q1 space
    # -----------------------------------------------------------------------
    lim_inf_scaled = _glucose_mgdl_to_q1_scaled(70,  scaler_states)   # hypo boundary
    lim_sup_scaled = _glucose_mgdl_to_q1_scaled(250, scaler_states)   # extreme hyper
    cgm_min_scaled = _glucose_mgdl_to_q1_scaled(40,  scaler_states)   # hard clamp low
    cgm_max_scaled = _glucose_mgdl_to_q1_scaled(400, scaler_states)   # hard clamp high
    print(f"\nScaled Q1 thresholds:  70 mg/dL -> {lim_inf_scaled:.4f} | "
          f"250 mg/dL -> {lim_sup_scaled:.4f} | "
          f"clamp [{cgm_min_scaled:.4f}, {cgm_max_scaled:.4f}]")

    # -----------------------------------------------------------------------
    # 8. Build model, simulator, loss, optimizer
    # -----------------------------------------------------------------------
    print(f"\nBuilding CGMOHSUSimStateSpaceModel_V2  (n_neurons={n_neurons_pop})  on {DEVICE}")
    model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
    if WARM_START_CHECKPOINT:
        print(f"  Warm-starting from: {WARM_START_CHECKPOINT}")
        model.load_state_dict(torch.load(WARM_START_CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)

    simulator = ForwardEulerSimulatorPop(
        model, cgm_min_scaled, cgm_max_scaled, ts=5.0
    ).to(DEVICE)
    loss_fn = PopulationLoss(
        STATE_WEIGHTS, state_min, C1_INDEX, alpha=ALPHA, beta=BETA
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # -----------------------------------------------------------------------
    # 9. Window samplers
    #    Training  : 75 % overlap  (Sec. 2.2.3)
    #    Validation:  0 % overlap  ("split into sequences ... with no overlap")
    # -----------------------------------------------------------------------
    train_sampler = FramedWindowSampler(
        x_train, u_train, SEQ_LEN, overlap=OVERLAP,
        batch_size=BATCH_SIZE, device=DEVICE, seed=SEED,
    )
    val_sampler = FramedWindowSampler(
        x_val, u_val, SEQ_LEN, overlap=0.0,          # <-- NO overlap for validation
        batch_size=BATCH_SIZE, device=DEVICE, seed=SEED + 1,
    )

    full_iters_per_epoch = train_sampler.n_batches_per_epoch()
    iters_per_epoch      = full_iters_per_epoch
    val_bpe              = val_sampler.n_batches_per_epoch()
    print(f"\nWindows:  train {len(train_sampler.pairs)} ({iters_per_epoch} batches, "
          f"hop={train_sampler.hop}) | val {len(val_sampler.pairs)} "
          f"({val_bpe} batches, hop={val_sampler.hop})")

    if MAX_TRAIN_BATCHES_PER_EPOCH is not None:
        iters_per_epoch = min(iters_per_epoch, MAX_TRAIN_BATCHES_PER_EPOCH)
    if MAX_VAL_BATCHES_PER_EPOCH is not None:
        val_bpe = min(val_bpe, MAX_VAL_BATCHES_PER_EPOCH)
    print(f"Using {iters_per_epoch} train / {val_bpe} val batches per epoch.")

    # LR schedule: gamma = e^{-0.1} per FULL data pass; rescale for capped epochs
    if SCALE_LR_DECAY_TO_FULL_EPOCH and iters_per_epoch < full_iters_per_epoch:
        decay_per_epoch = LR_DECAY_PER_EPOCH ** (iters_per_epoch / full_iters_per_epoch)
    else:
        decay_per_epoch = LR_DECAY_PER_EPOCH
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_per_epoch)
    print(f"LR decay/epoch: x{decay_per_epoch:.6f}  "
          f"(final LR after {MAX_EPOCHS} epochs: {LR * decay_per_epoch**MAX_EPOCHS:.2e})")

    # Fixed validation batch set -- evaluated identically every epoch
    val_fixed = val_sampler.get_fixed_subset(val_bpe)
    print(f"Fixed validation subset: {len(val_fixed)} batches.")

    # -----------------------------------------------------------------------
    # 10. Timing probe (also serves as model warm-up)
    # -----------------------------------------------------------------------
    WARMUP_BATCHES = min(20, iters_per_epoch)
    print(f"\nTiming probe ({WARMUP_BATCHES} batches) ...")
    model.train()
    t_probe = time.time()
    for _ in range(WARMUP_BATCHES):
        optimizer.zero_grad()
        x0, u_seq, x_true = train_sampler.next_batch()
        x_sim = simulator(x0, u_seq)
        loss, _, _ = loss_fn(x_sim, x_true, lim_inf_scaled, lim_sup_scaled)
        loss.backward()
        optimizer.step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    spb = (time.time() - t_probe) / WARMUP_BATCHES
    print(f"  {spb:.3f} s/batch  ->  "
          f"~{spb * iters_per_epoch / 60:.1f} min/epoch (train) + "
          f"~{spb * val_bpe / 60:.1f} min (val)")
    train_sampler._reshuffle()   # reset position after probe

    # -----------------------------------------------------------------------
    # 11. Training loop  (Algorithm 1)
    # -----------------------------------------------------------------------
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    print(f"\n{'='*70}")
    print(f"Starting training  ({MAX_EPOCHS} epochs, early-stop patience={PATIENCE})")
    print(f"{'='*70}\n")

    for epoch in range(1, MAX_EPOCHS + 1):

        # ---- TRAIN ----
        model.train()
        train_losses = []
        t_epoch      = time.time()
        print_every  = max(1, min(50, iters_per_epoch // 10))

        for it in range(iters_per_epoch):
            optimizer.zero_grad()
            x0, u_seq, x_true = train_sampler.next_batch()
            x_sim = simulator(x0, u_seq)

            if torch.isnan(x_sim).any() or torch.isinf(x_sim).any():
                print(f"  WARNING: nan/inf at epoch {epoch} iter {it} -- batch skipped")
                continue

            loss, l_fit, l_con = loss_fn(x_sim, x_true, lim_inf_scaled, lim_sup_scaled)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if (it + 1) % print_every == 0 or (it + 1) == iters_per_epoch:
                print(f"  epoch {epoch} [train {it+1:4d}/{iters_per_epoch}]  "
                      f"loss={np.mean(train_losses):.6f}  "
                      f"L_fit={l_fit:.6f}  L_cons={l_con:.6f}  "
                      f"{time.time()-t_epoch:.1f}s")

        scheduler.step()

        # ---- VAL ----
        model.eval()
        val_losses, sq_sum, n_sq = [], 0.0, 0
        t_val     = time.time()
        pev       = max(1, min(50, len(val_fixed) // 10))

        with torch.no_grad():
            for vi, pair_batch in enumerate(val_fixed):
                x0, u_seq, x_true = val_sampler.batch_from_pairs(pair_batch)
                x_sim = simulator(x0, u_seq)
                loss, _, _ = loss_fn(x_sim, x_true, lim_inf_scaled, lim_sup_scaled)
                val_losses.append(loss.item())

                # Track RMSE in mg/dL for clinical interpretability
                y_sim_mg  = _q1_scaled_to_mgdl(
                    x_sim [1:, :, [0]].cpu().numpy(), scaler_states)
                y_true_mg = _q1_scaled_to_mgdl(
                    x_true[1:, :, [0]].cpu().numpy(), scaler_states)
                sq_sum += ((y_sim_mg - y_true_mg) ** 2).sum()
                n_sq   += y_sim_mg.size

                if (vi + 1) % pev == 0 or (vi + 1) == len(val_fixed):
                    print(f"  epoch {epoch} [val  {vi+1:4d}/{len(val_fixed)}]  "
                          f"{time.time()-t_val:.1f}s elapsed")

        mean_train = np.mean(train_losses) if train_losses else float("nan")
        mean_val   = np.mean(val_losses)   if val_losses   else float("nan")
        rmse_mg    = np.sqrt(sq_sum / n_sq) if n_sq > 0     else float("nan")
        cur_lr     = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:4d} | "
            f"train={mean_train:.6f} | val={mean_val:.6f} | "
            f"RMSE={rmse_mg:.2f} mg/dL | "
            f"lr={cur_lr:.2e} | "
            f"time={time.time()-t_epoch:.1f}s | "
            f"no_improve={epochs_no_improve}/{PATIENCE}\n"
        )

        # ---- Checkpoint ----
        if mean_val < best_val_loss:
            best_val_loss     = mean_val
            epochs_no_improve = 0
            save_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
            torch.save(model.state_dict(), save_path)
            print(f"  -> new best val_loss={best_val_loss:.6f}  saved -> {save_path}\n")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epochs_no_improve} "
                      "epochs without improvement.")
                break

    print("\nTraining complete.")
    print("Best model saved at:", os.path.join(OUTPUT_DIR, MODEL_FILENAME))


if __name__ == "__main__":
    main()
