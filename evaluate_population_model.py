"""
evaluate_population_model.py  (PKL dataset version)

Evaluates the trained population NN model (T1DSim_NN^P) on the held-out
TEST split and reproduces the population-level rows of Table 3
(Roquemen-Echeverri et al., arXiv:2508.05705):

  per-5-hour-window TIR / TAR / TBR / LBGI / HBGI / MG
  for NN-simulated vs. ODE-actual glucose, with paired TOST equivalence tests.

TEST data source (produced by datasetPop.py):
  <SIMULATION_RESULTS_DIR>/SIM002/POPULATION/TEST/*.pkl

Run after train_population_model.py (same folder, same OUTPUT_DIR).
Requires scaler_states.pkl, scaler_inputs.pkl, and the trained .pt checkpoint.
"""

import os
import sys
import types
from pickle import load

import numpy as np
import pandas as pd
import torch
import scipy.stats as st

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

# Reuse the PKL loader, simulator, sampler, and constants from the training script
from train_population_model import (
    # Data loading
    build_split_arrays,
    _transform_in_place,
    # Helpers
    _glucose_mgdl_to_q1_scaled,
    _q1_scaled_to_mgdl,
    # Classes
    FramedWindowSampler,
    ForwardEulerSimulatorPop,
    # Constants
    SIMULATION_RESULTS_DIR,
    OUTPUT_DIR,
    MODEL_FILENAME,
    SEED,
    SEQ_LEN,
    T_SIM,
    N_STATES,
    N_INPUTS,
)

# ==============================================================================
# CONFIG
# ==============================================================================

# TEST split directory (SIM002 / POPULATION / TEST only -- never seen during training)
TEST_SCAN_PATTERNS = [
    os.path.join("SIM002", "POPULATION", "TEST"),
]

EVAL_BATCH_SIZE    = 256
MAX_TEST_SCENARIOS = 10_000   # None = all (~0.9 GB for 825 trajectories)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clinical metric thresholds
TIR_LOW, TIR_HIGH = 70.0, 180.0

# TOST equivalence margins (from paper Table 3)
MARGINS = {"TIR": 5.0, "TAR": 5.0, "TBR": 1.0, "LBGI": 0.5, "HBGI": 1.0, "MG": 10.0}
ALPHA   = 0.05


# ==============================================================================
# Clinical outcome metrics  (vectorized over windows)
# ==============================================================================

def window_outcomes_batch(bg):
    """
    Compute per-window clinical metrics.

    Parameters
    ----------
    bg : np.ndarray  (Q, m)   glucose values in mg/dL, Q windows of m steps each

    Returns
    -------
    dict mapping metric name -> (Q,) float array
    """
    tir = 100.0 * np.mean((bg >= TIR_LOW) & (bg <= TIR_HIGH), axis=1)
    tar = 100.0 * np.mean(bg >  TIR_HIGH,                      axis=1)
    tbr = 100.0 * np.mean(bg <  TIR_LOW,                       axis=1)

    bgc  = np.clip(bg, 1.0, None)
    f_bg = 1.509 * (np.log(bgc) ** 1.084 - 5.381)
    r_bg = 10.0 * f_bg ** 2
    neg  = f_bg < 0
    pos  = ~neg

    lbgi = np.where(
        neg.any(axis=1),
        (r_bg * neg).sum(axis=1) / np.maximum(neg.sum(axis=1), 1),
        0.0,
    )
    hbgi = np.where(
        pos.any(axis=1),
        (r_bg * pos).sum(axis=1) / np.maximum(pos.sum(axis=1), 1),
        0.0,
    )

    return {
        "TIR": tir,
        "TAR": tar,
        "TBR": tbr,
        "LBGI": lbgi,
        "HBGI": hbgi,
        "MG":   bg.mean(axis=1),
    }


# ==============================================================================
# Paired TOST equivalence test
# ==============================================================================

def paired_tost(sim, actual, margin, alpha=ALPHA):
    """
    Two one-sided paired t-tests for equivalence within +/- margin.
    p_value = max of the two one-sided p-values (conservative, paper convention).

    Returns
    -------
    mean_sim, sd_sim, mean_act, sd_act, p_value, is_equivalent
    """
    diff      = sim - actual
    n         = len(diff)
    mean_diff = diff.mean()
    se        = diff.std(ddof=1) / np.sqrt(n)
    se        = max(se, 1e-12)   # guard against zero std

    t1 = (mean_diff + margin) / se
    p1 = 1.0 - st.t.cdf(t1, df=n - 1)

    t2 = (mean_diff - margin) / se
    p2 = st.t.cdf(t2, df=n - 1)

    p_value = max(p1, p2)
    return (
        float(sim.mean()),    float(sim.std(ddof=1)),
        float(actual.mean()), float(actual.std(ddof=1)),
        p_value,              bool(p_value < alpha),
    )


# ==============================================================================
# Main
# ==============================================================================

def main():

    # -----------------------------------------------------------------------
    # 1. Load TEST trajectories  (SIM002 / POPULATION / TEST only)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Loading TEST data from: {SIMULATION_RESULTS_DIR}")
    print(f"Patterns: {TEST_SCAN_PATTERNS}")
    x_test_raw, u_test_raw = build_split_arrays(
        SIMULATION_RESULTS_DIR, TEST_SCAN_PATTERNS,
        max_scenarios=MAX_TEST_SCENARIOS, seed=SEED,
    )
    n_test = x_test_raw.shape[0]
    print(f"TEST: {n_test} trajectories")

    # -----------------------------------------------------------------------
    # 2. Load scalers saved by train_population_model.py and scale TEST data
    # -----------------------------------------------------------------------
    scaler_states_path = os.path.join(OUTPUT_DIR, "scaler_states.pkl")
    scaler_inputs_path = os.path.join(OUTPUT_DIR, "scaler_inputs.pkl")

    if not os.path.isfile(scaler_states_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_states_path}.\n"
            "Run train_population_model.py first."
        )

    print(f"\nLoading scalers from {OUTPUT_DIR} ...")
    with open(scaler_states_path, "rb") as fh:
        scaler_states = load(fh)
    with open(scaler_inputs_path, "rb") as fh:
        scaler_inputs = load(fh)

    print("Scaling TEST data ...")
    n_test_flat  = n_test * T_SIM
    x_test_flat  = x_test_raw.reshape(n_test_flat, N_STATES)
    u_test_flat  = u_test_raw.reshape(n_test_flat, N_INPUTS)
    _transform_in_place(scaler_states, x_test_flat)
    _transform_in_place(scaler_inputs, u_test_flat)
    x_test = x_test_flat.reshape(n_test, T_SIM, N_STATES)
    u_test = u_test_flat.reshape(n_test, T_SIM, N_INPUTS)
    del x_test_raw, u_test_raw, x_test_flat, u_test_flat

    # -----------------------------------------------------------------------
    # 3. Glucose clamp thresholds in scaled Q1 space
    # -----------------------------------------------------------------------
    cgm_min_scaled = _glucose_mgdl_to_q1_scaled(40,  scaler_states)
    cgm_max_scaled = _glucose_mgdl_to_q1_scaled(400, scaler_states)

    # -----------------------------------------------------------------------
    # 4. Load trained model
    # -----------------------------------------------------------------------
    model_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}.\n"
            "Run train_population_model.py first."
        )

    print(f"\nLoading model from: {model_path}")
    model = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    simulator = ForwardEulerSimulatorPop(
        model, cgm_min_scaled, cgm_max_scaled, ts=5.0
    ).to(DEVICE)

    # -----------------------------------------------------------------------
    # 5. Build non-overlapping 5-hour window sampler for TEST
    #    Paper: "Dvalidation_P was split into sequences of size M with no overlap."
    #    We apply the same convention to the TEST split.
    # -----------------------------------------------------------------------
    sampler   = FramedWindowSampler(
        x_test, u_test, SEQ_LEN, overlap=0.0,
        batch_size=EVAL_BATCH_SIZE, device=DEVICE, seed=SEED,
    )
    all_pairs = sampler.pairs
    n_windows = len(all_pairs)
    print(f"Non-overlapping 5-hour TEST windows: {n_windows}")

    # -----------------------------------------------------------------------
    # 6. Run inference, accumulate per-window clinical metrics
    # -----------------------------------------------------------------------
    sim_chunks = {k: [] for k in MARGINS}
    act_chunks = {k: [] for k in MARGINS}
    n_skipped  = 0

    print("\nRunning inference ...")
    with torch.no_grad():
        for start in range(0, n_windows, EVAL_BATCH_SIZE):
            pair_batch = all_pairs[start:start + EVAL_BATCH_SIZE]
            x0, u_seq, x_true = sampler.batch_from_pairs(pair_batch)
            x_sim = simulator(x0, u_seq)   # (M, Q, 10), scaled

            # Inverse-scale Q1 (channel 0) back to mg/dL, then transpose to (Q, M)
            y_sim_mg  = _q1_scaled_to_mgdl(
                x_sim [:, :, [0]].cpu().numpy(), scaler_states)[..., 0].T   # (Q, M)
            y_true_mg = _q1_scaled_to_mgdl(
                x_true[:, :, [0]].cpu().numpy(), scaler_states)[..., 0].T  # (Q, M)

            valid     = np.isfinite(y_sim_mg).all(axis=1) & np.isfinite(y_true_mg).all(axis=1)
            n_skipped += int((~valid).sum())
            if not valid.any():
                continue

            y_sim_mg  = y_sim_mg [valid]
            y_true_mg = y_true_mg[valid]

            sim_out = window_outcomes_batch(y_sim_mg)
            act_out = window_outcomes_batch(y_true_mg)
            for k in MARGINS:
                sim_chunks[k].append(sim_out[k])
                act_chunks[k].append(act_out[k])

            done = min(start + EVAL_BATCH_SIZE, n_windows)
            if (start // EVAL_BATCH_SIZE) % 50 == 0:
                print(f"  {done}/{n_windows} windows processed ...")

    n_used = n_windows - n_skipped
    if n_used < 2:
        raise RuntimeError(
            "Fewer than 2 valid windows -- simulation produced NaN/inf. "
            "Check the trained model checkpoint."
        )

    # -----------------------------------------------------------------------
    # 7. Compute TOST equivalence tests and print results table
    # -----------------------------------------------------------------------
    print(f"\n{'='*95}")
    print("  POPULATION MODEL EVALUATION  --  Table 3 (Roquemen-Echeverri et al.)")
    print(f"{'='*95}")
    print(
        f"{'Metric':8s} | "
        f"{'ODE-actual (mean ± SD)':26s} | "
        f"{'NN-simulated (mean ± SD)':26s} | "
        f"{'p-value':9s} | Equivalent?"
    )
    print("-" * 95)

    for k in ["TIR", "TAR", "TBR", "LBGI", "HBGI", "MG"]:
        sim_arr = np.concatenate(sim_chunks[k])
        act_arr = np.concatenate(act_chunks[k])
        mean_sim, sd_sim, mean_act, sd_act, p, equiv = paired_tost(
            sim_arr, act_arr, MARGINS[k]
        )
        unit = "%" if k in ("TIR", "TAR", "TBR") else ("mg/dL" if k == "MG" else "")
        print(
            f"{k:8s} | "
            f"{mean_act:6.1f} +/- {sd_act:5.1f} {unit:5s}  | "
            f"{mean_sim:6.1f} +/- {sd_sim:5.1f} {unit:5s}  | "
            f"{p:9.4f} | {'YES' if equiv else 'no'}"
        )

    print("=" * 95)
    print(
        f"\nEvaluated {n_used} windows from {n_test} held-out TEST trajectories "
        f"({n_skipped} windows skipped for NaN/inf)."
    )
    print("Equivalence = p-value < 0.05 for both one-sided TOST tests.")


if __name__ == "__main__":
    main()
