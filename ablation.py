"""
ablation.py — Ablation experiments for Table 4
------------------------------------------------
Runs three training conditions on the existing data and prints a
per-string accuracy comparison table.

Conditions:
  A  13-feat   All data (clean + open + comparison), 13 features only (no MFCCs)
  B  26-feat   Clean + open sessions only, full 26 features
  C  26-feat   All data (full model — should reproduce ~97.1%)

Usage:
  py -3.11 ablation.py \
      --clean  data/strings_clean_session1.json data/strings_open_session1.json \
      --comparison data/strings_comparison_session1.json

Requirements: same as train.py
  py -3.11 -m pip install tensorflow-cpu==2.13.0 numpy scikit-learn
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# ─── Config ───────────────────────────────────────────────────────────────────

LABELS           = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
EPOCHS           = 40
BATCH_SIZE       = 32
VALIDATION_SPLIT = 0.2
SEED             = 42   # fixed seed so all three runs use the same shuffle

# ─── Feature extraction (copied verbatim from train.py) ───────────────────────

BANDS = [
    (1,   5),
    (5,   20),
    (20,  60),
    (60,  120),
    (120, 200),
    (200, 350),
    (350, 600),
    (600, 1024),
]

_N_BINS      = 1024
_SAMPLE_RATE = 44100
_N_MELS      = 40
_N_MFCC      = 13

def _hz_to_mel(hz):  return 2595.0 * np.log10(1.0 + hz / 700.0)
def _mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _build_mel_filterbank():
    fft_freqs = np.linspace(0, _SAMPLE_RATE / 2, _N_BINS)
    mel_min   = _hz_to_mel(0.0)
    mel_max   = _hz_to_mel(_SAMPLE_RATE / 2)
    mel_pts   = np.linspace(mel_min, mel_max, _N_MELS + 2)
    hz_pts    = np.array([_mel_to_hz(m) for m in mel_pts])
    filters   = np.zeros((_N_MELS, _N_BINS), dtype=np.float32)
    for m in range(_N_MELS):
        f_left, f_center, f_right = hz_pts[m], hz_pts[m+1], hz_pts[m+2]
        rising  = (fft_freqs - f_left)  / max(f_center - f_left,  1e-10)
        falling = (f_right - fft_freqs) / max(f_right  - f_center, 1e-10)
        filters[m] = np.maximum(0, np.minimum(rising, falling))
    return filters

def _build_dct_matrix():
    n   = np.arange(_N_MELS, dtype=np.float64)
    k   = np.arange(_N_MFCC, dtype=np.float64)[:, np.newaxis]
    dct = np.cos(np.pi * k * (2 * n + 1) / (2 * _N_MELS))
    dct[0]  *= 1.0 / np.sqrt(_N_MELS)
    dct[1:] *= np.sqrt(2.0 / _N_MELS)
    return dct.astype(np.float32)

_MEL_FB  = _build_mel_filterbank()
_DCT_MAT = _build_dct_matrix()


def extract_features_26(bins: np.ndarray) -> np.ndarray:
    """Full 26-feature vector (matches train.py exactly)."""
    bins         = bins.astype(np.float64)
    total_energy = bins.sum()

    band_energies = np.array([bins[lo:hi].sum() for lo, hi in BANDS], dtype=np.float64)
    if total_energy > 0:
        band_energies /= total_energy

    centroid = (np.arange(len(bins), dtype=np.float64) * bins).sum() / total_energy \
               if total_energy > 0 else 0.0
    centroid_norm = centroid / (len(bins) - 1)

    rolloff_idx = 0
    if total_energy > 0:
        cumsum = np.cumsum(bins)
        rolloff_idx = int(np.searchsorted(cumsum, 0.85 * total_energy))
    rolloff_norm = rolloff_idx / (len(bins) - 1)

    eps        = 1e-10
    arith_mean = bins.mean() + eps
    log_mean   = np.exp(np.log(bins + eps).mean())
    flatness   = float(np.clip(log_mean / arith_mean, 0.0, 1.0))

    peak_idx      = int(np.argmax(bins))
    peak_val      = float(bins[peak_idx])
    peak_idx_norm = peak_idx / (len(bins) - 1)
    peak_val_norm = peak_val / 255.0

    mel_energies = _MEL_FB @ bins.astype(np.float32)
    log_mel      = np.log(mel_energies + 1e-6)
    mfccs        = _DCT_MAT @ log_mel
    mfccs_norm   = np.clip(mfccs / 20.0, -1.0, 1.0)

    return np.array([
        *band_energies, centroid_norm, rolloff_norm, flatness,
        peak_idx_norm, peak_val_norm, *mfccs_norm,
    ], dtype=np.float32)


def extract_features_13(bins: np.ndarray) -> np.ndarray:
    """13-feature vector — band energies + spectral stats only, no MFCCs."""
    bins         = bins.astype(np.float64)
    total_energy = bins.sum()

    band_energies = np.array([bins[lo:hi].sum() for lo, hi in BANDS], dtype=np.float64)
    if total_energy > 0:
        band_energies /= total_energy

    centroid = (np.arange(len(bins), dtype=np.float64) * bins).sum() / total_energy \
               if total_energy > 0 else 0.0
    centroid_norm = centroid / (len(bins) - 1)

    rolloff_idx = 0
    if total_energy > 0:
        cumsum = np.cumsum(bins)
        rolloff_idx = int(np.searchsorted(cumsum, 0.85 * total_energy))
    rolloff_norm = rolloff_idx / (len(bins) - 1)

    eps        = 1e-10
    arith_mean = bins.mean() + eps
    log_mean   = np.exp(np.log(bins + eps).mean())
    flatness   = float(np.clip(log_mean / arith_mean, 0.0, 1.0))

    peak_idx      = int(np.argmax(bins))
    peak_val      = float(bins[peak_idx])

    return np.array([
        *band_energies, centroid_norm, rolloff_norm, flatness,
        peak_idx / (len(bins) - 1), peak_val / 255.0,
    ], dtype=np.float32)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_files(paths, feature_fn):
    X, Y = [], []
    for path in paths:
        print(f"  {path}")
        with open(path, 'r') as f:
            raw = json.load(f)
        frames = raw if isinstance(raw, list) else raw.get('frames', [])
        for frame in frames:
            data  = frame.get('data') or frame.get('frequencyData')
            label = frame.get('label')
            if data is None or label not in LABELS:
                continue
            bins = np.array(data, dtype=np.float32)
            bins = np.pad(bins, (0, max(0, 1024 - len(bins))))[:1024]
            X.append(feature_fn(bins))
            Y.append(LABELS.index(label))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)


# ─── Model builder ────────────────────────────────────────────────────────────

def build_model(input_size: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(LABELS), activation='softmax'),
    ], name='katana_vision')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def print_confusion_matrix(Y_true: np.ndarray, Y_pred: np.ndarray) -> None:
    """Print a 6×6 confusion matrix with raw counts, rows=True, cols=Pred."""
    n = len(LABELS)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(Y_true, Y_pred):
        cm[t, p] += 1

    col_w = 7  # width per cell
    header_pad = 10  # width of the row-label column

    # Header
    print(f"\n{'':>{header_pad}}", end="")
    print("  Pred →")
    print(f"{'True ↓':>{header_pad}}", end="")
    for label in LABELS:
        print(f"{label:>{col_w}}", end="")
    print()

    # Separator
    print(" " * header_pad + "-" * (col_w * n))

    # Rows
    for i, label in enumerate(LABELS):
        print(f"{label:>{header_pad}}", end="")
        for j in range(n):
            print(f"{cm[i, j]:>{col_w}}", end="")
        print()
    print()


# ─── Training run ─────────────────────────────────────────────────────────────

def run_experiment(name: str, X: np.ndarray, Y: np.ndarray,
                   confusion: bool = False) -> dict[str, float]:
    print(f"\n{'='*55}")
    print(f"  {name}  ({len(X):,} frames, {X.shape[1]} features)")
    print(f"{'='*55}")

    # Reproducible shuffle — same split across all experiments
    rng     = np.random.default_rng(SEED)
    indices = rng.permutation(len(X))
    X, Y    = X[indices], Y[indices]

    weights_arr = compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    class_weight = dict(enumerate(weights_arr))

    model = build_model(X.shape[1])

    model.fit(
        X, Y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=6,
                restore_best_weights=True, verbose=0,
            ),
        ],
        verbose=0,
    )

    # Evaluate on the held-out validation slice
    n_val  = int(len(X) * VALIDATION_SPLIT)
    X_val  = X[-n_val:]
    Y_val  = Y[-n_val:]
    Y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    results: dict[str, float] = {}
    for i, label in enumerate(LABELS):
        mask    = Y_val == i
        total   = int(mask.sum())
        correct = int((Y_pred[mask] == i).sum()) if total > 0 else 0
        acc     = correct / total if total > 0 else 0.0
        results[label] = acc
        print(f"  {label}  {correct:>5}/{total:<5}  {acc*100:5.1f}%")

    overall = (Y_pred == Y_val).mean()
    results['Overall'] = overall
    print(f"  {'Overall':6}  {int((Y_pred==Y_val).sum()):>5}/{n_val:<5}  {overall*100:5.1f}%")

    if confusion:
        print_confusion_matrix(Y_val, Y_pred)

    tf.keras.backend.clear_session()
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Ablation experiments for Table 4')
    parser.add_argument('--clean',      nargs='+', required=True,
                        metavar='FILE', help='Clean/open session JSON files')
    parser.add_argument('--comparison', nargs='+', required=True,
                        metavar='FILE', help='Comparison session JSON files')
    parser.add_argument('--skip-a', action='store_true',
                        help='Skip condition A (13-feat) to save time')
    args = parser.parse_args()

    all_files = args.clean + args.comparison

    # ── Load data ─────────────────────────────────────────────────────────────
    if not args.skip_a:
        print("\nLoading data for Experiment A (13-feat, all data)...")
        X_all_13, Y_all_13 = load_files(all_files, extract_features_13)

    print("\nLoading data for Experiment B (26-feat, clean only)...")
    X_clean_26, Y_clean = load_files(args.clean, extract_features_26)

    print("\nLoading data for Experiment C (26-feat, all data)...")
    X_all_26, Y_all_26 = load_files(all_files, extract_features_26)

    # ── Run experiments ───────────────────────────────────────────────────────
    res_A = {}
    if not args.skip_a:
        res_A = run_experiment(
            "A: 13 features, all data (no MFCCs)", X_all_13, Y_all_13,
            confusion=False)

    res_B = run_experiment(
        "B: 26 features, clean+open only (no comparison)", X_clean_26, Y_clean,
        confusion=True)

    res_C = run_experiment(
        "C: 26 features, all data (full model)", X_all_26, Y_all_26,
        confusion=True)

    # ── Print comparison table ────────────────────────────────────────────────
    cols   = LABELS + ['Overall']
    header = f"{'Condition':<42} " + "  ".join(f"{c:>6}" for c in cols)
    sep    = "-" * len(header)

    print(f"\n\n{'='*len(header)}")
    print("  ABLATION RESULTS — Table 4")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    rows = [
        ("A: 13-feat, all data",              res_A),
        ("B: 26-feat, clean+open only",        res_B),
        ("C: 26-feat, all data (full model)",  res_C),
    ]
    for label, res in rows:
        if not res:
            print(f"{'  ' + label:<42} (skipped)")
            continue
        vals = "  ".join(f"{res.get(c, 0)*100:5.1f}%" for c in cols)
        print(f"{label:<42} {vals}")

    print(sep)

    # ── CSV output ────────────────────────────────────────────────────────────
    csv_path = "ablation_results.csv"
    with open(csv_path, 'w') as f:
        f.write("Condition," + ",".join(cols) + "\n")
        for label, res in rows:
            vals = ",".join(f"{res.get(c, 0)*100:.1f}" for c in cols)
            f.write(f"{label},{vals}\n")
    print(f"\nCSV saved → {csv_path}")


if __name__ == '__main__':
    main()
