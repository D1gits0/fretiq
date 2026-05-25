"""
evaluate_heldout.py — Held-out test set evaluation
----------------------------------------------------
Loads a held-out recording (e.g. free_playing.json), runs the full
26-feature extraction pipeline, loads the trained model weights from
weights.bin, and reports per-string and overall accuracy.

The model is reconstructed from weights.bin rather than a .keras file
to guarantee the evaluation uses the exact same weights deployed in the
browser — no serialization format differences.

Usage:
  py -3.11 evaluate_heldout.py \
      --data data/free_playing.json \
      --weights public/model/weights.bin

Requirements: same as train.py
  py -3.11 -m pip install tensorflow-cpu==2.13.0 numpy
"""

import argparse
import json
import struct
import numpy as np
import tensorflow as tf

# ─── Config ───────────────────────────────────────────────────────────────────

LABELS = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']

# Weight layout in weights.bin — must match train.py build_model() and
# the manual loader in usePitchDetection.ts exactly.
WEIGHT_SPECS = [
    {'shape': (26, 128)},   # dense/kernel
    {'shape': (128,)},      # dense/bias
    {'shape': (128, 32)},   # dense_1/kernel
    {'shape': (32,)},       # dense_1/bias
    {'shape': (32, 6)},     # dense_2/kernel
    {'shape': (6,)},        # dense_2/bias
]

# ─── Feature extraction (verbatim from train.py) ──────────────────────────────

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


def extract_features(bins: np.ndarray) -> np.ndarray:
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


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_model(weights_path: str) -> tf.keras.Model:
    """
    Reconstruct the model architecture and load weights from weights.bin.
    Identical to the browser loader in usePitchDetection.ts.
    """
    print(f"Loading weights from {weights_path}...")
    with open(weights_path, 'rb') as f:
        buf = f.read()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(26,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax'),
    ], name='katana_vision')

    tensors = []
    offset  = 0
    for spec in WEIGHT_SPECS:
        n_values = int(np.prod(spec['shape']))
        values   = struct.unpack_from(f'{n_values}f', buf, offset)
        tensors.append(tf.constant(np.array(values, dtype=np.float32).reshape(spec['shape'])))
        offset  += n_values * 4

    model.set_weights(tensors)
    print(f"  Loaded {offset:,} bytes, {len(tensors)} weight tensors")
    return model


# ─── Data loader ──────────────────────────────────────────────────────────────

def load_data(path: str):
    print(f"Loading {path}...")
    with open(path, 'r') as f:
        raw = json.load(f)

    frames = raw if isinstance(raw, list) else raw.get('frames', [])

    X, Y = [], []
    skipped = 0
    for frame in frames:
        data  = frame.get('data') or frame.get('frequencyData')
        label = frame.get('label')
        if data is None or label not in LABELS:
            skipped += 1
            continue
        bins = np.array(data, dtype=np.float32)
        bins = np.pad(bins, (0, max(0, 1024 - len(bins))))[:1024]
        X.append(extract_features(bins))
        Y.append(LABELS.index(label))

    if skipped:
        print(f"  Skipped {skipped} frames (missing data or unknown label)")

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def print_confusion_matrix(Y_true: np.ndarray, Y_pred: np.ndarray) -> None:
    n      = len(LABELS)
    cm     = np.zeros((n, n), dtype=int)
    for t, p in zip(Y_true, Y_pred):
        cm[t, p] += 1

    col_w      = 7
    header_pad = 10

    print(f"\n{'':>{header_pad}}  Pred →")
    print(f"{'True ↓':>{header_pad}}", end="")
    for label in LABELS:
        print(f"{label:>{col_w}}", end="")
    print()
    print(" " * header_pad + "-" * (col_w * n))
    for i, label in enumerate(LABELS):
        print(f"{label:>{header_pad}}", end="")
        for j in range(n):
            print(f"{cm[i, j]:>{col_w}}", end="")
        print()
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained model on a held-out test set')
    parser.add_argument('--data',    required=True,
                        help='Path to held-out JSON file (e.g. data/free_playing.json)')
    parser.add_argument('--weights', default='public/model/weights.bin',
                        help='Path to weights.bin (default: public/model/weights.bin)')
    args = parser.parse_args()

    # Load
    X, Y = load_data(args.data)
    print(f"  {len(X):,} frames loaded across {len(set(Y.tolist()))} classes\n")

    if len(X) == 0:
        print("No valid frames found. Check the JSON file.")
        return

    # Print class distribution
    print("Class distribution in test set:")
    for i, label in enumerate(LABELS):
        count = int(np.sum(Y == i))
        pct   = count / len(Y) * 100
        bar   = '█' * (count * 30 // max(len(Y), 1))
        print(f"  {label:<4} {count:>6,}  ({pct:5.1f}%)  {bar}")
    print()

    # Load model
    model = load_model(args.weights)

    # Predict (no dropout at inference — model.predict uses training=False by default)
    print("\nRunning inference...")
    Y_pred = np.argmax(model.predict(X, batch_size=512, verbose=0), axis=1)

    # Per-string accuracy
    print(f"\n{'='*55}")
    print(f"  HELD-OUT TEST RESULTS: {args.data}")
    print(f"{'='*55}")

    for i, label in enumerate(LABELS):
        mask    = Y == i
        total   = int(mask.sum())
        correct = int((Y_pred[mask] == i).sum()) if total > 0 else 0
        acc     = correct / total if total > 0 else 0.0
        print(f"  {label}  {correct:>5}/{total:<5}  {acc*100:5.1f}%")

    overall = (Y_pred == Y).mean()
    print(f"  {'Overall':6}  {int((Y_pred==Y).sum()):>5}/{len(Y):<5}  {overall*100:5.1f}%")

    # Confusion matrix
    print_confusion_matrix(Y, Y_pred)


if __name__ == '__main__':
    main()
