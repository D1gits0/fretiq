"""
train.py — Katana-Vision Model Trainer
--------------------------------------
Trains a 1D CNN on your recorded guitar sessions and exports
directly to TF.js format (model.json + weights.bin) with zero
dependency on the tensorflowjs CLI.

Usage:
    py -3.11 train.py --data session1.json session2.json --out ../public/model

Requirements:
    py -3.11 -m pip install tensorflow-cpu==2.13.0 numpy scikit-learn
"""

import argparse
import json
import os
import struct
import numpy as np
import tensorflow as tf

# ─── Config ───────────────────────────────────────────────────────────────────

LABELS = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
INPUT_SIZE = 13   # engineered features, not raw bins (see extract_features)
EPOCHS = 25
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# ─── Feature extraction ───────────────────────────────────────────────────────

# Frequency band boundaries (bin indices into a 1024-bin FFT)
BANDS = [
    (1,   5),    # Sub-bass
    (5,   20),   # Bass
    (20,  60),   # Low-mid
    (60,  120),  # Mid
    (120, 200),  # Upper-mid
    (200, 350),  # Presence
    (350, 600),  # Brilliance
    (600, 1024), # Air
]

def extract_features(bins: np.ndarray) -> np.ndarray:
    """
    Transform a 1024-bin FFT frame into a 13-element feature vector.

    Features (in order):
      0–7  : energy in each of the 8 frequency bands (sum of bins, normalised)
      8    : spectral centroid (weighted mean bin index, normalised to [0, 1])
      9    : spectral rolloff  (bin below which 85% of energy falls, normalised)
      10   : spectral flatness (geometric mean / arithmetic mean)
      11   : peak bin index    (normalised to [0, 1])
      12   : peak bin value    (normalised to [0, 1])
    """
    bins = bins.astype(np.float64)
    total_energy = bins.sum()

    # ── Band energies (0–7) ───────────────────────────────────────────────
    band_energies = np.array(
        [bins[lo:hi].sum() for lo, hi in BANDS], dtype=np.float64
    )
    # Normalise by total energy so the vector is scale-invariant
    if total_energy > 0:
        band_energies /= total_energy

    # ── Spectral centroid (8) ─────────────────────────────────────────────
    if total_energy > 0:
        indices = np.arange(len(bins), dtype=np.float64)
        centroid = (indices * bins).sum() / total_energy
    else:
        centroid = 0.0
    centroid_norm = centroid / (len(bins) - 1)   # normalise to [0, 1]

    # ── Spectral rolloff (9) ──────────────────────────────────────────────
    if total_energy > 0:
        cumsum = np.cumsum(bins)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * total_energy)
    else:
        rolloff_idx = 0
    rolloff_norm = rolloff_idx / (len(bins) - 1)

    # ── Spectral flatness (10) ────────────────────────────────────────────
    # Ratio of geometric mean to arithmetic mean.
    # 1.0 = white noise (flat), 0.0 = pure tone (single spike).
    eps = 1e-10
    arith_mean = bins.mean() + eps
    # Geometric mean via log-space to avoid underflow on large arrays
    log_mean = np.exp(np.log(bins + eps).mean())
    flatness = float(np.clip(log_mean / arith_mean, 0.0, 1.0))

    # ── Peak bin (11, 12) ─────────────────────────────────────────────────
    peak_idx = int(np.argmax(bins))
    peak_val = float(bins[peak_idx])
    peak_idx_norm = peak_idx / (len(bins) - 1)
    peak_val_norm = peak_val / 255.0   # bins are 0–255

    return np.array([
        *band_energies,      # 8 values
        centroid_norm,       # 1
        rolloff_norm,        # 1
        flatness,            # 1
        peak_idx_norm,       # 1
        peak_val_norm,       # 1
    ], dtype=np.float32)     # total: 13


# ─── Load Data ────────────────────────────────────────────────────────────────

def load_sessions(paths):
    """
    Load JSON files exported from the Data Recorder UI.

    Supports two frame formats:
      New format (string recorder): { data: number[], label: string, preset: string }
      Old format (style recorder):  { data: number[], label: string }

    The 'preset' field is optional — frames without it are loaded with
    preset='unknown' so they still contribute to training.
    """
    X, Y = [], []
    preset_counts: dict[str, int] = {}

    for path in paths:
        print(f"Loading {path}...")
        with open(path, 'r') as f:
            raw = json.load(f)

        # Support both top-level array and { frames: [...] } wrapper
        frames = raw if isinstance(raw, list) else raw.get('frames', [])

        for frame in frames:
            data  = frame.get('data') or frame.get('frequencyData')
            label = frame.get('label')

            if data is None or label not in LABELS:
                continue

            preset = frame.get('preset', 'unknown')

            # Extract 13 engineered features from the raw 1024-bin FFT frame
            raw = np.array(data, dtype=np.float32)
            if len(raw) < 1024:
                raw = np.pad(raw, (0, 1024 - len(raw)))
            else:
                raw = raw[:1024]

            X.append(extract_features(raw))
            Y.append(LABELS.index(label))
            preset_counts[preset] = preset_counts.get(preset, 0) + 1

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)

    total = len(X)
    if total == 0:
        return X, Y

    print(f"\nLoaded {total:,} frames total")

    print("\nFrames per string:")
    for i, label in enumerate(LABELS):
        count = int(np.sum(Y == i))
        bar   = '█' * (count * 30 // max(total, 1))
        print(f"  {label:<4} {count:>6,}  ({count / total * 100:5.1f}%)  {bar}")

    print("\nFrames per preset:")
    for preset, count in sorted(preset_counts.items(), key=lambda x: -x[1]):
        bar = '█' * (count * 30 // max(total, 1))
        print(f"  {preset:<10} {count:>6,}  ({count / total * 100:5.1f}%)  {bar}")

    return X, Y

# ─── Build Model ──────────────────────────────────────────────────────────────

def build_model():
    """
    Small dense network trained on 13 engineered spectral features.

    Using engineered features instead of raw 1024-bin FFT vectors:
    - Dramatically reduces input dimensionality (1024 → 13)
    - Features are physically meaningful (band energy, centroid, etc.)
    - Trains faster, generalises better on small datasets
    - Runs in microseconds in the browser

    Output: softmax over 6 classes (E2, A2, D3, G3, B3, E4).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_SIZE,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(LABELS), activation='softmax'),
    ], name='katana_vision')

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

# ─── TF.js Export (no CLI needed) ────────────────────────────────────────────

def export_tfjs(model, out_dir):
    """
    Manually serialize the Keras model to TF.js Layers format.
    Produces:
      out_dir/model.json     — architecture + weight manifest
      out_dir/weights.bin    — all weights concatenated as float32 binary

    This is exactly what tf.loadLayersModel() expects in the browser.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Collect weights ───────────────────────────────────────────────────
    weight_specs = []
    weight_buffers = []

    for layer in model.layers:
        for weight in layer.weights:
            arr = weight.numpy().astype(np.float32)
            weight_specs.append({
                'name': weight.name,
                'shape': list(arr.shape),
                'dtype': 'float32',
            })
            weight_buffers.append(arr.flatten().tobytes())

    # Concatenate all weights into one binary blob
    weights_bin = b''.join(weight_buffers)
    weights_path = os.path.join(out_dir, 'weights.bin')
    with open(weights_path, 'wb') as f:
        f.write(weights_bin)

    print(f"Wrote {len(weights_bin):,} bytes → {weights_path}")

    # ── Build weight manifest ─────────────────────────────────────────────
    byte_offset = 0
    manifest_weights = []
    for spec in weight_specs:
        num_elements = int(np.prod(spec['shape'])) if spec['shape'] else 1
        byte_length = num_elements * 4  # float32 = 4 bytes
        manifest_weights.append({
            'name': spec['name'],
            'shape': spec['shape'],
            'dtype': spec['dtype'],
        })
        byte_offset += byte_length

    weight_manifest = [{
        'paths': ['weights.bin'],
        'weights': manifest_weights,
    }]

    # ── Build model config ────────────────────────────────────────────────
    keras_config = json.loads(model.to_json())

    model_json = {
        'format': 'layers-model',
        'generatedBy': 'katana-vision/train.py',
        'convertedBy': None,
        'modelTopology': keras_config,
        'weightsManifest': weight_manifest,
    }

    model_json_path = os.path.join(out_dir, 'model.json')
    with open(model_json_path, 'w') as f:
        json.dump(model_json, f, indent=2)

    print(f"Wrote model topology → {model_json_path}")
    print(f"\n✅ Export complete. Drop '{out_dir}' into your Next.js /public folder.")
    print(f"   Load in browser with: tf.loadLayersModel('/model/model.json')")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train Katana-Vision string classifier')
    parser.add_argument(
        '--data', nargs='+', required=True,
        help='Path(s) to JSON session files from the Data Recorder'
    )
    parser.add_argument(
        '--out', default='./model_output',
        help='Output directory for TF.js model files (default: ./model_output)'
    )
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help=f'Training epochs (default: {EPOCHS})'
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Katana-Vision Model Trainer")
    print("=" * 50)

    # Load + validate data
    X, Y = load_sessions(args.data)

    if len(X) < 100:
        print("\n⚠️  Warning: very few frames. Record more sessions for better accuracy.")
        print("   Aim for 2-3 minutes per string (E2, A2, D3, G3, B3, E4).\n")

    # Build + train
    model = build_model()

    # Shuffle before splitting so validation set isn't just the last recording
    # session (temporal ordering would make val_accuracy meaningless otherwise)
    indices = np.random.permutation(len(X))
    X = X[indices]
    Y = Y[indices]

    # Compute class weights to handle imbalanced string frame counts
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    class_weight = dict(enumerate(weights))

    print("\nClass weights (imbalance correction):")
    for i, label in enumerate(LABELS):
        print(f"  {label}: {class_weight.get(i, 1.0):.3f}")

    print(f"\nTraining for {args.epochs} epochs...")
    history = model.fit(
        X, Y,
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1,
            )
        ],
        verbose=1,
    )

    final_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal validation accuracy: {final_acc * 100:.1f}%")

    if final_acc < 0.7:
        print("⚠️  Accuracy is low. Try recording more varied sessions per style.")
    elif final_acc < 0.85:
        print("👍 Decent accuracy. More training data will improve this further.")
    else:
        print("🔥 Great accuracy. Model is ready.")

    # ── Per-class accuracy breakdown ──────────────────────────────────────
    # Use the held-out validation slice (last VALIDATION_SPLIT fraction after shuffle)
    n_val     = int(len(X) * VALIDATION_SPLIT)
    X_val     = X[-n_val:]
    Y_val     = Y[-n_val:]
    Y_pred    = np.argmax(model.predict(X_val, verbose=0), axis=1)

    print("\nPer-string accuracy on validation set:")
    print(f"  {'String':<8} {'Correct':>7} {'Total':>7} {'Accuracy':>9}")
    print(f"  {'-'*35}")
    for i, label in enumerate(LABELS):
        mask    = Y_val == i
        total   = int(mask.sum())
        correct = int((Y_pred[mask] == i).sum()) if total > 0 else 0
        acc     = correct / total if total > 0 else 0.0
        bar     = '█' * int(acc * 20)
        print(f"  {label:<8} {correct:>7} {total:>7} {acc * 100:>8.1f}%  {bar}")

    # Export to TF.js format
    print(f"\nExporting to TF.js format → {args.out}")
    export_tfjs(model, args.out)

if __name__ == '__main__':
    main()