"""
train.py — Katana-Vision Model Trainer
--------------------------------------
Trains a 1D CNN on your recorded guitar sessions and exports
directly to TF.js format (model.json + weights.bin) with zero
dependency on the tensorflowjs CLI.

Usage:
    py -3.11 train.py --data session1.json session2.json --out ../public/model

Requirements:
    py -3.11 -m pip install tensorflow-cpu==2.13.0 numpy
"""

import argparse
import json
import os
import struct
import numpy as np
import tensorflow as tf

# ─── Config ───────────────────────────────────────────────────────────────────

LABELS = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
INPUT_SIZE = 1024
EPOCHS = 25
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

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

            # Pad or truncate to INPUT_SIZE
            arr = np.array(data, dtype=np.float32)
            if len(arr) < INPUT_SIZE:
                arr = np.pad(arr, (0, INPUT_SIZE - len(arr)))
            else:
                arr = arr[:INPUT_SIZE]

            X.append(arr)
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
    Small 1D CNN. Input: 1024 normalized frequency bins.
    Output: softmax over 6 classes (E2, A2, D3, G3, B3, E4).

    Deliberately small — trains in minutes on CPU, runs in <1ms on browser.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_SIZE,)),
        tf.keras.layers.Reshape((INPUT_SIZE, 1)),

        # First conv block — catches broad frequency patterns
        tf.keras.layers.Conv1D(32, kernel_size=8, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Dropout(0.2),

        # Second conv block — catches finer tonal details
        tf.keras.layers.Conv1D(64, kernel_size=4, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Dropout(0.2),

        # Global pooling — collapses frequency axis
        tf.keras.layers.GlobalAveragePooling1D(),

        # Dense head
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
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

    print(f"\nTraining for {args.epochs} epochs...")
    history = model.fit(
        X, Y,
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
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

    # Export to TF.js format
    print(f"\nExporting to TF.js format → {args.out}")
    export_tfjs(model, args.out)

if __name__ == '__main__':
    main()