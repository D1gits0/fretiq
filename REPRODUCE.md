# Reproducing Fretiq Results

## 1. System Requirements

- **Python 3.11.x** (not 3.12+, not 3.14 — TensorFlow does not support these)
- **Node.js 18+**
- **Chrome browser** (tested with Web Audio API and TensorFlow.js WebGL backend)
- **A USB-C audio interface** (tested with Boss Katana Gen 3 in USB audio mode)

## 2. Running the App (Inference Only)

```bash
npm install
npm run dev
```

Navigate to `http://localhost:3000`. Connect your guitar via USB-C and click **Connect Katana**.

Model weights are included in `/public/model/weights.bin` — no training required to run inference.

## 3. Python Environment Setup

Install Python 3.11.x separately if needed. On Windows, use the `py` launcher for side-by-side versions:

```bash
py -3.11 -m pip install -r requirements.txt --break-system-packages
```

> **Note:** numpy must be pinned to 1.26.4 — numpy 2.x breaks tensorflow-cpu.

## 4. Training Data Format

Training data is **not included** in this repository (gitignored under `/data`).

Data is collected using the DataRecorder component in the running app. Each session exports a JSON file with this frame format:

```json
{
  "data": [0, 3, 12, ...],
  "label": "E2",
  "preset": "Clean"
}
```

- `data` — a 1024-element FFT magnitude array (values 0–255, from Web Audio `getByteFrequencyData`)
- `label` — one of: `E2`, `A2`, `D3`, `G3`, `B3`, `E4`
- `preset` — a string describing the amp setting (e.g. `"Clean"`, `"Crunch"`, `"Lead"`)

Frames with `mean(data) < 2.0` are discarded as silence during training.

## 5. Running Training

```bash
py -3.11 train.py --data data/session1.json data/session2.json --out public/model
```

This trains the model and exports `weights.bin` + `model.json` to the specified output directory.

## 6. Running Ablation Study

```bash
py -3.11 ablation.py \
  --clean data/strings_clean_session1.json data/strings_open_session1.json \
  --comparison data/strings_comparison_session1.json
```

Outputs per-string accuracy for three conditions (A: 13-feat, B: 26-feat no comparison, C: full model) and saves `ablation_results.csv`.

## 7. Running Held-Out Evaluation

```bash
py -3.11 evaluate_heldout.py --data data/free_playing.json
```

Loads the deployed model from `public/model/weights.bin` and evaluates on a held-out free-play recording. Reports per-string accuracy and confusion matrix.

## 8. Model Architecture

```
Input(26) → Dense(128, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(6, softmax)
```

**26 features:**
- 8 frequency band energies (sub-bass through air)
- 5 spectral statistics (centroid, rolloff, flatness, peak bin index, peak bin value)
- 13 MFCCs (40-filter mel filterbank → log compression → orthonormal DCT-II, normalized by /20 and clipped to [-1, 1])

**Trained on:** Python 3.11.9, tensorflow-cpu==2.13.0, numpy==1.26.4
