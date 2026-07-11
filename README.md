# Fretiq

**Browser-native real-time electric guitar string classification via engineered spectral features.**

## Overview

Fretiq is a real-time electric guitar string classifier that runs entirely in the browser. Given a monophonic audio signal from a guitar, the system detects the pitch using the McLeod Pitch Method and classifies which string produced it using a dense neural network trained on a 26-dimensional feature vector of engineered spectral features. No hexaphonic pickup, fretboard sensor, camera, or multi-microphone setup is required — only a USB-C audio interface and Chrome. The system renders a live 3D fretboard heatmap showing per-string confidence in real time via React Three Fiber.

## Results

- **97.1%** shuffled frame-level validation accuracy (322,215 frames, 6 strings, balanced)
- **87.8%** held-out free-play accuracy (103,000 frames, recorded after training)
- **2.07 ms** average inference latency (p95: 4.40 ms) on Intel Core Ultra 9 275HX in Chrome

## Paper

Companion arXiv preprint: [link coming soon]

## Requirements

- Boss Katana Gen 3 or any USB-C audio interface providing a clean DI signal
- Chrome browser (Web Audio API + TensorFlow.js WebGL backend)
- Node.js 18+

## Getting Started

```bash
npm install
npm run dev
```

Navigate to `http://localhost:3000`, connect your audio interface, and click **Connect Katana**.

## Architecture

| Layer | Technology |
|---|---|
| Framework | Next.js 16 (App Router, TypeScript) |
| 3D Visualization | React Three Fiber + Drei |
| ML Inference | TensorFlow.js (weights loaded from `public/model/weights.bin`) |
| Audio | Web Audio API — `getUserMedia` → `GainNode` → `AnalyserNode` |
| Pitch Detection | Pitchy (McLeod Pitch Method) |

**Feature vector (26 dimensions):**
- 8 frequency band energies (sub-bass through air)
- 5 spectral statistics (centroid, rolloff, flatness, peak index, peak value)
- 13 MFCCs (40-filter mel filterbank → log compression → orthonormal DCT-II)

**Model architecture:** `Input(26)` → `Dense(128, ReLU)` → `Dropout(0.3)` → `Dense(32, ReLU)` → `Dropout(0.2)` → `Dense(6, softmax)`

Feature extraction is implemented identically in Python (`train.py`) and TypeScript (`hooks/usePitchDetection.ts`) to guarantee training-inference parity.

## Training

The following scripts are included for reproducibility:

| Script | Purpose |
|---|---|
| `train.py` | Train the string classifier on recorded session JSON files |
| `ablation.py` | Run ablation conditions A/B/C with confusion matrix output |
| `evaluate_heldout.py` | Evaluate the deployed model on a held-out free-play recording |

**Usage:**
```bash
py -3.11 train.py --data data/session1.json --out public/model
py -3.11 ablation.py --clean data/clean.json --comparison data/comp.json
py -3.11 evaluate_heldout.py --data data/free_playing.json
```

The `data/` folder is gitignored and not included in this repository. Recording sessions are collected via the in-browser DataRecorder UI.

**Requirements:**
```bash
pip install tensorflow-cpu==2.13.0 numpy scikit-learn
```

## License

MIT
