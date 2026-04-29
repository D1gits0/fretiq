# fretiq
Real-time guitar string classification and fretboard visualization in the browser.
Fretiq listens to your guitar through a USB audio interface, identifies which string and fret you're playing using a trained neural network, and lights up a 3D fretboard in real time — no specialized hardware required beyond a modern guitar amplifier with USB output.

What it does

Captures live guitar audio via Web Audio API from any USB audio interface
Detects pitch in real time using the YIN/McLeod algorithm (Pitchy)
Classifies which string is being played using a custom trained neural network (97.1% validation accuracy)
Combines pitch detection + string classification to pinpoint exact string + fret position
Renders a reactive 3D fretboard in the browser using React Three Fiber
Shows confidence heatmap across all possible note positions weighted by model certainty


Tech Stack

Next.js 16 — App Router, TypeScript
React Three Fiber + Drei — 3D fretboard rendering
Zustand — real-time state management
TensorFlow.js — in-browser neural network inference
Pitchy — real-time pitch detection (McLeod algorithm)
Web Audio API — native browser audio capture
Python + TensorFlow — model training pipeline


How it works
Guitar → USB Audio Interface → Web Audio API → FFT (1024 bins) →
26 engineered features (band energies + spectral centroid + MFCCs) →
Dense neural network → pitch-constrained string prediction →
3D fretboard visualization
The model is trained on labeled frequency data recorded string by string using the built-in Data Recorder. Each frame captures 1024 FFT bins which are transformed into 26 features including 8 frequency band energies, 5 spectral features, and 13 MFCCs computed via mel filterbank + DCT. The model never sees raw FFT data — only engineered features that capture the tonal fingerprint of each string.

Getting Started
Prerequisites:

Node.js 18+
A guitar amp or audio interface with USB output (tested on Boss Katana Gen 3)
Chrome browser (Web Audio API most reliable)

Install and run:
\`\`\`bash
git clone https://github.com/yourusername/fretiq
cd fretiq
npm install
npm run dev
\`\`\`
Open http://localhost:3000, click Connect, select your audio interface from the browser mic picker, and play.

Training your own model
The model is trained on YOUR specific guitar and audio interface. General models don't exist for this problem — the tonal fingerprint varies by instrument and signal chain.
1. Install Python dependencies:
bashpy -3.11 -m pip install tensorflow-cpu==2.13.0 numpy scikit-learn
2. Record training data:
Open the app, go to Data Recorder, toggle each string and play up and down the neck for 3-5 minutes per string. Export the JSON file.
3. Train:
bashpy -3.11 train.py --data data/your_session.json --out public/model
4. Done. The app automatically loads the new model on next page load.
For best results record multiple sessions across different amp presets and feed them all into train.py together:
bashpy -3.11 train.py --data data/clean.json data/crunch.json data/lead.json --out public/model

Project Status
Currently in active development. Real-world accuracy testing ongoing. Planned additions:

Technique detection (palm mutes, hammer-ons, slides)
Temporal modeling for context-aware string inference
Chord detection
Multi-preset model support
Research paper targeting NIME / ISMIR — Fall 2026


Background
Built by a Physics student at Cal Poly SLO as a self-directed machine learning project. Started April 20, 2026. No prior ML experience at project start.
The core research problem — identifying which guitar string produced a given pitch from a mono audio signal — is an unsolved problem in music information retrieval. Most existing solutions require hexaphonic pickups or proprietary hardware. Fretiq solves it in software using spectral feature engineering and constrained neural classification.

License
MIT
