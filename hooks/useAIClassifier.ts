import { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { useStore, GuitarStyle, StyleConfidence } from '@/store/useStore';

// ─── Model path ───────────────────────────────────────────────────────────────
// Next.js serves /public statically, so this resolves to public/model/model.json.
// The file won't exist until after item 4 (train.py + tensorflowjs_converter).
const MODEL_URL = '/model/model.json';

// ─── Frequency Bin Ranges (at 44.1kHz, fftSize=2048, 1024 bins, ~21Hz/bin) ──

const BINS = {
  bass:      { lo: 2,   hi: 14  },  //   42Hz –  294Hz  (fundamentals, power chords)
  lowMid:    { lo: 14,  hi: 50  },  //  294Hz – 1050Hz  (chord body, warmth)
  highMid:   { lo: 50,  hi: 160 },  // 1050Hz – 3360Hz  (pick attack, presence)
  air:       { lo: 160, hi: 400 },  // 3360Hz – 8400Hz  (shred harmonics, shimmer)
} as const;

// ─── Config ───────────────────────────────────────────────────────────────────

/** Minimum RMS intensity to bother classifying (below = Idle) */
const SILENCE_THRESHOLD = 0.005;

/** How much to smooth confidence scores frame-to-frame (0=none, 1=frozen) */
const CONFIDENCE_SMOOTHING = 0.8;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Average energy across a slice of frequency bins, normalized to [0, 1] */
function bandEnergy(data: Uint8Array, lo: number, hi: number): number {
  let sum = 0;
  const count = hi - lo;
  for (let i = lo; i < hi; i++) sum += data[i];
  return sum / (count * 255);
}

/** Softmax so confidence scores always sum to 1 */
function softmax(scores: number[]): number[] {
  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

/** Linear interpolation for smoothing */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * (1 - t);
}

// ─── Heuristic Classifier ────────────────────────────────────────────────────

/**
 * Derives style scores from frequency band energy.
 *
 * Heuristic logic (replace with model.predict() when ready):
 *
 *  SHRED   → strong high-mid + air energy (pick attack, screaming harmonics)
 *            weak bass relative to highs (single-note runs, not chunky chords)
 *
 *  AMBIENT → low overall energy, spread evenly across bands (reverb tails,
 *            swells, clean pads — no sharp transients)
 *
 *  CHORDS  → strong bass + low-mid energy (full chord voicings, power chords)
 *            moderate highs (strumming, not shredding)
 *
 * To swap in a real TF.js model:
 *   1. Load it once:  const model = await tf.loadLayersModel('/model/model.json');
 *   2. Replace the score block below with:
 *      const tensor = tf.tensor2d([Array.from(data)]).div(255);
 *      const [shred, ambient, chords] = Array.from(model.predict(tensor).dataSync());
 */
function classifyFrequencies(data: Uint8Array): StyleConfidence {
  const bass    = bandEnergy(data, BINS.bass.lo,    BINS.bass.hi);
  const lowMid  = bandEnergy(data, BINS.lowMid.lo,  BINS.lowMid.hi);
  const highMid = bandEnergy(data, BINS.highMid.lo, BINS.highMid.hi);
  const air     = bandEnergy(data, BINS.air.lo,     BINS.air.hi);

  const shredScore   = (highMid * 1.8) + (air * 1.4) - (bass * 0.6);
  const ambientScore = (1 - (bass + highMid + air) / 3) * 1.5 + (lowMid * 0.4);
  const chordsScore  = (bass * 1.6) + (lowMid * 1.2) - (air * 0.5);

  const [shred, ambient, chords] = softmax([shredScore, ambientScore, chordsScore]);

  return { Shred: shred, Ambient: ambient, Chords: chords };
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

// ─── Manual model loader ──────────────────────────────────────────────────────
// Shared with usePitchDetection — same architecture, same weights.bin.
// See usePitchDetection.ts for full weight layout documentation.

const WEIGHT_SPECS_AC = [
  { shape: [26, 128] as [number, number] },
  { shape: [128]     as [number]         },
  { shape: [128, 32] as [number, number] },
  { shape: [32]      as [number]         },
  { shape: [32, 6]   as [number, number] },
  { shape: [6]       as [number]         },
] as const;

async function loadStringClassifier(tag: string): Promise<tf.Sequential | null> {
  try {
    const res = await fetch('/model/weights.bin');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = await res.arrayBuffer();

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [26] }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 6, activation: 'softmax' }));

    const tensors: tf.Tensor[] = [];
    let offset = 0;
    for (const spec of WEIGHT_SPECS_AC) {
      const nValues = spec.shape.reduce((a, b) => a * b, 1);
      tensors.push(tf.tensor(Array.from(new Float32Array(buf, offset, nValues)), spec.shape));
      offset += nValues * 4;
    }
    model.setWeights(tensors);
    tensors.forEach(t => t.dispose());

    const dummy = tf.zeros([1, 26]);
    (model.predict(dummy) as tf.Tensor).dispose();
    dummy.dispose();

    console.log(`[${tag}] Model ready`);
    return model;
  } catch (err) {
    console.info(`[${tag}] Model load failed:`, err);
    return null;
  }
}

/**
 * useAIClassifier
 *
 * On mount, attempts to load a trained TF.js model from MODEL_URL.
 * - If the model loads successfully, uses it for inference each frame.
 * - If the fetch fails (file not found, network error, shape mismatch),
 *   silently falls back to the heuristic classifier — no error is surfaced
 *   to the user since the heuristic is a valid working state.
 *
 * Call this once at the app root alongside useKatanaAudio.
 */
export function useAIClassifier() {
  const smoothedConfidence = useRef<StyleConfidence>({
    Shred: 0,
    Ambient: 0,
    Chords: 0,
  });
  const rafRef      = useRef<number | null>(null);
  const modelRef    = useRef<tf.LayersModel | null>(null); // set when load succeeds
  const modelFailed = useRef(false);                       // set when load fails/not found

  const setStyle      = useStore((s) => s.setStyle);
  const setConfidence = useStore((s) => s.setConfidence);

  // ── Model load (once on mount) ─────────────────────────────────────────────
  // useAIClassifier uses the same string classifier model as usePitchDetection.
  // If the model isn't present or fails to load, falls back to heuristic silently.
  useEffect(() => {
    let cancelled = false;

    (async () => {
      const model = await loadStringClassifier('useAIClassifier');
      if (!cancelled) {
        if (model) {
          modelRef.current = model;
          console.info('[useAIClassifier] Model ready — using neural classifier');
        } else {
          modelFailed.current = true;
          console.info('[useAIClassifier] Using heuristic classifier');
        }
      } else {
        model?.dispose();
      }
    })();

    return () => {
      cancelled = true;
      if (modelRef.current) {
        modelRef.current.dispose();
        modelRef.current = null;
      }
    };
  }, []);

  // ── Per-frame classification loop ──────────────────────────────────────────
  useEffect(() => {
    const tick = () => {
      const { frequencyData, intensity } = useStore.getState();

      if (frequencyData.length === 0) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      if (intensity < SILENCE_THRESHOLD) {
        // Silent — decay confidence smoothly toward zero, set Idle
        smoothedConfidence.current = {
          Shred:   lerp(smoothedConfidence.current.Shred,   0, CONFIDENCE_SMOOTHING),
          Ambient: lerp(smoothedConfidence.current.Ambient, 0, CONFIDENCE_SMOOTHING),
          Chords:  lerp(smoothedConfidence.current.Chords,  0, CONFIDENCE_SMOOTHING),
        };
        setStyle('Idle');
        setConfidence(smoothedConfidence.current);
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      // ── Classify: model if loaded, heuristic otherwise ───────────────────
      let raw: StyleConfidence;

      if (modelRef.current) {
        // Neural path — runs inside tf.tidy() so intermediate tensors are
        // automatically disposed; only the extracted JS values escape.
        const [shred, ambient, chords] = tf.tidy(() => {
          // Normalise bins to [0, 1] and reshape to (1, 1024, 1) for Conv1D
          const input = tf.tensor(Array.from(frequencyData), [1, 1024, 1], 'float32')
                          .div(255) as tf.Tensor3D;
          const output = modelRef.current!.predict(input) as tf.Tensor2D;
          // dataSync() blocks the JS thread briefly — acceptable at 60fps for
          // a 3-element output; use .data() + await if this ever becomes a bottleneck
          return Array.from(output.dataSync()) as [number, number, number];
        });
        raw = { Shred: shred, Ambient: ambient, Chords: chords };
      } else {
        // Heuristic path (model not loaded yet, or load failed)
        raw = classifyFrequencies(frequencyData);
      }

      // Smooth confidence scores to avoid jittery style flipping
      smoothedConfidence.current = {
        Shred:   lerp(smoothedConfidence.current.Shred,   raw.Shred,   CONFIDENCE_SMOOTHING),
        Ambient: lerp(smoothedConfidence.current.Ambient, raw.Ambient, CONFIDENCE_SMOOTHING),
        Chords:  lerp(smoothedConfidence.current.Chords,  raw.Chords,  CONFIDENCE_SMOOTHING),
      };

      // Dominant style = highest smoothed confidence
      const dominant = (
        Object.entries(smoothedConfidence.current) as [GuitarStyle, number][]
      ).reduce((a, b) => (b[1] > a[1] ? b : a))[0];

      setStyle(dominant);
      setConfidence({ ...smoothedConfidence.current });

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [setStyle, setConfidence]);
}