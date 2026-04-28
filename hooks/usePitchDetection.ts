'use client';

import { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { PitchDetector } from 'pitchy';
import { useStore } from '@/store/useStore';

// ─── Tuning config ────────────────────────────────────────────────────────────
// Standard EADGBE tuning. Each entry is the MIDI note number of the open string.
// String index 0 = low E (thickest), 5 = high E (thinnest).
//
// To support alternate tunings later, replace this array and nothing else needs
// to change — all mapping logic below derives from these values.
//
// MIDI note numbers: C4 = 60, so E2 = 40, A2 = 45, D3 = 50, G3 = 55, B3 = 59, E4 = 64
const TUNING_MIDI: readonly number[] = [40, 45, 50, 55, 59, 64];

// ─── String model ─────────────────────────────────────────────────────────────
// Label order must match train.py LABELS = ['E2','A2','D3','G3','B3','E4']
const MODEL_URL = '/model/model.json';

// ─── Constants ────────────────────────────────────────────────────────────────

const FRETS       = 24;
const SAMPLE_RATE = 44100;
const FFT_SIZE    = 2048; // must match useKatanaAudio — time-domain buffer size

const CLARITY_BANDS: ReadonlyArray<{ maxHz: number; threshold: number }> = [
  { maxHz: 200,      threshold: 0.90 },
  { maxHz: 500,      threshold: 0.86 },
  { maxHz: 800,      threshold: 0.80 },
  { maxHz: Infinity, threshold: 0.72 },
];

function clarityThresholdForFreq(hz: number): number {
  for (const band of CLARITY_BANDS) {
    if (hz < band.maxHz) return band.threshold;
  }
  return CLARITY_BANDS[CLARITY_BANDS.length - 1].threshold;
}

const MIN_FREQ_HZ             = 70;
const MAX_FREQ_HZ             = 1200;
const PITCH_SILENCE_THRESHOLD = 0.01;
const NOTE_SUSTAIN_MS         = 400;
const DEBOUNCE_FRAMES         = 3;

// ─── Feature extraction ───────────────────────────────────────────────────────
// Must mirror train.py extract_features() exactly — same band boundaries,
// same normalisation, same mel filterbank construction, same DCT, same order.
// Any divergence = garbage model input.

const BANDS: ReadonlyArray<readonly [number, number]> = [
  [1,   5],    // Sub-bass
  [5,   20],   // Bass
  [20,  60],   // Low-mid
  [60,  120],  // Mid
  [120, 200],  // Upper-mid
  [200, 350],  // Presence
  [350, 600],  // Brilliance
  [600, 1024], // Air
];

// ── Mel filterbank — built once at module load, never per-frame ───────────────
// Matches train.py _build_mel_filterbank(n_bins=1024, n_mels=40, sr=44100)

const _N_BINS      = 1024;
const _SAMPLE_RATE = 44100;
const _N_MELS      = 40;
const _N_MFCC      = 13;

function _hzToMel(hz: number): number {
  return 2595.0 * Math.log10(1.0 + hz / 700.0);
}

function _melToHz(mel: number): number {
  return 700.0 * (Math.pow(10, mel / 2595.0) - 1.0);
}

/**
 * Build a Float32Array of shape (N_MELS × N_BINS) — row-major.
 * Matches train.py _build_mel_filterbank exactly:
 *   fft_freqs = linspace(0, sr/2, n_bins)   ← n_bins points, NOT n_bins+1
 *   mel points = linspace(mel(0), mel(sr/2), n_mels+2)
 *   triangular filters with rising/falling slopes
 */
function _buildMelFilterbank(): Float32Array {
  // FFT bin frequencies: 1024 points from 0 to 22050 Hz (inclusive)
  const fftFreqs = new Float32Array(_N_BINS);
  for (let i = 0; i < _N_BINS; i++) {
    fftFreqs[i] = (i / (_N_BINS - 1)) * (_SAMPLE_RATE / 2);
  }

  // n_mels+2 evenly-spaced mel points → Hz
  const melMin = _hzToMel(0.0);
  const melMax = _hzToMel(_SAMPLE_RATE / 2);
  const hzPts  = new Float32Array(_N_MELS + 2);
  for (let i = 0; i < _N_MELS + 2; i++) {
    const mel = melMin + (melMax - melMin) * (i / (_N_MELS + 1));
    hzPts[i]  = _melToHz(mel);
  }

  // Build (N_MELS × N_BINS) filter matrix, row-major
  const filters = new Float32Array(_N_MELS * _N_BINS);
  for (let m = 0; m < _N_MELS; m++) {
    const fLeft   = hzPts[m];
    const fCenter = hzPts[m + 1];
    const fRight  = hzPts[m + 2];
    const riseDiv = Math.max(fCenter - fLeft,  1e-10);
    const fallDiv = Math.max(fRight  - fCenter, 1e-10);

    for (let i = 0; i < _N_BINS; i++) {
      const f       = fftFreqs[i];
      const rising  = (f - fLeft)  / riseDiv;
      const falling = (fRight - f) / fallDiv;
      filters[m * _N_BINS + i] = Math.max(0, Math.min(rising, falling));
    }
  }
  return filters;
}

/**
 * Build a Float32Array of shape (N_MFCC × N_MELS) — row-major.
 * Matches train.py _build_dct_matrix exactly:
 *   DCT-II orthonormal: cos(pi * k * (2n+1) / (2*N))
 *   row 0 scaled by 1/sqrt(N), rows 1+ scaled by sqrt(2/N)
 */
function _buildDctMatrix(): Float32Array {
  const dct = new Float32Array(_N_MFCC * _N_MELS);
  for (let k = 0; k < _N_MFCC; k++) {
    const scale = k === 0
      ? 1.0 / Math.sqrt(_N_MELS)
      : Math.sqrt(2.0 / _N_MELS);
    for (let n = 0; n < _N_MELS; n++) {
      dct[k * _N_MELS + n] = scale * Math.cos(Math.PI * k * (2 * n + 1) / (2 * _N_MELS));
    }
  }
  return dct;
}

// Pre-computed — allocated once at module load
const _MEL_FILTERBANK = _buildMelFilterbank(); // Float32Array, shape (40 × 1024)
const _DCT_MATRIX     = _buildDctMatrix();     // Float32Array, shape (13 × 40)

/**
 * Transform a 1024-bin Uint8Array FFT frame into a 26-element feature vector.
 *
 * Features (in order, matching train.py extract_features exactly):
 *   0–7   : band energies (normalised by total energy)
 *   8     : spectral centroid (normalised to [0,1])
 *   9     : spectral rolloff at 85% (normalised to [0,1])
 *   10    : spectral flatness (geometric / arithmetic mean)
 *   11    : peak bin index (normalised to [0,1])
 *   12    : peak bin value (normalised to [0,1], bins 0–255)
 *   13–25 : 13 MFCCs (mel filterbank → log → DCT-II → /20 → clip[-1,1])
 */
function extractFeatures(bins: Uint8Array): number[] {
  const N = bins.length; // 1024
  let totalEnergy = 0;
  for (let i = 0; i < N; i++) totalEnergy += bins[i];

  // ── Band energies (0–7) ───────────────────────────────────────────────
  const bandEnergies: number[] = [];
  for (let b = 0; b < BANDS.length; b++) {
    const [lo, hi] = BANDS[b];
    let sum = 0;
    for (let i = lo; i < hi; i++) sum += bins[i];
    bandEnergies.push(totalEnergy > 0 ? sum / totalEnergy : 0);
  }

  // ── Spectral centroid (8) ─────────────────────────────────────────────
  let centroid = 0;
  if (totalEnergy > 0) {
    for (let i = 0; i < N; i++) centroid += i * bins[i];
    centroid /= totalEnergy;
  }
  const centroidNorm = centroid / (N - 1);

  // ── Spectral rolloff at 85% (9) ───────────────────────────────────────
  let rolloffIdx = 0;
  if (totalEnergy > 0) {
    const target = 0.85 * totalEnergy;
    let cumsum = 0;
    for (let i = 0; i < N; i++) {
      cumsum += bins[i];
      if (cumsum >= target) { rolloffIdx = i; break; }
    }
  }
  const rolloffNorm = rolloffIdx / (N - 1);

  // ── Spectral flatness (10) ────────────────────────────────────────────
  // Matches train.py: arith_mean = bins.mean() + eps (NOT totalEnergy/N + eps)
  // Python bins.mean() on a float64 array = sum/N, same as totalEnergy/N.
  const eps       = 1e-10;
  const arithMean = totalEnergy / N + eps;
  let logSum = 0;
  for (let i = 0; i < N; i++) logSum += Math.log(bins[i] + eps);
  const geoMean = Math.exp(logSum / N);
  const flatness = Math.min(1, Math.max(0, geoMean / arithMean));

  // ── Peak bin (11, 12) ─────────────────────────────────────────────────
  let peakIdx = 0;
  let peakVal = 0;
  for (let i = 0; i < N; i++) {
    if (bins[i] > peakVal) { peakVal = bins[i]; peakIdx = i; }
  }

  // ── MFCCs (13–25) ─────────────────────────────────────────────────────
  // Step 1: apply mel filterbank — (N_MELS × N_BINS) @ bins → (N_MELS,)
  const melEnergies = new Float32Array(_N_MELS);
  for (let m = 0; m < _N_MELS; m++) {
    let acc = 0;
    const rowOffset = m * _N_BINS;
    for (let i = 0; i < N; i++) acc += _MEL_FILTERBANK[rowOffset + i] * bins[i];
    melEnergies[m] = acc;
  }

  // Step 2: log compression — matches train.py: log(mel_energy + 1e-6)
  const logMel = new Float32Array(_N_MELS);
  for (let m = 0; m < _N_MELS; m++) {
    logMel[m] = Math.log(melEnergies[m] + 1e-6);
  }

  // Step 3: DCT-II — (N_MFCC × N_MELS) @ logMel → (N_MFCC,)
  const mfccs = new Float32Array(_N_MFCC);
  for (let k = 0; k < _N_MFCC; k++) {
    let acc = 0;
    const rowOffset = k * _N_MELS;
    for (let n = 0; n < _N_MELS; n++) acc += _DCT_MATRIX[rowOffset + n] * logMel[n];
    mfccs[k] = acc;
  }

  // Step 4: normalise — matches train.py: clip(mfccs / 20.0, -1, 1)
  const mfccsNorm: number[] = [];
  for (let k = 0; k < _N_MFCC; k++) {
    mfccsNorm.push(Math.min(1, Math.max(-1, mfccs[k] / 20.0)));
  }

  return [
    ...bandEnergies,          // 8
    centroidNorm,             // 1
    rolloffNorm,              // 1
    flatness,                 // 1
    peakIdx / (N - 1),        // 1
    peakVal / 255,            // 1
    ...mfccsNorm,             // 13
  ]; // total: 26
}

// ─── Music theory helpers ─────────────────────────────────────────────────────

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;

function freqToMidi(freq: number): number {
  return Math.round(12 * Math.log2(freq / 440) + 69);
}

function midiToNoteName(midi: number): string {
  const octave = Math.floor(midi / 12) - 1;
  return `${NOTE_NAMES[midi % 12]}${octave}`;
}

// ─── Heuristic inference (fallback when model is not loaded) ──────────────────

function allCandidates(midi: number): Array<{ string: number; fret: number }> {
  const out: Array<{ string: number; fret: number }> = [];
  for (let s = 0; s < TUNING_MIDI.length; s++) {
    const fret = midi - TUNING_MIDI[s];
    if (fret >= 0 && fret <= FRETS) out.push({ string: s, fret });
  }
  return out;
}

function proximityScore(
  c: { string: number; fret: number },
  prevString: number,
  prevFret: number,
): number {
  if (prevString < 0 || prevFret < 0) return 0.5;
  const fd = Math.abs(c.fret   - prevFret);
  const sd = Math.abs(c.string - prevString);
  const fp = fd > 5 ? Math.min(1, (fd - 5) / 7) : 0;
  const sp = sd > 2 ? Math.min(1, (sd - 2) / 3) : 0;
  return 1 - (fp * 0.6 + sp * 0.4);
}

function spectralTilt(buffer: Float32Array): number {
  const half = Math.floor(buffer.length / 2);
  let lo = 0, hi = 0;
  for (let i = 0; i < half; i++)               lo += buffer[i] * buffer[i];
  for (let i = half; i < buffer.length; i++)   hi += buffer[i] * buffer[i];
  const total = lo + hi;
  return total < 1e-10 ? 0.5 : lo / total;
}

function thicknessScore(stringIdx: number, tilt: number): number {
  const norm = (TUNING_MIDI.length - 1 - stringIdx) / (TUNING_MIDI.length - 1);
  return 1 - Math.abs(norm - tilt);
}

function inferCandidate(
  midi: number,
  buffer: Float32Array,
  frequency: number,
  clarity: number,
  prevString: number,
  prevFret: number,
): { string: number; fret: number; stringConfidence: number } {
  const candidates = allCandidates(midi);
  if (candidates.length === 0) return { string: -1, fret: -1, stringConfidence: 0 };
  if (candidates.length === 1) return { ...candidates[0], stringConfidence: 1.0 };

  const threshold = clarityThresholdForFreq(frequency);
  const greyZone  = clarity - threshold < 0.05;
  const tilt      = greyZone ? spectralTilt(buffer) : 0.5;

  const scored = candidates.map((c) => {
    const prox  = proximityScore(c, prevString, prevFret);
    const thick = greyZone ? thicknessScore(c.string, tilt) : 0.5;
    return { ...c, score: greyZone ? thick * 0.4 + prox * 0.6 : prox };
  });
  scored.sort((a, b) => b.score - a.score);

  const gap = scored[0].score - scored[1].score;
  return {
    string: scored[0].string,
    fret:   scored[0].fret,
    stringConfidence: Math.min(1, gap / 0.3),
  };
}

// ─── Manual model loader ──────────────────────────────────────────────────────
// tf.loadLayersModel() cannot parse the Keras 2 SavedModel format produced by
// train.py's custom export. Instead we reconstruct the architecture in TF.js
// and load the raw weights from weights.bin directly.
//
// Weight layout in weights.bin (float32, 4 bytes each):
//   dense/kernel   [26, 128]  → 3328 values → 13312 bytes  @ offset 0
//   dense/bias     [128]      →  128 values →   512 bytes  @ offset 13312
//   dense_1/kernel [128, 32]  → 4096 values → 16384 bytes  @ offset 13824
//   dense_1/bias   [32]       →   32 values →   128 bytes  @ offset 30208
//   dense_2/kernel [32, 6]    →  192 values →   768 bytes  @ offset 30336
//   dense_2/bias   [6]        →    6 values →    24 bytes  @ offset 31104
//   Total: 31128 bytes

const WEIGHT_SPECS = [
  { shape: [26, 128] as [number, number] },  // dense/kernel
  { shape: [128]     as [number]         },  // dense/bias
  { shape: [128, 32] as [number, number] },  // dense_1/kernel
  { shape: [32]      as [number]         },  // dense_1/bias
  { shape: [32, 6]   as [number, number] },  // dense_2/kernel
  { shape: [6]       as [number]         },  // dense_2/bias
] as const;

async function loadStringClassifier(
  tag: string,
): Promise<tf.Sequential | null> {
  try {
    console.log(`[${tag}] Fetching /model/weights.bin`);
    const res = await fetch('/model/weights.bin');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    console.log(`[${tag}] weights.bin size: ${buf.byteLength} bytes`);

    // Build the model architecture — must match train.py build_model() exactly
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [26] }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 6, activation: 'softmax' }));

    // Parse weights.bin into tensors at the correct byte offsets
    const tensors: tf.Tensor[] = [];
    let offset = 0;
    for (const spec of WEIGHT_SPECS) {
      const nValues = spec.shape.reduce((a, b) => a * b, 1);
      const slice   = new Float32Array(buf, offset, nValues);
      tensors.push(tf.tensor(Array.from(slice), spec.shape));
      offset += nValues * 4;
    }

    // setWeights only accepts the trainable weight tensors (no dropout layers)
    model.setWeights(tensors);
    tensors.forEach(t => t.dispose());

    // Warm up — first predict is slow due to graph compilation
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

export function usePitchDetection(getAnalyser: () => AnalyserNode | null) {
  const rafRef      = useRef<number | null>(null);
  const bufferRef   = useRef<Float32Array>(new Float32Array(FFT_SIZE));  // time-domain
  const freqBufRef  = useRef<Uint8Array>(new Uint8Array(1024));          // freq-domain
  const detectorRef = useRef<PitchDetector<Float32Array>>(
    PitchDetector.forFloat32Array(FFT_SIZE),
  );

  // null = loading, false = failed, LayersModel = ready
  const modelRef       = useRef<tf.LayersModel | null>(null);
  const modelFailedRef = useRef(false);
  const featuresLoggedRef = useRef(false); // fires the feature debug log once

  const lastValidDetectionRef   = useRef<number>(0);
  const committedMidiRef        = useRef<number>(-1);
  const committedTimeRef        = useRef<number>(0);
  const pendingMidiRef          = useRef<number>(-1);
  const pendingCountRef         = useRef<number>(0);
  const historyRef              = useRef<Array<{ string: number; fret: number }>>([]);
  const pendingStringRef        = useRef<number>(-1);
  const pendingStringCountRef   = useRef<number>(0);
  const committedStringRef      = useRef<number>(-1);

  const setNote      = useStore((s) => s.setNote);
  const setChordName = useStore((s) => s.setChordName);

  // ── Load string classifier on mount ───────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const model = await loadStringClassifier('usePitchDetection');
      if (!cancelled) {
        if (model) {
          modelRef.current = model;
        } else {
          modelFailedRef.current = true;
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

  // ── Per-frame pitch detection loop ────────────────────────────────────────
  useEffect(() => {
    const tick = () => {
      const analyser = getAnalyser();
      if (!analyser) { rafRef.current = requestAnimationFrame(tick); return; }

      const buffer   = bufferRef.current;
      const detector = detectorRef.current;
      const now      = performance.now();

      analyser.getFloatTimeDomainData(buffer as Float32Array<ArrayBuffer>);

      // RMS silence check
      let sumSq = 0;
      for (let i = 0; i < buffer.length; i++) sumSq += buffer[i] * buffer[i];
      const rms = Math.sqrt(sumSq / buffer.length);

      if (rms < PITCH_SILENCE_THRESHOLD) {
        pendingMidiRef.current        = -1;
        pendingCountRef.current       = 0;
        lastValidDetectionRef.current = 0;
        pendingStringRef.current      = -1;
        pendingStringCountRef.current = 0;
        committedStringRef.current    = -1;
        committedMidiRef.current      = -1;
        committedTimeRef.current      = 0;
        setNote('', 0, -1, -1, 0);
        setChordName('');
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      const [frequency, clarity] = detector.findPitch(buffer, SAMPLE_RATE);
      const clarityThreshold     = clarityThresholdForFreq(frequency);
      const isValid =
        clarity >= clarityThreshold &&
        frequency > 0 &&
        frequency >= MIN_FREQ_HZ &&
        frequency <= MAX_FREQ_HZ;

      if (!isValid) {
        pendingMidiRef.current        = -1;
        pendingCountRef.current       = 0;
        pendingStringRef.current      = -1;
        pendingStringCountRef.current = 0;

        const { intensity }      = useStore.getState();
        const stringStillRinging = intensity >= 0.02;
        if (!stringStillRinging || now - lastValidDetectionRef.current > NOTE_SUSTAIN_MS) {
          committedMidiRef.current   = -1;
          committedTimeRef.current   = 0;
          committedStringRef.current = -1;
          setNote('', 0, -1, -1, 0);
          setChordName('');
        }
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      lastValidDetectionRef.current = now;

      // ── Per-frame model debug log ─────────────────────────────────────────
      // Runs every valid pitch frame — before debounce or note lock — so the
      // softmax output is visible in the console while a note is ringing.
      if (modelRef.current) {
        analyser.getByteFrequencyData(freqBufRef.current as Uint8Array<ArrayBuffer>);
        const debugFeatures = extractFeatures(freqBufRef.current);
        const predictionData = tf.tidy(() => {
          const input  = tf.tensor2d([debugFeatures], [1, 26]);
          const output = modelRef.current!.predict(input) as tf.Tensor2D;
          return Array.from(output.dataSync());
        });
        console.log('softmax:', predictionData.map(n => n.toFixed(3)));
      }

      // Debounce — require DEBOUNCE_FRAMES consecutive frames of the same MIDI
      const midi = freqToMidi(frequency);
      if (midi === pendingMidiRef.current) {
        pendingCountRef.current += 1;
      } else {
        pendingMidiRef.current  = midi;
        pendingCountRef.current = 1;
      }
      if (pendingCountRef.current < DEBOUNCE_FRAMES) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      // Note lock — don't switch notes until the committed one has been absent
      if (
        committedMidiRef.current !== -1 &&
        midi !== committedMidiRef.current &&
        now - committedTimeRef.current < NOTE_SUSTAIN_MS
      ) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      committedMidiRef.current = midi;
      committedTimeRef.current = now;

      const note    = midiToNoteName(midi);
      const history = historyRef.current;
      const prev    = history.length > 0 ? history[history.length - 1] : { string: -1, fret: -1 };

      // ── String inference ─────────────────────────────────────────────────
      let inferredString: number;
      let fret: number;
      let stringConfidence: number;
      let stringProbs: number[] = [0, 0, 0, 0, 0, 0];

      const model = modelRef.current;

      if (model) {
        // Model path — read freq-domain data and extract 26 features
        analyser.getByteFrequencyData(freqBufRef.current as Uint8Array<ArrayBuffer>);
        const features = extractFeatures(freqBufRef.current);

        // Debug: log features once so you can compare with train.py output
        if (!featuresLoggedRef.current) {
          featuresLoggedRef.current = true;
          console.log('[usePitchDetection] 26 features for this frame:');
          console.log('  bands(0-7):', features.slice(0, 8).map(v => v.toFixed(4)));
          console.log('  centroid(8):', features[8].toFixed(4));
          console.log('  rolloff(9):', features[9].toFixed(4));
          console.log('  flatness(10):', features[10].toFixed(4));
          console.log('  peakIdx(11):', features[11].toFixed(4));
          console.log('  peakVal(12):', features[12].toFixed(4));
          console.log('  mfccs(13-25):', features.slice(13, 26).map(v => v.toFixed(4)));
          console.log('  total features:', features.length);
        }

        const rawProbs = tf.tidy(() => {
          const input  = tf.tensor2d([features], [1, 26]);
          const output = model.predict(input) as tf.Tensor2D;
          return Array.from(output.dataSync()) as number[];
        });

        console.log('raw model output:', rawProbs.map(v => v.toFixed(4)));

        // ── Constrain to physically valid strings ──────────────────────────
        const masked = rawProbs.map((score, s) => {
          const candidateFret = midi - TUNING_MIDI[s];
          if (candidateFret < 0 || candidateFret > FRETS) return 0;
          let penalty = 1.0;
          if (midi > 52 && s === 0) penalty *= 0.3;
          if (midi > 59 && s === 1) penalty *= 0.3;
          if (midi > 64 && s === 2) penalty *= 0.3;
          return score * penalty;
        });

        const maskedSum = masked.reduce((a, b) => a + b, 0);

        if (maskedSum > 0) {
          const normalised    = masked.map((s) => s / maskedSum);
          const predStringIdx = normalised.indexOf(Math.max(...normalised));
          inferredString   = predStringIdx;
          fret             = midi - TUNING_MIDI[predStringIdx];
          stringConfidence = normalised[predStringIdx];
          stringProbs      = normalised; // full per-string distribution for heatmap
        } else {
          const h          = inferCandidate(midi, buffer, frequency, clarity, prev.string, prev.fret);
          inferredString   = h.string;
          fret             = h.fret;
          stringConfidence = h.stringConfidence * 0.5;
          // Synthesise a uniform distribution over valid candidates for heatmap
          const candidates: number[] = TUNING_MIDI.map((open, s) => {
            const f = midi - open;
            return (f >= 0 && f <= FRETS) ? 1 : 0;
          });
          const cSum = candidates.reduce((a, b) => a + b, 0);
          stringProbs = candidates.map((v) => cSum > 0 ? v / cSum : 0);
        }
      } else {
        // Heuristic fallback
        const h          = inferCandidate(midi, buffer, frequency, clarity, prev.string, prev.fret);
        inferredString   = h.string;
        fret             = h.fret;
        stringConfidence = h.stringConfidence;
        // Synthesise probs: winner gets confidence, rest share remainder
        stringProbs = TUNING_MIDI.map((open, s) => {
          const f = midi - open;
          if (f < 0 || f > FRETS) return 0;
          return s === inferredString ? h.stringConfidence : (1 - h.stringConfidence) / 5;
        });
      }

      // ── String stability gate ─────────────────────────────────────────────
      // Require 4 consecutive frames before committing a new string.
      // Once committed, require 6 consecutive frames of a different string
      // before switching — prevents bouncing while a note rings.
      const framesNeeded = committedStringRef.current >= 0 &&
                           committedStringRef.current !== inferredString
        ? 6  // switching away from a committed string — higher bar
        : 4; // initial commit — lower bar

      if (inferredString === pendingStringRef.current) {
        pendingStringCountRef.current += 1;
      } else {
        pendingStringRef.current      = inferredString;
        pendingStringCountRef.current = 1;
      }

      let string: number;
      if (pendingStringCountRef.current >= framesNeeded) {
        committedStringRef.current = inferredString;
        string = inferredString;
      } else {
        string = committedStringRef.current >= 0 ? committedStringRef.current : inferredString;
      }

      if (string >= 0) {
        history.push({ string, fret });
        if (history.length > 3) history.shift();
      }

      setNote(note, frequency, string, fret, stringConfidence, stringProbs);
      setChordName('');

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current !== null) cancelAnimationFrame(rafRef.current); };
  }, [getAnalyser, setNote, setChordName]);
}
