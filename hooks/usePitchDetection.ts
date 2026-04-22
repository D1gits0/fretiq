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
// same normalisation, same feature order.

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

/**
 * Transform a 1024-bin Uint8Array FFT frame into a 13-element feature vector.
 * Features: 8 band energies, centroid, rolloff, flatness, peak index, peak value.
 */
function extractFeatures(bins: Uint8Array): number[] {
  const N = bins.length;
  let totalEnergy = 0;
  for (let i = 0; i < N; i++) totalEnergy += bins[i];

  // Band energies (0–7)
  const bandEnergies: number[] = [];
  for (let b = 0; b < BANDS.length; b++) {
    const [lo, hi] = BANDS[b];
    let sum = 0;
    for (let i = lo; i < hi; i++) sum += bins[i];
    bandEnergies.push(totalEnergy > 0 ? sum / totalEnergy : 0);
  }

  // Spectral centroid (8)
  let centroid = 0;
  if (totalEnergy > 0) {
    for (let i = 0; i < N; i++) centroid += i * bins[i];
    centroid /= totalEnergy;
  }
  const centroidNorm = centroid / (N - 1);

  // Spectral rolloff at 85% (9)
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

  // Spectral flatness (10) — geometric mean / arithmetic mean in log-space
  const eps       = 1e-10;
  const arithMean = totalEnergy / N + eps;
  let logSum = 0;
  for (let i = 0; i < N; i++) logSum += Math.log(bins[i] + eps);
  const geoMean = Math.exp(logSum / N);
  const flatness = Math.min(1, Math.max(0, geoMean / arithMean));

  // Peak bin index + value (11, 12)
  let peakIdx = 0;
  let peakVal = 0;
  for (let i = 0; i < N; i++) {
    if (bins[i] > peakVal) { peakVal = bins[i]; peakIdx = i; }
  }

  return [
    ...bandEnergies,
    centroidNorm,
    rolloffNorm,
    flatness,
    peakIdx / (N - 1),
    peakVal / 255,
  ];
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

// ─── Hook ─────────────────────────────────────────────────────────────────────

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
      try {
        console.log('[usePitchDetection] Loading model from:', MODEL_URL);
        const model = await tf.loadLayersModel(MODEL_URL);
        const dummy = tf.zeros([1, 13]);
        (model.predict(dummy) as tf.Tensor).dispose();
        dummy.dispose();
        if (!cancelled) {
          modelRef.current = model;
          console.info('[usePitchDetection] String model loaded');
        }
      } catch {
        if (!cancelled) {
          modelFailedRef.current = true;
          console.info('[usePitchDetection] Model not found — using heuristic fallback');
        }
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
        // Model path — read freq-domain data and extract 13 features
        analyser.getByteFrequencyData(freqBufRef.current as Uint8Array<ArrayBuffer>);
        const features = extractFeatures(freqBufRef.current);

        const rawProbs = tf.tidy(() => {
          const input  = tf.tensor2d([features], [1, 13]);
          const output = model.predict(input) as tf.Tensor2D;
          return Array.from(output.dataSync()) as number[];
        });

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
