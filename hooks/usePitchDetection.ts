'use client';

import { useEffect, useRef } from 'react';
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

// ─── Constants ────────────────────────────────────────────────────────────────

const FRETS       = 24;
const SAMPLE_RATE = 44100;
const FFT_SIZE    = 2048; // must match useKatanaAudio — time-domain buffer size

/**
 * Frequency-scaled clarity threshold.
 * McLeod clarity scores drop naturally at higher frequencies due to fewer
 * complete cycles fitting in the analysis window. Each band reflects the
 * minimum clarity a clean guitar note produces at that frequency range.
 */
const CLARITY_BANDS: ReadonlyArray<{ maxHz: number; threshold: number }> = [
  { maxHz: 200,       threshold: 0.90 },
  { maxHz: 500,       threshold: 0.86 },
  { maxHz: 800,       threshold: 0.80 },
  { maxHz: Infinity,  threshold: 0.72 },
];

function clarityThresholdForFreq(hz: number): number {
  for (const band of CLARITY_BANDS) {
    if (hz < band.maxHz) return band.threshold;
  }
  return CLARITY_BANDS[CLARITY_BANDS.length - 1].threshold;
}

/** Practical guitar frequency range — reject detections outside this window */
const MIN_FREQ_HZ = 70;
const MAX_FREQ_HZ = 1200;

/** Minimum RMS intensity before we bother running pitch detection */
const PITCH_SILENCE_THRESHOLD = 0.01;

/**
 * How long (ms) to hold the last valid note after clarity drops below threshold.
 * Sustain only applies while intensity >= 0.02 (string still ringing).
 */
const NOTE_SUSTAIN_MS = 400;

/**
 * Number of consecutive frames a pitch must be detected before it is committed
 * to the store. Filters out single-frame transient detections on pick attack.
 */
const DEBOUNCE_FRAMES = 3;

// ─── Music theory helpers ─────────────────────────────────────────────────────

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;

/** Convert a frequency in Hz to the nearest MIDI note number */
function freqToMidi(freq: number): number {
  return Math.round(12 * Math.log2(freq / 440) + 69);
}

/** Convert a MIDI note number to a human-readable note name, e.g. 40 → 'E2' */
function midiToNoteName(midi: number): string {
  const octave = Math.floor(midi / 12) - 1;
  const name   = NOTE_NAMES[midi % 12];
  return `${name}${octave}`;
}

/**
 * Map a MIDI note to the best string + fret position in the current tuning.
 *
 * Strategy: prefer the highest-numbered string (thinnest) that can play the
 * note within FRETS frets. Iterates from high E (5) down to low E (0).
 *
 * This matches real playing behaviour — a guitarist plays a high note on the
 * thinnest string that can reach it, not buried on a thick wound string.
 * e.g. fret 19 on high E (MIDI 83) → string 5, not fret 22 on B string.
 *
 * Returns { string: -1, fret: -1 } if the note is out of range for all strings.
 */
function midiToStringFret(midi: number): { string: number; fret: number } {
  for (let s = TUNING_MIDI.length - 1; s >= 0; s--) {
    const fret = midi - TUNING_MIDI[s];
    if (fret >= 0 && fret <= FRETS) return { string: s, fret };
  }
  return { string: -1, fret: -1 };
}

// ─── Soft inference helpers ───────────────────────────────────────────────────

/** All valid (string, fret) candidates for a MIDI note — fret bounds enforced */
function allCandidates(midi: number): Array<{ string: number; fret: number }> {
  const out: Array<{ string: number; fret: number }> = [];
  for (let s = 0; s < TUNING_MIDI.length; s++) {
    const fret = midi - TUNING_MIDI[s];
    if (fret >= 0 && fret <= FRETS) out.push({ string: s, fret });
  }
  return out;
}

/**
 * Proximity score for a candidate relative to the previous note position.
 * Returns 1.0 when at the same position, decreasing with fret/string distance.
 * Penalties kick in above 5 frets or 2 strings.
 */
function proximityScore(
  c: { string: number; fret: number },
  prevString: number,
  prevFret: number,
): number {
  if (prevString < 0 || prevFret < 0) return 0.5; // no history — neutral
  const fd = Math.abs(c.fret   - prevFret);
  const sd = Math.abs(c.string - prevString);
  const fp = fd > 5 ? Math.min(1, (fd - 5) / 7) : 0;
  const sp = sd > 2 ? Math.min(1, (sd - 2) / 3) : 0;
  return 1 - (fp * 0.6 + sp * 0.4);
}

/**
 * Spectral tilt score — low/high energy ratio of the time-domain buffer.
 * Returns a "thickness" value in [0, 1]:
 *   > 0.5 → more low-frequency energy → likely thicker string
 *   < 0.5 → more high-frequency energy → likely thinner string
 */
function spectralTilt(buffer: Float32Array): number {
  const half = Math.floor(buffer.length / 2);
  let lo = 0, hi = 0;
  for (let i = 0; i < half; i++)          lo += buffer[i] * buffer[i];
  for (let i = half; i < buffer.length; i++) hi += buffer[i] * buffer[i];
  const total = lo + hi;
  return total < 1e-10 ? 0.5 : lo / total;
}

/**
 * String thickness score — how well a string index matches the measured tilt.
 * String 0 (low E) = thickest = 1.0, string 5 (high E) = thinnest = 0.0.
 */
function thicknessScore(stringIdx: number, tilt: number): number {
  const norm = (TUNING_MIDI.length - 1 - stringIdx) / (TUNING_MIDI.length - 1);
  return 1 - Math.abs(norm - tilt);
}

/**
 * Pick the best string+fret candidate using a two-tier soft inference.
 *
 * HIGH clarity (> threshold + 0.05): proximity bias only.
 * GREY ZONE (within 0.05 of threshold): spectral tilt (40%) + proximity (60%).
 *
 * If only one candidate exists, return it directly with confidence 1.
 * Confidence = normalised score gap between winner and runner-up (0–1).
 */
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

  const threshold  = clarityThresholdForFreq(frequency);
  const greyZone   = clarity - threshold < 0.05; // within 0.05 of threshold
  const tilt       = greyZone ? spectralTilt(buffer) : 0.5; // only compute when needed

  const scored = candidates.map((c) => {
    const prox  = proximityScore(c, prevString, prevFret);
    const thick = greyZone ? thicknessScore(c.string, tilt) : 0.5;
    // Grey zone: blend tilt (40%) + proximity (60%)
    // High clarity: proximity only (tilt weight = 0)
    const score = greyZone
      ? thick * 0.4 + prox * 0.6
      : prox;
    return { ...c, score };
  });

  scored.sort((a, b) => b.score - a.score);

  const gap        = scored[0].score - scored[1].score;
  const confidence = Math.min(1, gap / 0.3);

  return { string: scored[0].string, fret: scored[0].fret, stringConfidence: confidence };
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

/**
 * usePitchDetection
 *
 * Reads the AnalyserNode from useKatanaAudio (passed in as a getter), runs
 * pitchy's McLeod Pitch Method on each time-domain frame, and writes the
 * detected note, frequency, string, and fret to the Zustand store.
 *
 * Call this once at the app root, after useKatanaAudio:
 *   const { getAnalyser } = useKatanaAudio();
 *   usePitchDetection(getAnalyser);
 *
 * @param getAnalyser - stable callback ref from useKatanaAudio that returns
 *                      the live AnalyserNode, or null if not yet connected.
 */
export function usePitchDetection(getAnalyser: () => AnalyserNode | null) {
  const rafRef      = useRef<number | null>(null);
  // Reusable float32 buffer — allocated once, sized to FFT_SIZE
  const bufferRef   = useRef<Float32Array>(new Float32Array(FFT_SIZE));
  // PitchDetector instance — created once, reused every frame
  const detectorRef = useRef<PitchDetector<Float32Array>>(
    PitchDetector.forFloat32Array(FFT_SIZE)
  );

  // ── Sustain: timestamp of the last frame that produced a valid detection ──
  const lastValidDetectionRef = useRef<number>(0);

  // ── Note lock: once a MIDI note is committed, hold it until either:
  //    (a) a different note has been consistently detected for DEBOUNCE_FRAMES, AND
  //    (b) the committed note has been absent for NOTE_SUSTAIN_MS
  // This prevents a ringing note from bouncing when pitch drifts slightly.
  const committedMidiRef    = useRef<number>(-1); // currently displayed MIDI note
  const committedTimeRef    = useRef<number>(0);  // when the committed note was last seen

  // ── Debounce: track consecutive frames of the same MIDI note ─────────────
  const pendingMidiRef  = useRef<number>(-1);
  const pendingCountRef = useRef<number>(0);

  // ── History: last 3 committed (string, fret) positions for context ────────
  const historyRef = useRef<Array<{ string: number; fret: number }>>([]);

  // ── String stability gate ─────────────────────────────────────────────────
  // A new string assignment only commits after 2 consecutive frames agree.
  // Prevents single-frame flicker between string candidates.
  const pendingStringRef      = useRef<number>(-1); // string candidate from last frame
  const pendingStringCountRef = useRef<number>(0);  // consecutive agreement count
  const committedStringRef    = useRef<number>(-1); // last confirmed string

  const setNote      = useStore((s) => s.setNote);
  const setChordName = useStore((s) => s.setChordName);

  useEffect(() => {
    const tick = () => {
      const analyser = getAnalyser();

      if (!analyser) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      const buffer   = bufferRef.current;
      const detector = detectorRef.current;
      const now      = performance.now();

      // Get time-domain waveform (float, -1 to +1)
      analyser.getFloatTimeDomainData(buffer as Float32Array<ArrayBuffer>);

      // Quick RMS check — skip pitch detection when silent to save CPU
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

      // Run McLeod Pitch Method
      const [frequency, clarity] = detector.findPitch(buffer, SAMPLE_RATE);

      // Use a frequency-scaled clarity threshold — McLeod scores drop at
      // higher frequencies, so each band has its own minimum.
      const clarityThreshold = clarityThresholdForFreq(frequency);

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

        const { intensity } = useStore.getState();
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

      // Valid detection — update sustain timestamp
      lastValidDetectionRef.current = now;

      // ── Debounce ──────────────────────────────────────────────────────────
      const midi = freqToMidi(frequency);

      if (midi === pendingMidiRef.current) {
        pendingCountRef.current += 1;
      } else {
        pendingMidiRef.current  = midi;
        pendingCountRef.current = 1;
      }

      // Only commit once we have DEBOUNCE_FRAMES consecutive frames
      if (pendingCountRef.current < DEBOUNCE_FRAMES) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      // ── Note lock ─────────────────────────────────────────────────────────
      // If a note is already committed and the new MIDI is different, only
      // switch if the committed note has been absent for NOTE_SUSTAIN_MS.
      // This locks a ringing note in place even if pitch detection drifts.
      if (
        committedMidiRef.current !== -1 &&
        midi !== committedMidiRef.current &&
        now - committedTimeRef.current < NOTE_SUSTAIN_MS
      ) {
        // New note hasn't been absent long enough — hold the committed note
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      // Committed — run soft inference to pick the best string+fret candidate
      committedMidiRef.current = midi;
      committedTimeRef.current = now;
      const note    = midiToNoteName(midi);
      const history = historyRef.current;
      const prev    = history.length > 0 ? history[history.length - 1] : { string: -1, fret: -1 };

      const { string: inferredString, fret, stringConfidence } = inferCandidate(
        midi, buffer, frequency, clarity, prev.string, prev.fret,
      );

      // ── String stability gate ──────────────────────────────────────────────
      // Only switch to a new string after 2 consecutive frames agree on it.
      // If the gate hasn't passed yet, hold the previously committed string.
      let string: number;
      if (inferredString === pendingStringRef.current) {
        pendingStringCountRef.current += 1;
      } else {
        pendingStringRef.current      = inferredString;
        pendingStringCountRef.current = 1;
      }

      if (pendingStringCountRef.current >= 2) {
        // Gate passed — commit the new string
        committedStringRef.current = inferredString;
        string = inferredString;
      } else {
        // Hold previous string for this frame
        string = committedStringRef.current >= 0 ? committedStringRef.current : inferredString;
      }

      // Update history — cap at 3 entries
      if (string >= 0) {
        history.push({ string, fret });
        if (history.length > 3) history.shift();
      }

      setNote(note, frequency, string, fret, stringConfidence);
      setChordName('');

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [getAnalyser, setNote, setChordName]);
}
