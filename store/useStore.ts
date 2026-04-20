import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

// ─── Types ────────────────────────────────────────────────────────────────────

export type GuitarStyle = 'Shred' | 'Ambient' | 'Chords' | 'Idle';

export interface StyleConfidence {
  Shred: number;    // 0–1
  Ambient: number;  // 0–1
  Chords: number;   // 0–1
}

export interface KatanaStore {
  // ── Audio State ──────────────────────────────────────────────────────────
  isListening: boolean;
  audioError: string | null;

  // ── AI Predictions ───────────────────────────────────────────────────────
  style: GuitarStyle;
  intensity: number;           // 0–1, derived from RMS volume
  confidence: StyleConfidence; // per-class softmax outputs
  frequencyData: Uint8Array;   // raw FFT snapshot (length = analyser.frequencyBinCount)

  // ── Pitch / Fretboard ─────────────────────────────────────────────────────
  note: string;             // e.g. 'E2', 'A3', '' when silent
  frequency: number;        // detected Hz, 0 when silent
  fret: number;             // 0–24, -1 when no note detected
  string: number;           // 0–5 (0 = low E), -1 when no note detected
  chordName: string;        // e.g. 'Em', 'G', '' when not a recognised chord
  stringConfidence: number; // 0–1, how confident the string mapping is

  // ── Actions ──────────────────────────────────────────────────────────────
  setListening: (listening: boolean) => void;
  setAudioError: (error: string | null) => void;
  setStyle: (style: GuitarStyle) => void;
  setIntensity: (intensity: number) => void;
  setConfidence: (confidence: StyleConfidence) => void;
  setFrequencyData: (data: Uint8Array) => void;
  setNote: (note: string, frequency: number, string: number, fret: number, stringConfidence: number) => void;
  setChordName: (chordName: string) => void;
  reset: () => void;
}

// ─── Initial State ────────────────────────────────────────────────────────────

const initialState = {
  isListening: false,
  audioError: null,
  style: 'Idle' as GuitarStyle,
  intensity: 0,
  confidence: { Shred: 0, Ambient: 0, Chords: 0 },
  frequencyData: new Uint8Array(0),
  note: '',
  frequency: 0,
  fret: -1,
  string: -1,
  chordName: '',
  stringConfidence: 0,
};

// ─── Store ────────────────────────────────────────────────────────────────────

export const useStore = create<KatanaStore>()(
  subscribeWithSelector((set) => ({
    ...initialState,

    setListening: (isListening) => set({ isListening }),

    setAudioError: (audioError) => set({ audioError }),

    setStyle: (style) => set({ style }),

    setIntensity: (intensity) => set({ intensity: Math.max(0, Math.min(1, intensity)) }),

    setConfidence: (confidence) => set({ confidence }),

    setFrequencyData: (frequencyData) => set({ frequencyData }),

    // Batch note fields into one set() call to avoid cascading re-renders
    setNote: (note, frequency, string, fret, stringConfidence) => set({ note, frequency, string, fret, stringConfidence }),

    setChordName: (chordName) => set({ chordName }),

    reset: () => set(initialState),
  }))
);

// ─── Derived Selectors ────────────────────────────────────────────────────────

export const selectDominantStyle = (state: KatanaStore) => ({
  style: state.style,
  score: state.confidence[state.style as keyof StyleConfidence] ?? 0,
});

export const selectIsPlaying = (state: KatanaStore) => state.intensity > 0.05;

export const selectActiveFret = (state: KatanaStore) => ({
  string: state.string,
  fret: state.fret,
  note: state.note,
});