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
  intensity: number;          // 0–1, derived from RMS volume
  confidence: StyleConfidence; // per-class softmax outputs
  frequencyData: Uint8Array;  // raw FFT snapshot (length = analyser.frequencyBinCount)

  // ── Actions ──────────────────────────────────────────────────────────────
  setListening: (listening: boolean) => void;
  setAudioError: (error: string | null) => void;
  setStyle: (style: GuitarStyle) => void;
  setIntensity: (intensity: number) => void;
  setConfidence: (confidence: StyleConfidence) => void;
  setFrequencyData: (data: Uint8Array) => void;
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
};

// ─── Store ────────────────────────────────────────────────────────────────────

export const useStore = create<KatanaStore>()(
  // subscribeWithSelector lets R3F components react to specific slices
  // without re-rendering on every frequency update (critical for 60fps)
  subscribeWithSelector((set) => ({
    ...initialState,

    setListening: (isListening) => set({ isListening }),

    setAudioError: (audioError) => set({ audioError }),

    setStyle: (style) => set({ style }),

    // Clamp intensity to [0, 1] before storing
    setIntensity: (intensity) => set({ intensity: Math.max(0, Math.min(1, intensity)) }),

    setConfidence: (confidence) => set({ confidence }),

    // Typed array assignment — store keeps a reference, not a copy.
    // The hook should pass a *new* Uint8Array each frame so React detects change.
    setFrequencyData: (frequencyData) => set({ frequencyData }),

    reset: () => set(initialState),
  }))
);

// ─── Derived Selectors (use these in components for perf) ─────────────────────

/** Returns the dominant style label and its confidence score */
export const selectDominantStyle = (state: KatanaStore) => ({
  style: state.style,
  score: state.confidence[state.style as keyof StyleConfidence] ?? 0,
});

/** Returns true only when the amp is actively playing (not idle/silent) */
export const selectIsPlaying = (state: KatanaStore) => state.intensity > 0.05;