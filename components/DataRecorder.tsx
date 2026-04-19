'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { useStore } from '@/store/useStore';
import type { GuitarStyle } from '@/store/useStore';

// ─── Types ────────────────────────────────────────────────────────────────────

/** One captured frame: 1024 frequency bins (0–255) + RMS intensity */
interface Frame {
  bins: number[];
  intensity: number;
}

/** All frames collected under one style label */
interface LabeledSamples {
  label: Exclude<GuitarStyle, 'Idle'>;
  frames: Frame[];
}

type RecordableStyle = Exclude<GuitarStyle, 'Idle'>;

// ─── Config ───────────────────────────────────────────────────────────────────

const RECORD_STYLES: RecordableStyle[] = ['Shred', 'Ambient', 'Chords'];

const STYLE_COLOR: Record<RecordableStyle, { bg: string; border: string; active: string }> = {
  Shred:   { bg: 'rgba(220,38,38,0.12)',  border: '#dc2626', active: '#ef4444' },
  Ambient: { bg: 'rgba(8,145,178,0.12)',  border: '#0891b2', active: '#22d3ee' },
  Chords:  { bg: 'rgba(22,163,74,0.12)',  border: '#16a34a', active: '#4ade80' },
};

// ─── Hook: recording loop ─────────────────────────────────────────────────────

/**
 * useRecorder
 *
 * Manages the RAF-based capture loop. Reads frequencyData + intensity
 * directly from useStore.getState() each frame — no React re-renders
 * in the hot path.
 *
 * Returns:
 *  - samples: accumulated LabeledSamples[]
 *  - counts: per-label frame counts (for display)
 *  - totalFrames: sum across all labels
 *  - startRecording(label): begin capturing frames for label
 *  - stopRecording(): stop the current capture
 *  - clearAll(): wipe all collected data
 *  - activeLabel: which label is currently recording (null if idle)
 */
function useRecorder() {
  const [counts, setCounts]       = useState<Record<RecordableStyle, number>>({ Shred: 0, Ambient: 0, Chords: 0 });
  const [activeLabel, setActive]  = useState<RecordableStyle | null>(null);

  // Mutable refs — never trigger re-renders
  const samplesRef    = useRef<LabeledSamples[]>([]);
  const activeLabelRef = useRef<RecordableStyle | null>(null);
  const rafRef        = useRef<number | null>(null);

  const tick = useCallback(() => {
    const label = activeLabelRef.current;
    if (!label) return;

    const { frequencyData, intensity } = useStore.getState();

    if (frequencyData.length > 0) {
      // Find or create the bucket for this label
      let bucket = samplesRef.current.find((s) => s.label === label);
      if (!bucket) {
        bucket = { label, frames: [] };
        samplesRef.current.push(bucket);
      }

      bucket.frames.push({
        bins: Array.from(frequencyData), // copy — frequencyData ref changes each frame anyway
        intensity,
      });

      // Update display counts (batched by RAF cadence — cheap)
      setCounts((prev) => ({ ...prev, [label]: (prev[label] ?? 0) + 1 }));
    }

    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const startRecording = useCallback((label: RecordableStyle) => {
    if (activeLabelRef.current) return; // already recording
    activeLabelRef.current = label;
    setActive(label);
    rafRef.current = requestAnimationFrame(tick);
  }, [tick]);

  const stopRecording = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    activeLabelRef.current = null;
    setActive(null);
  }, []);

  const clearAll = useCallback(() => {
    stopRecording();
    samplesRef.current = [];
    setCounts({ Shred: 0, Ambient: 0, Chords: 0 });
  }, [stopRecording]);

  // Safety: cancel RAF on unmount
  useEffect(() => () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
  }, []);

  const totalFrames = counts.Shred + counts.Ambient + counts.Chords;

  return { samplesRef, counts, totalFrames, startRecording, stopRecording, clearAll, activeLabel };
}

// ─── Export helper ────────────────────────────────────────────────────────────

function exportJSON(samples: LabeledSamples[], totalFrames: number) {
  const payload = {
    exportedAt: new Date().toISOString(),
    totalFrames,
    fftBins: 1024,
    sampleRate: 44100,
    fftSize: 2048,
    samples,
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `katana-training-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function DataRecorder() {
  const isListening = useStore((s) => s.isListening);
  const { samplesRef, counts, totalFrames, startRecording, stopRecording, clearAll, activeLabel } =
    useRecorder();

  if (!isListening) {
    return (
      <div style={styles.panel}>
        <p style={styles.disabledMsg}>Connect Katana to start recording.</p>
      </div>
    );
  }

  return (
    <div style={styles.panel} aria-label="Data recorder">

      {/* Header */}
      <div style={styles.header}>
        <span style={styles.title}>Data Recorder</span>
        <span style={styles.totalBadge}>{totalFrames} frames</span>
      </div>

      {/* Record buttons */}
      <div style={styles.buttonRow}>
        {RECORD_STYLES.map((label) => {
          const isActive = activeLabel === label;
          const cfg = STYLE_COLOR[label];
          return (
            <button
              key={label}
              onPointerDown={() => startRecording(label)}
              onPointerUp={stopRecording}
              onPointerLeave={stopRecording}   // release if cursor drifts off
              style={{
                ...styles.recordBtn,
                background: isActive ? cfg.active : cfg.bg,
                borderColor: isActive ? cfg.active : cfg.border,
                color: isActive ? '#020617' : '#f1f5f9',
                transform: isActive ? 'scale(0.96)' : 'scale(1)',
              }}
              aria-pressed={isActive}
              aria-label={`Record ${label} — hold to capture`}
            >
              <span style={styles.btnLabel}>{label}</span>
              <span style={styles.btnCount}>{counts[label]} frames</span>
              {isActive && <span style={styles.recDot} aria-hidden="true">●</span>}
            </button>
          );
        })}
      </div>

      {/* Per-label breakdown */}
      <div style={styles.breakdown}>
        {RECORD_STYLES.map((label) => {
          const pct = totalFrames > 0 ? counts[label] / totalFrames : 0;
          const cfg = STYLE_COLOR[label];
          return (
            <div key={label} style={styles.barRow}>
              <span style={{ ...styles.barLabel, color: cfg.border }}>{label}</span>
              <div style={styles.barTrack}>
                <div
                  style={{
                    height: '100%',
                    width: `${Math.round(pct * 100)}%`,
                    background: cfg.border,
                    borderRadius: 3,
                    transition: 'width 120ms linear',
                  }}
                />
              </div>
              <span style={styles.barPct}>{Math.round(pct * 100)}%</span>
            </div>
          );
        })}
      </div>

      {/* Actions */}
      <div style={styles.actionRow}>
        <button
          onClick={() => exportJSON(samplesRef.current, totalFrames)}
          disabled={totalFrames === 0}
          style={{
            ...styles.actionBtn,
            ...styles.exportBtn,
            opacity: totalFrames === 0 ? 0.4 : 1,
            cursor: totalFrames === 0 ? 'default' : 'pointer',
          }}
          aria-label="Export training data as JSON"
        >
          ↓ Export JSON
        </button>

        <button
          onClick={clearAll}
          disabled={totalFrames === 0}
          style={{
            ...styles.actionBtn,
            ...styles.clearBtn,
            opacity: totalFrames === 0 ? 0.4 : 1,
            cursor: totalFrames === 0 ? 'default' : 'pointer',
          }}
          aria-label="Clear all recorded data"
        >
          ✕ Clear
        </button>
      </div>

      <p style={styles.hint}>Hold a button while playing to capture frames.</p>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = {
  panel: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    padding: '16px 18px',
    background: 'rgba(2, 6, 23, 0.85)',
    border: '1px solid #1e293b',
    borderRadius: 10,
    backdropFilter: 'blur(8px)',
    fontFamily: 'monospace',
    color: '#f1f5f9',
    minWidth: 280,
    maxWidth: 320,
  } satisfies React.CSSProperties,

  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  } satisfies React.CSSProperties,

  title: {
    fontSize: '0.85rem',
    fontWeight: 700,
    color: '#94a3b8',
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
  } satisfies React.CSSProperties,

  totalBadge: {
    fontSize: '0.8rem',
    color: '#64748b',
    background: '#0f172a',
    padding: '2px 8px',
    borderRadius: 4,
    border: '1px solid #1e293b',
  } satisfies React.CSSProperties,

  buttonRow: {
    display: 'flex',
    gap: 8,
  } satisfies React.CSSProperties,

  recordBtn: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 2,
    padding: '10px 6px',
    border: '1px solid',
    borderRadius: 7,
    cursor: 'pointer',
    fontFamily: 'monospace',
    transition: 'background 80ms ease, transform 80ms ease, border-color 80ms ease',
    userSelect: 'none',
    position: 'relative',
  } satisfies React.CSSProperties,

  btnLabel: {
    fontSize: '0.9rem',
    fontWeight: 700,
  } satisfies React.CSSProperties,

  btnCount: {
    fontSize: '0.7rem',
    opacity: 0.75,
  } satisfies React.CSSProperties,

  recDot: {
    position: 'absolute',
    top: 5,
    right: 7,
    fontSize: '0.55rem',
    color: '#020617',
    animation: 'none',
  } satisfies React.CSSProperties,

  breakdown: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  } satisfies React.CSSProperties,

  barRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  } satisfies React.CSSProperties,

  barLabel: {
    fontSize: '0.75rem',
    width: 52,
    flexShrink: 0,
  } satisfies React.CSSProperties,

  barTrack: {
    flex: 1,
    height: 6,
    background: '#0f172a',
    borderRadius: 3,
    overflow: 'hidden',
  } satisfies React.CSSProperties,

  barPct: {
    fontSize: '0.7rem',
    color: '#475569',
    width: 30,
    textAlign: 'right',
    flexShrink: 0,
  } satisfies React.CSSProperties,

  actionRow: {
    display: 'flex',
    gap: 8,
  } satisfies React.CSSProperties,

  actionBtn: {
    flex: 1,
    padding: '8px 0',
    fontSize: '0.8rem',
    fontWeight: 600,
    fontFamily: 'monospace',
    border: '1px solid',
    borderRadius: 6,
    transition: 'opacity 120ms ease',
  } satisfies React.CSSProperties,

  exportBtn: {
    background: 'rgba(241,245,249,0.08)',
    borderColor: '#334155',
    color: '#f1f5f9',
    cursor: 'pointer',
  } satisfies React.CSSProperties,

  clearBtn: {
    background: 'rgba(220,38,38,0.08)',
    borderColor: '#7f1d1d',
    color: '#fca5a5',
    cursor: 'pointer',
  } satisfies React.CSSProperties,

  hint: {
    margin: 0,
    fontSize: '0.72rem',
    color: '#475569',
    textAlign: 'center',
  } satisfies React.CSSProperties,

  disabledMsg: {
    margin: 0,
    fontSize: '0.85rem',
    color: '#475569',
    textAlign: 'center',
    padding: '8px 0',
  } satisfies React.CSSProperties,
} as const;
