'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { useStore } from '@/store/useStore';

// ─── Types ────────────────────────────────────────────────────────────────────

/** One captured frame */
interface Frame {
  data: number[];     // 1024 frequency bins, 0–255
  label: StringLabel; // which string was being played
  preset: Preset;     // amp/tone preset at time of capture
}

type StringLabel = 'E2' | 'A2' | 'D3' | 'G3' | 'B3' | 'E4';
type Preset      = 'Clean' | 'Crunch' | 'Lead' | 'Other';

// ─── Config ───────────────────────────────────────────────────────────────────

const STRING_LABELS: StringLabel[] = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4'];
const PRESETS: Preset[]            = ['Clean', 'Crunch', 'Lead', 'Other'];

/** Frame count targets shown in the progress indicators */
const TARGET_GOOD = 5000;
const TARGET_GREAT = 10000;

const STRING_COLOR: Record<StringLabel, { idle: string; active: string; track: string }> = {
  E2: { idle: 'rgba(239,68,68,0.12)',   active: '#ef4444', track: '#ef4444' },
  A2: { idle: 'rgba(249,115,22,0.12)',  active: '#f97316', track: '#f97316' },
  D3: { idle: 'rgba(234,179,8,0.12)',   active: '#eab308', track: '#eab308' },
  G3: { idle: 'rgba(34,197,94,0.12)',   active: '#22c55e', track: '#22c55e' },
  B3: { idle: 'rgba(59,130,246,0.12)',  active: '#3b82f6', track: '#3b82f6' },
  E4: { idle: 'rgba(168,85,247,0.12)',  active: '#a855f7', track: '#a855f7' },
};

// ─── Recording hook ───────────────────────────────────────────────────────────

function useRecorder(activePreset: Preset) {
  const [counts, setCounts] = useState<Record<StringLabel, number>>({
    E2: 0, A2: 0, D3: 0, G3: 0, B3: 0, E4: 0,
  });
  const [activeLabel, setActiveLabel] = useState<StringLabel | null>(null);

  const samplesRef     = useRef<Frame[]>([]);
  const activeLabelRef = useRef<StringLabel | null>(null);
  // Keep a ref to the current preset so the RAF loop always reads the latest value
  const activePresetRef = useRef<Preset>(activePreset);
  const rafRef          = useRef<number | null>(null);

  // Sync preset ref whenever the prop changes
  useEffect(() => {
    activePresetRef.current = activePreset;
  }, [activePreset]);

  const tick = useCallback(() => {
    const label = activeLabelRef.current;
    if (!label) return;

    const { frequencyData } = useStore.getState();

    if (frequencyData.length > 0) {
      samplesRef.current.push({
        data:   Array.from(frequencyData),
        label,
        preset: activePresetRef.current,
      });
      setCounts((prev) => ({ ...prev, [label]: prev[label] + 1 }));
    }

    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const startRecording = useCallback((label: StringLabel) => {
    // Auto-stop whatever is currently recording
    if (activeLabelRef.current) {
      cancelAnimationFrame(rafRef.current!);
      rafRef.current = null;
      activeLabelRef.current = null;
    }
    activeLabelRef.current = label;
    setActiveLabel(label);
    rafRef.current = requestAnimationFrame(tick);
  }, [tick]);

  const stopRecording = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    activeLabelRef.current = null;
    setActiveLabel(null);
  }, []);

  const toggleRecording = useCallback((label: StringLabel) => {
    if (activeLabelRef.current === label) {
      stopRecording();
    } else {
      startRecording(label);
    }
  }, [startRecording, stopRecording]);

  const clearAll = useCallback(() => {
    stopRecording();
    samplesRef.current = [];
    setCounts({ E2: 0, A2: 0, D3: 0, G3: 0, B3: 0, E4: 0 });
  }, [stopRecording]);

  useEffect(() => () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
  }, []);

  const totalFrames = Object.values(counts).reduce((a, b) => a + b, 0);

  return { samplesRef, counts, totalFrames, toggleRecording, clearAll, activeLabel };
}

// ─── Export ───────────────────────────────────────────────────────────────────

function exportJSON(frames: Frame[], totalFrames: number) {
  const payload = {
    exportedAt:  new Date().toISOString(),
    totalFrames,
    fftBins:     1024,
    sampleRate:  44100,
    fftSize:     2048,
    frames,
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `katana-strings-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── Progress bar ─────────────────────────────────────────────────────────────

function ProgressBar({ label, count, color }: { label: StringLabel; count: number; color: string }) {
  const pctGood  = Math.min(1, count / TARGET_GOOD);
  const pctGreat = Math.min(1, count / TARGET_GREAT);
  const isGood   = count >= TARGET_GOOD;
  const isGreat  = count >= TARGET_GREAT;

  const statusColor = isGreat ? '#4ade80' : isGood ? '#facc15' : '#64748b';
  const statusText  = isGreat ? '✓✓ great' : isGood ? '✓ good' : `${Math.round(pctGood * 100)}%`;

  return (
    <div style={styles.progressRow}>
      <span style={{ ...styles.progressLabel, color }}>{label}</span>

      {/* Two-tier track: good threshold at 50%, great at 100% */}
      <div style={styles.progressTrack}>
        {/* Good tier (0 → TARGET_GOOD) */}
        <div style={{
          position: 'absolute', left: 0, top: 0, height: '100%',
          width: `${pctGood * 50}%`,
          background: color, borderRadius: 3, opacity: 0.7,
          transition: 'width 120ms linear',
        }} />
        {/* Great tier (TARGET_GOOD → TARGET_GREAT) */}
        <div style={{
          position: 'absolute', left: '50%', top: 0, height: '100%',
          width: `${(pctGreat - pctGood * 0.5) * 100}%`,
          background: color, borderRadius: 3, opacity: 1.0,
          transition: 'width 120ms linear',
        }} />
        {/* Midpoint marker */}
        <div style={{
          position: 'absolute', left: '50%', top: 0,
          width: 1, height: '100%', background: '#334155',
        }} />
      </div>

      <span style={{ ...styles.progressCount, color: statusColor }}>
        {count >= 1000 ? `${(count / 1000).toFixed(1)}k` : count} {statusText}
      </span>
    </div>
  );
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function DataRecorder() {
  const isListening = useStore((s) => s.isListening);
  const [preset, setPreset] = useState<Preset>('Clean');

  const { samplesRef, counts, totalFrames, toggleRecording, clearAll, activeLabel } =
    useRecorder(preset);

  if (!isListening) {
    return (
      <div style={styles.panel}>
        <p style={styles.disabledMsg}>Connect Katana to start recording.</p>
      </div>
    );
  }

  return (
    <div style={styles.panel} aria-label="String data recorder">

      {/* Header */}
      <div style={styles.header}>
        <span style={styles.title}>String Recorder</span>
        <span style={styles.totalBadge}>{totalFrames.toLocaleString()} frames</span>
      </div>

      {/* Preset selector */}
      <div style={styles.presetRow}>
        <span style={styles.presetLabel}>Preset</span>
        <div style={styles.presetButtons}>
          {PRESETS.map((p) => (
            <button
              key={p}
              onClick={() => setPreset(p)}
              style={{
                ...styles.presetBtn,
                background: preset === p ? '#f1f5f9' : 'transparent',
                color:      preset === p ? '#020617' : '#94a3b8',
                borderColor: preset === p ? '#f1f5f9' : '#334155',
              }}
              aria-pressed={preset === p}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* String record buttons */}
      <div style={styles.stringGrid}>
        {STRING_LABELS.map((label) => {
          const isActive = activeLabel === label;
          const cfg      = STRING_COLOR[label];
          return (
            <button
              key={label}
              onClick={() => toggleRecording(label)}
              style={{
                ...styles.stringBtn,
                background:  isActive ? cfg.active : cfg.idle,
                borderColor: isActive ? cfg.active : cfg.track,
                color:       isActive ? '#020617' : '#f1f5f9',
                transform:   isActive ? 'scale(0.95)' : 'scale(1)',
              }}
              aria-pressed={isActive}
              aria-label={isActive ? `Stop recording ${label}` : `Start recording ${label}`}
            >
              <span style={styles.stringBtnLabel}>{label}</span>
              {isActive && <span style={styles.recDot} aria-hidden>●</span>}
            </button>
          );
        })}
      </div>

      {/* Per-string progress */}
      <div style={styles.progressSection}>
        {STRING_LABELS.map((label) => (
          <ProgressBar
            key={label}
            label={label}
            count={counts[label]}
            color={STRING_COLOR[label].track}
          />
        ))}
        <div style={styles.targets}>
          <span>5k = good</span>
          <span>10k = great</span>
        </div>
      </div>

      {/* Actions */}
      <div style={styles.actionRow}>
        <button
          onClick={() => exportJSON(samplesRef.current, totalFrames)}
          disabled={totalFrames === 0}
          style={{
            ...styles.actionBtn, ...styles.exportBtn,
            opacity: totalFrames === 0 ? 0.4 : 1,
            cursor:  totalFrames === 0 ? 'default' : 'pointer',
          }}
          aria-label="Export training data as JSON"
        >
          ↓ Export JSON
        </button>

        <button
          onClick={clearAll}
          disabled={totalFrames === 0}
          style={{
            ...styles.actionBtn, ...styles.clearBtn,
            opacity: totalFrames === 0 ? 0.4 : 1,
            cursor:  totalFrames === 0 ? 'default' : 'pointer',
          }}
          aria-label="Clear all recorded data"
        >
          ✕ Clear
        </button>
      </div>

      <p style={styles.hint}>Hold a string button while playing that string.</p>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = {
  panel: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
    padding: '14px 16px',
    background: 'rgba(2, 6, 23, 0.88)',
    border: '1px solid #1e293b',
    borderRadius: 10,
    backdropFilter: 'blur(8px)',
    fontFamily: 'monospace',
    color: '#f1f5f9',
    width: 300,
  } satisfies React.CSSProperties,

  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  } satisfies React.CSSProperties,

  title: {
    fontSize: '0.8rem',
    fontWeight: 700,
    color: '#94a3b8',
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
  } satisfies React.CSSProperties,

  totalBadge: {
    fontSize: '0.75rem',
    color: '#64748b',
    background: '#0f172a',
    padding: '2px 8px',
    borderRadius: 4,
    border: '1px solid #1e293b',
  } satisfies React.CSSProperties,

  presetRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  } satisfies React.CSSProperties,

  presetLabel: {
    fontSize: '0.75rem',
    color: '#64748b',
    flexShrink: 0,
  } satisfies React.CSSProperties,

  presetButtons: {
    display: 'flex',
    gap: 4,
    flex: 1,
  } satisfies React.CSSProperties,

  presetBtn: {
    flex: 1,
    padding: '4px 0',
    fontSize: '0.72rem',
    fontWeight: 600,
    fontFamily: 'monospace',
    border: '1px solid',
    borderRadius: 5,
    cursor: 'pointer',
    transition: 'background 100ms ease, color 100ms ease',
  } satisfies React.CSSProperties,

  stringGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(6, 1fr)',
    gap: 5,
  } satisfies React.CSSProperties,

  stringBtn: {
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '10px 0',
    border: '1px solid',
    borderRadius: 6,
    cursor: 'pointer',
    fontFamily: 'monospace',
    transition: 'background 80ms ease, transform 80ms ease',
    userSelect: 'none',
  } satisfies React.CSSProperties,

  stringBtnLabel: {
    fontSize: '0.78rem',
    fontWeight: 700,
  } satisfies React.CSSProperties,

  recDot: {
    position: 'absolute',
    top: 3,
    right: 4,
    fontSize: '0.45rem',
    color: '#020617',
  } satisfies React.CSSProperties,

  progressSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: 5,
    padding: '6px 0 2px',
    borderTop: '1px solid #1e293b',
  } satisfies React.CSSProperties,

  progressRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  } satisfies React.CSSProperties,

  progressLabel: {
    fontSize: '0.7rem',
    fontWeight: 700,
    width: 24,
    flexShrink: 0,
  } satisfies React.CSSProperties,

  progressTrack: {
    position: 'relative',
    flex: 1,
    height: 6,
    background: '#0f172a',
    borderRadius: 3,
    overflow: 'hidden',
  } satisfies React.CSSProperties,

  progressCount: {
    fontSize: '0.65rem',
    width: 60,
    textAlign: 'right',
    flexShrink: 0,
  } satisfies React.CSSProperties,

  targets: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.62rem',
    color: '#334155',
    paddingTop: 2,
  } satisfies React.CSSProperties,

  actionRow: {
    display: 'flex',
    gap: 8,
  } satisfies React.CSSProperties,

  actionBtn: {
    flex: 1,
    padding: '7px 0',
    fontSize: '0.78rem',
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
    fontSize: '0.68rem',
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
