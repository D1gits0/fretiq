'use client';

import dynamic from 'next/dynamic';
import { useStore } from '@/store/useStore';
import { useKatanaAudio } from '@/hooks/useKatanaAudio';
import { useAIClassifier } from '@/hooks/useAIClassifier';
import DataRecorder from '@/components/DataRecorder';

// KatanaScene uses R3F — must be client-only, no SSR
const KatanaScene = dynamic(() => import('@/components/KatanaScene'), { ssr: false });

// ─── Root page ────────────────────────────────────────────────────────────────

export default function Page() {
  const { startListening, stopListening } = useKatanaAudio();
  useAIClassifier();

  const isListening = useStore((s) => s.isListening);
  const audioError  = useStore((s) => s.audioError);

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>

      {/* ── 3D Scene ── always mounted so R3F initialises early; fades in on connect */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          opacity: isListening ? 1 : 0,
          transition: 'opacity 800ms ease',
          pointerEvents: isListening ? 'auto' : 'none',
        }}
      >
        <KatanaScene />
      </div>

      {/* ── Onboarding overlay ── fades out once listening */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 24,
          opacity: isListening ? 0 : 1,
          transition: 'opacity 600ms ease',
          pointerEvents: isListening ? 'none' : 'auto',
        }}
      >
        <Logo />

        <p style={styles.subtitle}>
          Connect your Boss Katana Gen 3 via USB-C, then hit the button below.
        </p>

        <ConnectButton
          onConnect={startListening}
          onDisconnect={stopListening}
          isListening={isListening}
        />

        {audioError && <ErrorBanner message={audioError} />}

        <p style={styles.hint}>
          When the browser asks for microphone access, select the{' '}
          <strong style={{ color: '#f1f5f9' }}>BOSS Katana</strong> input.
        </p>
      </div>

      {/* ── Disconnect button ── visible in scene so user can return to onboarding */}
      {isListening && (
        <button
          onClick={stopListening}
          style={styles.disconnectBtn}
          aria-label="Disconnect Katana"
        >
          ✕ Disconnect
        </button>
      )}

      {/* ── Data recorder ── bottom-right panel, only when audio is live */}
      {isListening && (
        <div style={styles.recorderAnchor}>
          <DataRecorder />
        </div>
      )}
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function Logo() {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={styles.logoMark}>⚡</div>
      <h1 style={styles.title}>Katana Vision</h1>
    </div>
  );
}

interface ConnectButtonProps {
  onConnect: () => void;
  onDisconnect: () => void;
  isListening: boolean;
}

function ConnectButton({ onConnect, isListening }: ConnectButtonProps) {
  return (
    <button
      onClick={onConnect}
      disabled={isListening}
      style={{
        ...styles.connectBtn,
        opacity: isListening ? 0.5 : 1,
        cursor: isListening ? 'default' : 'pointer',
      }}
      aria-label="Connect Boss Katana Gen 3"
    >
      {isListening ? 'Connecting…' : 'Connect Katana'}
    </button>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div role="alert" style={styles.errorBanner}>
      <span style={{ marginRight: 8 }}>⚠</span>
      {message}
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = {
  title: {
    margin: 0,
    fontSize: '2.5rem',
    fontWeight: 700,
    fontFamily: 'monospace',
    color: '#f1f5f9',
    letterSpacing: '-0.02em',
  } satisfies React.CSSProperties,

  logoMark: {
    fontSize: '3rem',
    lineHeight: 1,
    marginBottom: 8,
  } satisfies React.CSSProperties,

  subtitle: {
    margin: 0,
    maxWidth: 380,
    textAlign: 'center',
    fontFamily: 'monospace',
    fontSize: '0.95rem',
    color: '#94a3b8',
    lineHeight: 1.6,
  } satisfies React.CSSProperties,

  hint: {
    margin: 0,
    maxWidth: 360,
    textAlign: 'center',
    fontFamily: 'monospace',
    fontSize: '0.8rem',
    color: '#475569',
    lineHeight: 1.5,
  } satisfies React.CSSProperties,

  connectBtn: {
    padding: '14px 40px',
    fontSize: '1rem',
    fontWeight: 600,
    fontFamily: 'monospace',
    color: '#020617',
    background: '#f1f5f9',
    border: 'none',
    borderRadius: 8,
    letterSpacing: '0.04em',
    transition: 'background 150ms ease, transform 100ms ease',
  } satisfies React.CSSProperties,

  disconnectBtn: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    padding: '8px 16px',
    fontSize: '0.8rem',
    fontFamily: 'monospace',
    color: '#94a3b8',
    background: 'rgba(15, 23, 42, 0.7)',
    border: '1px solid #334155',
    borderRadius: 6,
    cursor: 'pointer',
    backdropFilter: 'blur(4px)',
  } satisfies React.CSSProperties,

  recorderAnchor: {
    position: 'absolute',
    bottom: 20,
    right: 20,
  } satisfies React.CSSProperties,

  errorBanner: {
    display: 'flex',
    alignItems: 'center',
    padding: '10px 16px',
    background: 'rgba(220, 38, 38, 0.15)',
    border: '1px solid rgba(220, 38, 38, 0.4)',
    borderRadius: 6,
    fontFamily: 'monospace',
    fontSize: '0.85rem',
    color: '#fca5a5',
    maxWidth: 400,
  } satisfies React.CSSProperties,
} as const;
