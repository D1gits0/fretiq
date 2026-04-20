'use client';

import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useStore } from '@/store/useStore';

// ─── Fretboard geometry constants ─────────────────────────────────────────────

const FRET_COUNT   = 24;
const STRING_COUNT = 6;

// Physical dimensions (Three.js units)
const BOARD_LENGTH = 8;       // X axis — nut (left) to body (right)
const BOARD_WIDTH  = 1.6;     // Z axis — low E (back) to high E (front)
const BOARD_DEPTH  = 0.08;    // Y axis — thickness of the fretboard slab
const NUT_X        = -BOARD_LENGTH / 2;

// String spacing along Z
const STRING_SPACING = BOARD_WIDTH / (STRING_COUNT - 1);

// Fret positions — equal temperament: fret n is at ratio 2^(n/12) from nut
// We map fret 0 (nut) to NUT_X and fret 24 to NUT_X + BOARD_LENGTH
const FRET_POSITIONS: number[] = Array.from({ length: FRET_COUNT + 1 }, (_, n) => {
  const ratio = 1 - Math.pow(2, -n / 12);   // 0 at nut, approaches 1 at infinity
  // Normalise so fret 24 lands at the end of the board
  const norm24 = 1 - Math.pow(2, -24 / 12);
  return NUT_X + (ratio / norm24) * BOARD_LENGTH;
});

// Standard fret marker positions — single dot frets only (no 12/24 doubles here)
const MARKER_FRETS = new Set([3, 5, 7, 9, 12, 15, 17, 19, 21]);
const DOUBLE_MARKER_FRETS = new Set([12]);

// ─── Per-string colors — nickel/steel wound strings ──────────────────────────
const STRING_COLORS = [
  '#9ca3af', // low E  — thick wound, slightly darker
  '#a1a1aa',
  '#b4b4b4',
  '#c4c4c4',
  '#d4d4d4',
  '#e5e5e5', // high E — thinnest, brightest
];

// ─── Active note glow ─────────────────────────────────────────────────────────
const ACTIVE_COLOR    = '#38bdf8';
const ACTIVE_EMISSIVE = '#0ea5e9';

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** X center of a fret slot (midpoint between fret n-1 and fret n wire) */
function fretCenterX(fret: number): number {
  if (fret === 0) {
    // Open string — position slightly left of the nut
    return NUT_X - 0.15;
  }
  return (FRET_POSITIONS[fret - 1] + FRET_POSITIONS[fret]) / 2;
}

/** Z position of string s (0 = low E at back, 5 = high E at front) */
function stringZ(s: number): number {
  return -BOARD_WIDTH / 2 + s * STRING_SPACING;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

/** The wooden fretboard slab — dark rosewood */
function BoardSlab() {
  return (
    <mesh position={[0, -BOARD_DEPTH / 2, 0]} receiveShadow>
      <boxGeometry args={[BOARD_LENGTH, BOARD_DEPTH, BOARD_WIDTH]} />
      <meshStandardMaterial color="#0d0500" roughness={0.85} metalness={0.0} />
    </mesh>
  );
}

/** All fret wires — silver/chrome */
function FretWires() {
  return (
    <>
      {FRET_POSITIONS.slice(1).map((x, i) => (
        <mesh key={i} position={[x, 0.003, 0]}>
          <boxGeometry args={[0.014, 0.014, BOARD_WIDTH + 0.05]} />
          <meshStandardMaterial color="#c0c0c0" metalness={0.95} roughness={0.1} />
        </mesh>
      ))}
    </>
  );
}

/** Nut (zero fret) */
function Nut() {
  return (
    <mesh position={[NUT_X, 0.004, 0]}>
      <boxGeometry args={[0.025, 0.02, BOARD_WIDTH + 0.04]} />
      <meshStandardMaterial color="#f5f5f4" roughness={0.4} />
    </mesh>
  );
}

/** Inlay dots on the fretboard face */
function InlayDots() {
  return (
    <>
      {Array.from(MARKER_FRETS).map((fret) => {
        const x        = fretCenterX(fret);
        const isDouble = DOUBLE_MARKER_FRETS.has(fret);
        return isDouble ? (
          <group key={fret}>
            <mesh position={[x, 0.005, -STRING_SPACING]}>
              <cylinderGeometry args={[0.045, 0.045, 0.01, 16]} />
              <meshStandardMaterial color="#d6d3d1" roughness={0.4} />
            </mesh>
            <mesh position={[x, 0.005, STRING_SPACING]}>
              <cylinderGeometry args={[0.045, 0.045, 0.01, 16]} />
              <meshStandardMaterial color="#d6d3d1" roughness={0.4} />
            </mesh>
          </group>
        ) : (
          <mesh key={fret} position={[x, 0.005, 0]}>
            <cylinderGeometry args={[0.045, 0.045, 0.01, 16]} />
            <meshStandardMaterial color="#d6d3d1" roughness={0.4} />
          </mesh>
        );
      })}
    </>
  );
}

/** All 6 strings — metallic nickel/steel sheen */
function Strings() {
  return (
    <>
      {Array.from({ length: STRING_COUNT }, (_, s) => {
        const z      = stringZ(s);
        const radius = 0.005 + (STRING_COUNT - 1 - s) * 0.002; // thicker for low strings
        return (
          <mesh key={s} position={[0, 0.007, z]} rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[radius, radius, BOARD_LENGTH + 0.4, 8]} />
            <meshStandardMaterial
              color={STRING_COLORS[s]}
              metalness={0.95}
              roughness={0.1}
              envMapIntensity={1.5}
            />
          </mesh>
        );
      })}
    </>
  );
}

/**
 * ActiveFretMarker
 *
 * A glowing sphere + halo ring that lerps smoothly to the active string + fret.
 * Position is interpolated each frame so note transitions crossfade rather than
 * snap. Two point lights (tight + wide) create a dramatic spill onto the fretboard.
 * Uses useFrame + getState() — zero React re-renders.
 */
function ActiveFretMarker() {
  const coreRef  = useRef<THREE.Mesh>(null);
  const haloRef  = useRef<THREE.Mesh>(null);
  const lightRef = useRef<THREE.PointLight>(null);
  const fillRef  = useRef<THREE.PointLight>(null);

  // Current lerped world position
  const currentXRef = useRef<number>(0);
  const currentZRef = useRef<number>(0);

  // Glow brightness 0–1, lerped independently so fade-out is smooth
  const glowRef = useRef<number>(0);

  useFrame(({ clock }, delta) => {
    const { string, fret, intensity, stringConfidence } = useStore.getState();
    const active = string >= 0 && fret >= 0;

    // tau for position lerp: 0.08s → ~80ms convergence
    const posTau   = 0.08;
    const posAlpha = 1 - Math.exp(-delta / posTau);

    // tau for glow fade-in: 0.05s (snappy on attack)
    // tau for glow fade-out: 0.18s → ~600ms to reach ~3% (perceptually gone)
    const glowTau   = active ? 0.05 : 0.18;
    const glowAlpha = 1 - Math.exp(-delta / glowTau);
    const glowTarget = active ? 1 : 0;
    glowRef.current += (glowTarget - glowRef.current) * glowAlpha;

    const glow = glowRef.current;

    // Hide completely only when glow is negligible — avoids GPU overdraw at rest
    const visible = glow > 0.005;
    [coreRef, haloRef].forEach(r => { if (r.current) r.current.visible = visible; });
    [lightRef, fillRef].forEach(r => { if (r.current) r.current.visible = visible; });

    if (!visible || !coreRef.current || !haloRef.current || !lightRef.current || !fillRef.current) return;

    // Lerp position toward target (or hold last position while fading out)
    if (active) {
      const targetX = fretCenterX(fret);
      const targetZ = stringZ(string);
      currentXRef.current += (targetX - currentXRef.current) * posAlpha;
      currentZRef.current += (targetZ - currentZRef.current) * posAlpha;
    }

    const cx = currentXRef.current;
    const cz = currentZRef.current;
    const t  = clock.elapsedTime;

    const conf = active ? (0.3 + stringConfidence * 0.7) : 1.0;

    // Core sphere
    const corePulse = 1 + intensity * 0.5 + Math.sin(t * 10) * 0.08;
    coreRef.current.position.set(cx, 0.07, cz);
    coreRef.current.scale.setScalar(corePulse);
    const coreMat = coreRef.current.material as THREE.MeshStandardMaterial;
    coreMat.emissiveIntensity = (2.0 + intensity * 3.0) * conf * glow;

    // Halo ring
    const haloPulse = 1.6 + intensity * 0.8 + Math.sin(t * 5) * 0.15;
    haloRef.current.position.set(cx, 0.07, cz);
    haloRef.current.scale.setScalar(haloPulse);
    const haloMat = haloRef.current.material as THREE.MeshStandardMaterial;
    haloMat.opacity = (0.25 + intensity * 0.3) * (0.2 + stringConfidence * 0.8) * glow;

    // Tight key light
    lightRef.current.position.set(cx, 0.25, cz);
    lightRef.current.intensity = (2.0 + intensity * 4.0) * conf * glow;

    // Wide fill light
    fillRef.current.position.set(cx, 0.8, cz);
    fillRef.current.intensity = (0.8 + intensity * 1.5) * conf * glow;
  });

  return (
    <group>
      {/* Core glow sphere */}
      <mesh ref={coreRef} visible={false}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshStandardMaterial
          color={ACTIVE_COLOR}
          emissive={ACTIVE_EMISSIVE}
          emissiveIntensity={2.0}
          roughness={0.0}
          metalness={0.2}
        />
      </mesh>

      {/* Outer halo — transparent, larger, slower pulse */}
      <mesh ref={haloRef} visible={false}>
        <sphereGeometry args={[0.08, 12, 12]} />
        <meshStandardMaterial
          color={ACTIVE_COLOR}
          emissive={ACTIVE_EMISSIVE}
          emissiveIntensity={1.0}
          transparent
          opacity={0.25}
          roughness={0.0}
          depthWrite={false}
        />
      </mesh>

      {/* Tight key light */}
      <pointLight
        ref={lightRef}
        color={ACTIVE_COLOR}
        intensity={2.0}
        distance={1.8}
        visible={false}
      />

      {/* Wide fill light */}
      <pointLight
        ref={fillRef}
        color={ACTIVE_COLOR}
        intensity={0.8}
        distance={5.0}
        visible={false}
      />
    </group>
  );
}

/**
 * NoteHUD
 *
 * DOM overlay showing the current note name and chord.
 * Uses useStore() hook so it re-renders on note changes.
 * Positioned via absolute CSS — lives outside the Canvas.
 */
export function NoteHUD() {
  const note      = useStore((s) => s.note);
  const chordName = useStore((s) => s.chordName);
  const intensity = useStore((s) => s.intensity);

  if (!note) return null;

  return (
    <div
      style={{
        position: 'absolute',
        top: 16,
        left: '50%',
        transform: 'translateX(-50%)',
        textAlign: 'center',
        fontFamily: 'monospace',
        pointerEvents: 'none',
        userSelect: 'none',
      }}
    >
      <div
        style={{
          fontSize: '3rem',
          fontWeight: 700,
          color: '#22d3ee',
          opacity: 0.4 + intensity * 0.6,
          transition: 'opacity 80ms linear',
          lineHeight: 1,
        }}
      >
        {note}
      </div>
      {chordName && (
        <div style={{ fontSize: '1.1rem', color: '#94a3b8', marginTop: 4 }}>
          {chordName}
        </div>
      )}
    </div>
  );
}

// ─── Root export ──────────────────────────────────────────────────────────────

/**
 * Fretboard
 *
 * Full 3D fretboard scene object. Mount this inside a <Canvas>.
 * The NoteHUD overlay is exported separately and should be placed
 * outside the Canvas as a DOM sibling.
 */
export default function Fretboard() {
  return (
    <group>
      <BoardSlab />
      <Nut />
      <FretWires />
      <InlayDots />
      <Strings />
      <ActiveFretMarker />
    </group>
  );
}
