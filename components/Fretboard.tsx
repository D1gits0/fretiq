'use client';

import { useRef, Suspense } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
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
 * ProbabilityHeatmap
 *
 * Single InstancedMesh with MAX_INSTANCES slots. Each frame, active instances
 * are positioned at valid string/fret positions for the detected pitch and
 * scaled by confidence. Inactive instances are scaled to zero (hidden).
 *
 * InstancedMesh = one draw call regardless of how many markers are visible.
 * No per-frame material uploads, no context loss.
 */

const TUNING_MIDI_FB: readonly number[] = [40, 45, 50, 55, 59, 64];
const FRETS_FB       = 24;
const MIN_BRIGHTNESS = 0.15;
const MAX_INSTANCES  = 6; // max valid strings for any single pitch

// Reusable scratch objects — allocated once, never inside useFrame
const _matrix     = new THREE.Matrix4();
const _matrixBack = new THREE.Matrix4();
const _color      = new THREE.Color(ACTIVE_COLOR);
const _dark       = new THREE.Color('#000000');

// Marker size constants
// Base radius of the sphere geometry is 1 — scale drives actual world size.
// MAX_MARKER_RADIUS = world-space radius of the highest-confidence marker.
// MIN_MARKER_RADIUS = minimum visible size for low-confidence markers.
const MAX_MARKER_RADIUS = 0.21; // 3× the old 0.07
const MIN_MARKER_RADIUS = 0.06; // always visible even at MIN_BRIGHTNESS

function ProbabilityHeatmap() {
  const meshRef    = useRef<THREE.InstancedMesh>(null); // glow spheres
  const backdropRef = useRef<THREE.InstancedMesh>(null); // dark backdrop discs

  // Per-instance lerped brightness (0 = hidden, 1 = full)
  const glowRefs   = useRef<number[]>(new Array(MAX_INSTANCES).fill(0));
  const lastMidi   = useRef<number>(-1);
  const targetGlow = useRef<number[]>(new Array(MAX_INSTANCES).fill(0));
  const targetX    = useRef<number[]>(new Array(MAX_INSTANCES).fill(0));
  const targetZ    = useRef<number[]>(new Array(MAX_INSTANCES).fill(0));

  useFrame((_, delta) => {
    const mesh     = meshRef.current;
    const backdrop = backdropRef.current;
    if (!mesh || !backdrop) return;

    const { note, frequency, stringProbs, intensity } = useStore.getState();
    const midi = frequency > 0
      ? Math.round(12 * Math.log2(frequency / 440) + 69)
      : -1;
    const active = note !== '' && midi >= 0;

    // Recompute targets only when midi changes
    if (midi !== lastMidi.current) {
      lastMidi.current = midi;
      for (let s = 0; s < MAX_INSTANCES; s++) {
        const fret = midi >= 0 ? midi - TUNING_MIDI_FB[s] : -1;
        if (active && fret >= 0 && fret <= FRETS_FB) {
          targetX.current[s]    = fretCenterX(fret);
          targetZ.current[s]    = stringZ(s);
          const prob            = (stringProbs && isFinite(stringProbs[s])) ? stringProbs[s] : 0;
          targetGlow.current[s] = Math.max(MIN_BRIGHTNESS, prob);
        } else {
          targetGlow.current[s] = 0;
        }
      }
    }

    const safeIntensity = isFinite(intensity) ? intensity : 0;

    // Find the max glow this frame so we can scale relative to it
    let maxGlow = 0;
    for (let i = 0; i < MAX_INSTANCES; i++) {
      if (glowRefs.current[i] > maxGlow) maxGlow = glowRefs.current[i];
    }

    for (let i = 0; i < MAX_INSTANCES; i++) {
      const tgt   = active ? targetGlow.current[i] : 0;
      const tau   = tgt > glowRefs.current[i] ? 0.025 : 0.18;
      const alpha = 1 - Math.exp(-delta / tau);
      glowRefs.current[i] += (tgt - glowRefs.current[i]) * alpha;

      const glow = Math.max(0, isFinite(glowRefs.current[i]) ? glowRefs.current[i] : 0);

      if (glow < 0.005) {
        // Hide by scaling to zero
        _matrix.makeScale(0, 0, 0);
        mesh.setMatrixAt(i, _matrix);
        backdrop.setMatrixAt(i, _matrix);
        continue;
      }

      const x = targetX.current[i];
      const z = targetZ.current[i];

      // Scale: highest-confidence marker gets MAX_MARKER_RADIUS,
      // others scale proportionally but never below MIN_MARKER_RADIUS.
      const normGlow  = maxGlow > 0 ? glow / maxGlow : glow;
      const radius    = MIN_MARKER_RADIUS + (MAX_MARKER_RADIUS - MIN_MARKER_RADIUS) * normGlow;
      const pulseSize = radius * (1 + safeIntensity * 0.2);

      _matrix.makeScale(pulseSize, pulseSize, pulseSize);
      _matrix.setPosition(x, 0.07, z);
      mesh.setMatrixAt(i, _matrix);

      // Backdrop disc: slightly larger than sphere, flat on the fretboard face
      const backRadius = pulseSize * 1.6;
      _matrixBack.makeScale(backRadius, 1, backRadius); // Y=1 because disc is flat
      _matrixBack.setPosition(x, 0.01, z);
      backdrop.setMatrixAt(i, _matrixBack);

      // Sphere color: full brightness for highest, dimmer for lower confidence
      const brightness = Math.max(0, Math.min(1, normGlow));
      mesh.setColorAt(i, _color.clone().multiplyScalar(0.3 + brightness * 0.7));

      // Backdrop always dark
      backdrop.setColorAt(i, _dark);
    }

    mesh.instanceMatrix.needsUpdate = true;
    backdrop.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor)     mesh.instanceColor.needsUpdate = true;
    if (backdrop.instanceColor) backdrop.instanceColor.needsUpdate = true;
  });

  return (
    <group>
      {/* Dark backdrop discs — rendered first (behind spheres) */}
      <instancedMesh
        ref={backdropRef}
        args={[undefined, undefined, MAX_INSTANCES]}
        frustumCulled={false}
        renderOrder={0}
      >
        {/* Flat disc, radius 1 — scaled by instance matrix */}
        <cylinderGeometry args={[1, 1, 0.005, 16]} />
        <meshBasicMaterial
          color="#000000"
          transparent
          opacity={0.65}
          depthWrite={false}
        />
      </instancedMesh>

      {/* Glow spheres */}
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, MAX_INSTANCES]}
        frustumCulled={false}
        renderOrder={1}
      >
        <sphereGeometry args={[1, 12, 12]} />
        <meshStandardMaterial
          color={ACTIVE_COLOR}
          emissive={ACTIVE_EMISSIVE}
          emissiveIntensity={2.5}
          transparent
          opacity={0.92}
          roughness={0.0}
          depthWrite={false}
        />
      </instancedMesh>
    </group>
  );
}

/**
 * FloatingNoteLabel
 *
 * Drei Text floating above the fretboard showing the current note name.
 * Visibility is driven by the store subscription — Text renders when note
 * is non-empty, hidden when silent. Drei handles its own material internally
 * so we don't touch it imperatively.
 */
function FloatingNoteLabel() {
  const note = useStore((s) => s.note);

  if (!note) return null;

  return (
    <Text
      position={[0, 0.6, 0]}
      fontSize={0.35}
      color={ACTIVE_COLOR}
      anchorX="center"
      anchorY="middle"
    >
      {note}
    </Text>
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
      <ProbabilityHeatmap />
      {/* Suspense required — Drei Text suspends while loading its font */}
      <Suspense fallback={null}>
        <FloatingNoteLabel />
      </Suspense>
    </group>
  );
}
