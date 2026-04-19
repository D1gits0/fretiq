'use client';

import { useRef, useMemo } from 'react';
import type React from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import {
  MeshDistortMaterial,
  OrbitControls,
} from '@react-three/drei';

// DistortMaterialImpl is not a public export — extract the ref type from the component
type DistortMaterialImpl = React.ComponentRef<typeof MeshDistortMaterial>;
import * as THREE from 'three';
import { useStore, GuitarStyle, StyleConfidence } from '@/store/useStore';

// ─── Per-style visual config ──────────────────────────────────────────────────

interface StyleConfig {
  color: string;
  emissive: string;
  distort: number;
  speed: number;
  particleColor: string;
}

const STYLE_CONFIG: Record<GuitarStyle, StyleConfig> = {
  Idle:    { color: '#1e293b', emissive: '#0f172a', distort: 0.1, speed: 0.5,  particleColor: '#334155' },
  Shred:   { color: '#dc2626', emissive: '#ea580c', distort: 0.8, speed: 3.0,  particleColor: '#f97316' },
  Ambient: { color: '#0891b2', emissive: '#0e7490', distort: 0.3, speed: 0.8,  particleColor: '#67e8f9' },
  Chords:  { color: '#16a34a', emissive: '#15803d', distort: 0.5, speed: 1.5,  particleColor: '#4ade80' },
};

// ─── KatanaOrb ────────────────────────────────────────────────────────────────

/**
 * Icosahedron with MeshDistortMaterial.
 * distort + speed react to style and intensity each frame via useFrame.
 *
 * Fix applied: matRef typed as React.RefObject<MeshDistortMaterialProps>
 * (the concrete ref type Drei exposes for MeshDistortMaterial).
 */
function KatanaOrb() {
  const matRef = useRef<DistortMaterialImpl>(null);
  // Stable color instances — never allocate inside useFrame
  const colorRef    = useRef(new THREE.Color());
  const emissiveRef = useRef(new THREE.Color());

  useFrame(() => {
    const { style, intensity } = useStore.getState();
    const cfg = STYLE_CONFIG[style];

    if (!matRef.current) return;

    colorRef.current.set(cfg.color);
    emissiveRef.current.set(cfg.emissive);

    // DistortMaterialImpl extends THREE.MeshPhysicalMaterial + adds distort/radius
    // speed is handled by Drei's internal useFrame (reads the JSX prop) — don't touch it here
    matRef.current.color.copy(colorRef.current);
    matRef.current.emissive.copy(emissiveRef.current);
    matRef.current.distort = cfg.distort + intensity * 0.4;
  });

  return (
    <mesh>
      <icosahedronGeometry args={[1.4, 4]} />
      <MeshDistortMaterial
        ref={matRef}
        color="#1e293b"
        emissive="#0f172a"
        emissiveIntensity={0.6}
        metalness={0.4}
        roughness={0.2}
        distort={0.1}
        speed={0.5}
      />
    </mesh>
  );
}

// ─── FrequencyRing ────────────────────────────────────────────────────────────

/**
 * 128 FFT points arranged as a 3D ring.
 * Each point's radius is displaced by its bin's energy.
 *
 * Fix applied: colorRef is a stable per-instance THREE.Color — no shared
 * mutation across instances.
 */
const RING_POINTS = 128;
const BASE_RADIUS = 2.2;

function FrequencyRing() {
  const pointsRef = useRef<THREE.Points>(null);
  // One stable Color per instance — safe to mutate inside useFrame
  const colorRef = useRef(new THREE.Color());

  const positions = useMemo(() => new Float32Array(RING_POINTS * 3), []);

  useFrame(() => {
    const { frequencyData, style, intensity } = useStore.getState();
    if (!pointsRef.current) return;

    const cfg = STYLE_CONFIG[style];
    colorRef.current.set(cfg.particleColor);

    const geo = pointsRef.current.geometry;
    const pos = geo.attributes.position as THREE.BufferAttribute;

    for (let i = 0; i < RING_POINTS; i++) {
      const angle = (i / RING_POINTS) * Math.PI * 2;
      const binIndex = Math.floor((i / RING_POINTS) * (frequencyData.length || 1));
      const energy = frequencyData.length > 0 ? frequencyData[binIndex] / 255 : 0;
      const r = BASE_RADIUS + energy * 0.8 + intensity * 0.3;

      pos.setXYZ(i, Math.cos(angle) * r, Math.sin(angle) * r, (energy - 0.5) * 0.6);
    }

    pos.needsUpdate = true;

    // Apply color to the Points material — safe: colorRef is instance-local
    const mat = pointsRef.current.material as THREE.PointsMaterial;
    mat.color.copy(colorRef.current);
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial size={0.04} sizeAttenuation vertexColors={false} color="#334155" />
    </points>
  );
}

// ─── ParticleField ────────────────────────────────────────────────────────────

const PARTICLE_COUNT = 600;

function ParticleField() {
  const pointsRef = useRef<THREE.Points>(null);
  const colorRef  = useRef(new THREE.Color());

  const { positions, velocities } = useMemo(() => {
    const positions  = new Float32Array(PARTICLE_COUNT * 3);
    const velocities = new Float32Array(PARTICLE_COUNT * 3);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      positions[i * 3]     = (Math.random() - 0.5) * 10;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 10;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
      velocities[i * 3]     = (Math.random() - 0.5) * 0.002;
      velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.002;
      velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.002;
    }
    return { positions, velocities };
  }, []);

  useFrame(() => {
    const { style, intensity } = useStore.getState();
    if (!pointsRef.current) return;

    const cfg = STYLE_CONFIG[style];
    colorRef.current.set(cfg.particleColor);

    const pos = pointsRef.current.geometry.attributes.position as THREE.BufferAttribute;
    const speedMult = 1 + intensity * 4;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      pos.array[i * 3]     += velocities[i * 3]     * speedMult;
      pos.array[i * 3 + 1] += velocities[i * 3 + 1] * speedMult;
      pos.array[i * 3 + 2] += velocities[i * 3 + 2] * speedMult;

      // Wrap particles back into bounds
      for (let axis = 0; axis < 3; axis++) {
        if (Math.abs(pos.array[i * 3 + axis]) > 5) {
          pos.array[i * 3 + axis] *= -0.9;
        }
      }
    }

    pos.needsUpdate = true;

    const mat = pointsRef.current.material as THREE.PointsMaterial;
    mat.color.copy(colorRef.current);
    mat.size = 0.02 + intensity * 0.04;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial size={0.02} sizeAttenuation color="#334155" />
    </points>
  );
}

// ─── StyleHUD ─────────────────────────────────────────────────────────────────

/**
 * Overlay showing style label, confidence bars, and intensity.
 *
 * Fix applied: uses useStore() hook (not getState()) so React re-renders
 * correctly when style/intensity/confidence change. This is intentional —
 * StyleHUD is a DOM overlay, not inside useFrame, so hook subscriptions
 * are the right pattern here.
 */
function StyleHUD() {
  const style      = useStore((s) => s.style);
  const intensity  = useStore((s) => s.intensity);
  const confidence = useStore((s) => s.confidence);

  const cfg = STYLE_CONFIG[style];

  const barStyle = (value: number, color: string): React.CSSProperties => ({
    height: '6px',
    width: `${Math.round(value * 100)}%`,
    backgroundColor: color,
    borderRadius: '3px',
    transition: 'width 80ms linear',
  });

  return (
    <div
      style={{
        position: 'absolute',
        top: 16,
        left: 16,
        color: '#f1f5f9',
        fontFamily: 'monospace',
        fontSize: '13px',
        pointerEvents: 'none',
        userSelect: 'none',
      }}
    >
      {/* Style label */}
      <div style={{ fontSize: '22px', fontWeight: 700, color: cfg.emissive, marginBottom: 8 }}>
        {style.toUpperCase()}
      </div>

      {/* Confidence bars */}
      {(['Shred', 'Ambient', 'Chords'] as const).map((s) => (
        <div key={s} style={{ marginBottom: 4 }}>
          <div style={{ marginBottom: 2 }}>
            {s} {Math.round((confidence as StyleConfidence)[s] * 100)}%
          </div>
          <div style={{ width: 140, backgroundColor: '#1e293b', borderRadius: 3 }}>
            <div style={barStyle((confidence as StyleConfidence)[s], STYLE_CONFIG[s].emissive)} />
          </div>
        </div>
      ))}

      {/* Intensity bar */}
      <div style={{ marginTop: 8 }}>
        <div style={{ marginBottom: 2 }}>Intensity {Math.round(intensity * 100)}%</div>
        <div style={{ width: 140, backgroundColor: '#1e293b', borderRadius: 3 }}>
          <div style={barStyle(intensity, '#94a3b8')} />
        </div>
      </div>
    </div>
  );
}

// ─── KatanaScene (root export) ────────────────────────────────────────────────

/**
 * Root R3F Canvas component.
 *
 * Fixes applied:
 * - 'use client' at top of file (required for hooks + Web Audio + R3F)
 * - OrbitControls imported from @react-three/drei (not three/examples)
 * - dpr={[1, 1.5]} is valid in R3F v9 — no change needed
 * - Canvas wrapped in a relative-positioned container so StyleHUD overlays correctly
 */
export default function KatanaScene() {
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Canvas
        dpr={[1, 1.5]}
        camera={{ position: [0, 0, 5], fov: 60 }}
        style={{ background: '#020617' }}
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[5, 5, 5]} intensity={1.2} />
        <pointLight position={[-5, -5, -5]} intensity={0.4} color="#60a5fa" />

        <KatanaOrb />
        <FrequencyRing />
        <ParticleField />

        {/* OrbitControls from @react-three/drei — correct import path */}
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.4} />
      </Canvas>

      {/* DOM overlay — outside Canvas so React state subscriptions work */}
      <StyleHUD />
    </div>
  );
}
