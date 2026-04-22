'use client';

import { useRef, useMemo } from 'react';
import type React from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useStore, GuitarStyle } from '@/store/useStore';
import Fretboard, { NoteHUD } from '@/components/Fretboard';

// ─── Per-style visual config ──────────────────────────────────────────────────

interface StyleConfig {
  emissive: string;
  particleColor: string;
}

const STYLE_CONFIG: Record<GuitarStyle, StyleConfig> = {
  Idle:    { emissive: '#0f172a', particleColor: '#1e293b' },
  Shred:   { emissive: '#ea580c', particleColor: '#7c2d12' },
  Ambient: { emissive: '#0e7490', particleColor: '#164e63' },
  Chords:  { emissive: '#15803d', particleColor: '#14532d' },
};

// ─── FrequencyRing ────────────────────────────────────────────────────────────
// Orbits the fretboard as a reactive halo

const RING_POINTS = 128;
const BASE_RADIUS = 3.2;

function FrequencyRing() {
  const pointsRef = useRef<THREE.Points>(null);
  const colorRef  = useRef(new THREE.Color());
  const positions = useMemo(() => new Float32Array(RING_POINTS * 3), []);

  useFrame(() => {
    const { frequencyData, style, intensity } = useStore.getState();
    if (!pointsRef.current) return;

    colorRef.current.set(STYLE_CONFIG[style].particleColor);

    const pos = pointsRef.current.geometry.attributes.position as THREE.BufferAttribute;

    for (let i = 0; i < RING_POINTS; i++) {
      const angle    = (i / RING_POINTS) * Math.PI * 2;
      const binIndex = Math.floor((i / RING_POINTS) * (frequencyData.length || 1));
      const energy   = frequencyData.length > 0 ? frequencyData[binIndex] / 255 : 0;
      const r        = BASE_RADIUS + energy * 0.8 + intensity * 0.3;

      // Ring lies in the XZ plane (horizontal, around the fretboard)
      pos.setXYZ(i, Math.cos(angle) * r, (energy - 0.5) * 0.8, Math.sin(angle) * r);
    }

    pos.needsUpdate = true;

    const mat = pointsRef.current.material as THREE.PointsMaterial;
    mat.color.copy(colorRef.current);
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          usage={THREE.DynamicDrawUsage}
        />
      </bufferGeometry>
      <pointsMaterial size={0.04} sizeAttenuation transparent opacity={0.3} color="#334155" />
    </points>
  );
}

// ─── ParticleField ────────────────────────────────────────────────────────────

const PARTICLE_COUNT = 200;

function ParticleField() {
  const pointsRef = useRef<THREE.Points>(null);
  const colorRef  = useRef(new THREE.Color());

  const { positions, velocities } = useMemo(() => {
    const positions  = new Float32Array(PARTICLE_COUNT * 3);
    const velocities = new Float32Array(PARTICLE_COUNT * 3);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      positions[i * 3]     = (Math.random() - 0.5) * 14;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 8;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 8;
      velocities[i * 3]     = (Math.random() - 0.5) * 0.002;
      velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.002;
      velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.002;
    }
    return { positions, velocities };
  }, []);

  useFrame(() => {
    const { style, intensity } = useStore.getState();
    if (!pointsRef.current) return;

    colorRef.current.set(STYLE_CONFIG[style].particleColor);

    const pos       = pointsRef.current.geometry.attributes.position as THREE.BufferAttribute;
    const speedMult = 1 + intensity * 4;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      pos.array[i * 3]     += velocities[i * 3]     * speedMult;
      pos.array[i * 3 + 1] += velocities[i * 3 + 1] * speedMult;
      pos.array[i * 3 + 2] += velocities[i * 3 + 2] * speedMult;

      // Wrap to opposite edge — keeps field evenly distributed
      if (pos.array[i * 3]     >  7) pos.array[i * 3]     = -7;
      if (pos.array[i * 3]     < -7) pos.array[i * 3]     =  7;
      if (pos.array[i * 3 + 1] >  4) pos.array[i * 3 + 1] = -4;
      if (pos.array[i * 3 + 1] < -4) pos.array[i * 3 + 1] =  4;
      if (pos.array[i * 3 + 2] >  4) pos.array[i * 3 + 2] = -4;
      if (pos.array[i * 3 + 2] < -4) pos.array[i * 3 + 2] =  4;
    }

    pos.needsUpdate = true;

    const mat  = pointsRef.current.material as THREE.PointsMaterial;
    mat.color.copy(colorRef.current);
    mat.size = 0.008 + intensity * 0.008; // smaller, subtler
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.008} sizeAttenuation color="#1e293b" />
    </points>
  );
}

// ─── StyleHUD ─────────────────────────────────────────────────────────────────

function StyleHUD() {
  const style     = useStore((s) => s.style);
  const intensity = useStore((s) => s.intensity);

  const cfg = STYLE_CONFIG[style];

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 80,
        left: 16,
        fontFamily: 'monospace',
        fontSize: '12px',
        color: '#94a3b8',
        pointerEvents: 'none',
        userSelect: 'none',
      }}
    >
      <div style={{ fontSize: '18px', fontWeight: 700, color: cfg.emissive, marginBottom: 4 }}>
        {style.toUpperCase()}
      </div>
      <div style={{ width: 100, height: 4, background: '#1e293b', borderRadius: 2 }}>
        <div
          style={{
            height: '100%',
            width: `${Math.round(intensity * 100)}%`,
            background: '#94a3b8',
            borderRadius: 2,
            transition: 'width 80ms linear',
          }}
        />
      </div>
    </div>
  );
}

// ─── KatanaScene ──────────────────────────────────────────────────────────────

export default function KatanaScene() {
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Canvas
        dpr={[1, 1.5]}
        // Camera pulled back and slightly elevated to view the fretboard
        // from a guitarist's-eye perspective
        camera={{ position: [0, 3.5, 6], fov: 55 }}
        style={{ background: '#020617' }}
      >
        <ambientLight intensity={0.25} />
        {/* Key light from above-front */}
        <pointLight position={[0, 4, 4]}  intensity={1.0} color="#f8fafc" />
        {/* Fill light from below — lifts shadows on fret markers */}
        <pointLight position={[0, -2, 2]} intensity={0.3} color="#60a5fa" />
        {/* Rim light from behind */}
        <pointLight position={[0, 1, -5]} intensity={0.4} color="#818cf8" />

        <Fretboard />
        <FrequencyRing />
        <ParticleField />

        <OrbitControls
          enableZoom={false}
          enablePan={false}
          // Constrain vertical orbit so the fretboard stays readable
          minPolarAngle={Math.PI / 6}
          maxPolarAngle={Math.PI / 2.2}
          autoRotate={false}
        />
      </Canvas>

      {/* DOM overlays — outside Canvas */}
      <NoteHUD />
      <StyleHUD />
    </div>
  );
}
