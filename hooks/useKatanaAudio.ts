import { useEffect, useRef, useCallback } from 'react';
import { useStore } from '@/store/useStore';
 
// ─── Config ───────────────────────────────────────────────────────────────────
 
/**
 * FFT_SIZE controls the resolution of the frequency analysis.
 * Must be a power of 2. Higher = more detail, more CPU.
 *
 * 2048 → 1024 frequency bins, ~21Hz resolution per bin at 44.1kHz
 * Great for distinguishing guitar frequency ranges:
 *   - Bass/fundamentals:  80–300 Hz  (bins 2–14)
 *   - Mids/body:         300–2k Hz  (bins 14–93)
 *   - Presence/attack:    2k–6k Hz  (bins 93–280)
 *   - Air/harmonics:      6k–20kHz  (bins 280–1024)
 */
const FFT_SIZE = 2048;
 
/**
 * SMOOTHING: 0 = jumpy/raw, 1 = totally smoothed (laggy).
 * 0.75 is a good balance for musical responsiveness.
 */
const SMOOTHING = 0.75;
 
// ─── Types ────────────────────────────────────────────────────────────────────
 
interface AudioNodes {
  context: AudioContext;
  source: MediaStreamAudioSourceNode;
  analyser: AnalyserNode;
  stream: MediaStream;
}
 
// ─── Hook ─────────────────────────────────────────────────────────────────────
 
/**
 * useKatanaAudio
 *
 * Connects to the Boss Katana Gen 3 USB-C audio interface, sets up an
 * AnalyserNode, and pumps frequency data into the Zustand store every
 * animation frame. The store feeds the AI layer and R3F scene.
 *
 * Usage:
 *   const { startListening, stopListening } = useKatanaAudio();
 */
export function useKatanaAudio() {
  const nodesRef = useRef<AudioNodes | null>(null);
  const rafRef = useRef<number | null>(null);
  const bufferRef = useRef<Uint8Array | null>(null);
 
  const setListening = useStore((s) => s.setListening);
  const setAudioError = useStore((s) => s.setAudioError);
  const setFrequencyData = useStore((s) => s.setFrequencyData);
  const setIntensity = useStore((s) => s.setIntensity);
 
  // ── Teardown ──────────────────────────────────────────────────────────────
 
  const stopListening = useCallback(() => {
    // Cancel animation loop
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
 
    // Disconnect Web Audio nodes + stop media stream tracks
    if (nodesRef.current) {
      const { source, analyser, context, stream } = nodesRef.current;
      source.disconnect();
      analyser.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      // Close the AudioContext to release OS audio resources
      context.close().catch(() => {});
      nodesRef.current = null;
    }
 
    setListening(false);
  }, [setListening]);
 
  // ── Per-frame analysis loop ───────────────────────────────────────────────
 
  const tick = useCallback(() => {
    if (!nodesRef.current || !bufferRef.current) return;
 
    const { analyser } = nodesRef.current;
    const buffer = bufferRef.current;
 
    // Fill buffer with frequency data (0–255 per bin)
    // This is the FFT snapshot the AI will consume
    analyser.getByteFrequencyData(buffer as Uint8Array<ArrayBuffer>);
 
    // ── Compute RMS intensity from time-domain data ────────────────────────
    // We use a *separate* time-domain buffer for intensity so frequency
    // bins stay clean for the AI.
    const timeDomain = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(timeDomain as Uint8Array<ArrayBuffer>);
 
    // RMS: sqrt(mean of squared deviations from center (128))
    let sumOfSquares = 0;
    for (let i = 0; i < timeDomain.length; i++) {
      const deviation = (timeDomain[i] - 128) / 128; // normalize to [-1, 1]
      sumOfSquares += deviation * deviation;
    }
    const rms = Math.sqrt(sumOfSquares / timeDomain.length);
 
    // Push to store — pass a *new* Uint8Array so React detects the change
    setFrequencyData(new Uint8Array(buffer) as Uint8Array);
    setIntensity(Math.min(1, rms * 10));
 
    // Schedule next frame
    rafRef.current = requestAnimationFrame(tick);
  }, [setFrequencyData, setIntensity]);
 
  // ── Startup ───────────────────────────────────────────────────────────────
 
  const startListening = useCallback(async () => {
    // Guard: don't double-init
    if (nodesRef.current) return;
    setAudioError(null);
 
    try {
      /**
       * getUserMedia audio constraints:
       *
       * - echoCancellation: false  → we want the raw amp signal, not processed voice
       * - noiseSuppression: false  → same reason; noise suppression mangles guitar tone
       * - autoGainControl: false   → we need accurate amplitude for intensity tracking
       * - sampleRate: 44100        → CD quality; Katana Gen 3 supports up to 96kHz but
       *                               44.1kHz is the Web Audio default and sufficient for guitar
       *
       * The browser will show a device picker if multiple audio inputs exist.
       * User must select "Boss Katana Gen 3" (or similar USB name).
       * TODO: filter by deviceId for production — see selectKatanaDevice() below.
       */
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: 44100,
        },
        video: false,
      });
 
      // Build the Web Audio graph
      const context = new AudioContext({ sampleRate: 44100 });
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
 
      analyser.fftSize = FFT_SIZE;
      analyser.smoothingTimeConstant = SMOOTHING;
 
      // source → analyser (do NOT connect analyser → destination,
      // or you'll hear the amp signal doubled through your speakers)
      source.connect(analyser);
 
      nodesRef.current = { context, source, analyser, stream };
 
      // Allocate the reusable frequency buffer
      // frequencyBinCount = fftSize / 2 = 1024 bins
      bufferRef.current = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount));
 
      setListening(true);
 
      // Kick off the animation loop
      rafRef.current = requestAnimationFrame(tick);
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : 'Unknown audio error';
 
      // Common causes:
      // - "NotAllowedError"  → user denied mic permission
      // - "NotFoundError"    → no audio input devices found
      // - "NotReadableError" → device in use by another app
      setAudioError(`Audio init failed: ${msg}`);
      setListening(false);
    }
  }, [setAudioError, setListening, tick]);
 
  // ── Cleanup on unmount ────────────────────────────────────────────────────
 
  useEffect(() => {
    return () => {
      stopListening();
    };
  }, [stopListening]);
 
  // ── Expose analyser node for direct TF.js access if needed ───────────────
 
  const getAnalyser = useCallback((): AnalyserNode | null => {
    return nodesRef.current?.analyser ?? null;
  }, []);
 
  return { startListening, stopListening, getAnalyser };
}
 
// ─── Utility: enumerate & select the Katana device by name ───────────────────
 
/**
 * selectKatanaDevice
 *
 * In production, call this *before* getUserMedia to pin to the correct deviceId.
 * Avoids the browser's generic picker dialog.
 *
 * Usage:
 *   const deviceId = await selectKatanaDevice();
 *   navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: deviceId } } });
 */
export async function selectKatanaDevice(): Promise<string | undefined> {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const katana = devices.find(
    (d) =>
      d.kind === 'audioinput' &&
      d.label.toLowerCase().includes('katana') // "BOSS KATANA Gen 3" etc.
  );
  return katana?.deviceId;
}