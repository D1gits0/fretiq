# Katana-Vision Project Specs

## Tech Stack
- Framework: Next.js 16 (App Router)
- 3D: React Three Fiber + Drei
- AI: TensorFlow.js (Style Prediction)
- Audio/MIDI: Web Audio API + Web MIDI API
- State: Zustand

## Coding Standards
- Use Feature-based architecture (src/features/).
- Logic for 3D must stay in the `guitar-3d` feature folder.
- Audio math should run in Web Workers where possible to save FPS.
- Style: TypeScript, functional components, Tailwind CSS.

## Commands
- Dev: `npm run dev`
- Build: `npm run build`
- Typecheck: `npm test`