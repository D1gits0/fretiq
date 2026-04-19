import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Katana Vision',
  description: 'Real-time 3D visualizer for Boss Katana Gen 3',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: '#020617', overflow: 'hidden' }}>
        {children}
      </body>
    </html>
  );
}
