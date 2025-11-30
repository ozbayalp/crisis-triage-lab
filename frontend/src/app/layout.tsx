import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'CrisisTriage AI - Dashboard',
  description: 'Real-time triage dashboard for mental health hotline conversations',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gray-950 text-gray-100">
          {/* Header */}
          <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm">
            <div className="container mx-auto px-4 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600" />
                  <h1 className="text-xl font-semibold">CrisisTriage AI</h1>
                </div>
                <nav className="flex items-center gap-6 text-sm text-gray-400">
                  <a href="/" className="hover:text-white transition-colors">
                    Live Triage
                  </a>
                  <a href="/calls" className="hover:text-white transition-colors">
                    📞 Calls
                  </a>
                  <a href="/analytics" className="hover:text-white transition-colors">
                    Analytics
                  </a>
                  <a href="/sessions" className="hover:text-white transition-colors">
                    Sessions
                  </a>
                </nav>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
