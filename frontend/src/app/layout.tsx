import type { Metadata } from 'next';
import './globals.css';
import { ThemeProvider } from './theme-provider';
import { ThemeToggle } from './theme-toggle';

export const metadata: Metadata = {
  title: 'CrisisTriage AI',
  description: 'Real-time triage system for mental health hotline conversations',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          <div 
            className="min-h-screen transition-colors duration-200"
            style={{ 
              background: 'var(--bg-primary)', 
              color: 'var(--text-primary)' 
            }}
          >
            {/* Header */}
            <header 
              className="sticky top-0 z-50 backdrop-blur-sm transition-colors duration-200"
              style={{ 
                borderBottom: '1px solid var(--border-primary)',
                background: 'var(--bg-primary)',
              }}
            >
              <div className="mx-auto max-w-[1280px] px-6 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div 
                      className="h-8 w-8 rounded-lg flex items-center justify-center"
                      style={{ 
                        background: 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)',
                        boxShadow: '0 2px 4px rgba(59, 130, 246, 0.3)'
                      }} 
                    >
                      <svg 
                        className="w-5 h-5" 
                        viewBox="0 0 24 24" 
                        fill="none" 
                        stroke="white" 
                        strokeWidth="2.5"
                        strokeLinecap="round" 
                        strokeLinejoin="round"
                      >
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                      </svg>
                    </div>
                    <span className="text-base font-semibold" style={{ letterSpacing: '-0.01em' }}>
                      CrisisTriage AI
                    </span>
                  </div>
                  <nav className="flex items-center gap-8">
                    <a 
                      href="/" 
                      className="text-sm transition-colors hover:opacity-70"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      Live Triage
                    </a>
                    <a 
                      href="/calls" 
                      className="text-sm transition-colors hover:opacity-70"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      Calls
                    </a>
                    <a 
                      href="/analytics" 
                      className="text-sm transition-colors hover:opacity-70"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      Analytics
                    </a>
                    <a 
                      href="/sessions" 
                      className="text-sm transition-colors hover:opacity-70"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      Sessions
                    </a>
                    <ThemeToggle />
                  </nav>
                </div>
              </div>
            </header>

            {/* Main Content */}
            <main className="mx-auto max-w-[1280px] px-6 py-8">
              {children}
            </main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
