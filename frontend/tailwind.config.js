/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Risk level colors
        risk: {
          low: '#22c55e',      // green-500
          medium: '#f59e0b',   // amber-500
          high: '#ef4444',     // red-500
          imminent: '#dc2626', // red-600
        },
        // Emotional state colors
        emotion: {
          calm: '#3b82f6',       // blue-500
          anxious: '#f59e0b',    // amber-500
          distressed: '#f97316', // orange-500
          panicked: '#ef4444',   // red-500
        },
      },
      animation: {
        'pulse-alert': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
};
