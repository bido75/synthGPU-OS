/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        'bg-base': '#0a0a0f',
        'bg-surface': '#12121a',
        'bg-card': '#1a1a2e',
        'accent': '#00d4ff',
        'accent-dim': '#0099bb',
        'secondary': '#7c3aed',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'text-primary': '#f1f5f9',
        'text-secondary': '#94a3b8',
        'border': '#2a2a3e',
      },
      animation: {
        'pulse-cyan': 'pulse-cyan 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'scan': 'scan 3s linear infinite',
      },
      keyframes: {
        'pulse-cyan': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.4 },
        },
        'glow': {
          'from': { boxShadow: '0 0 5px #00d4ff44' },
          'to': { boxShadow: '0 0 20px #00d4ff88, 0 0 40px #00d4ff44' },
        },
        'scan': {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
      },
    },
  },
  plugins: [],
}
