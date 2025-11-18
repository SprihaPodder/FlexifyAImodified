import React from 'react';
import { BarChart3, RefreshCw, Moon, Sun } from 'lucide-react';
import { ThemeMode, HeaderProps } from '../types';

export const Header: React.FC<HeaderProps> = ({ onReset, theme, onThemeToggle }) => {
  const toggleTheme = () => {
    const newTheme: ThemeMode = theme === 'light' ? 'dark' : 'light';
    onThemeToggle(newTheme);
  };

  const isDark = theme === 'dark';

  return (
    <header className={`${isDark ? 'bg-gray-900/70 border-gray-800/20' : 'bg-white/70 border-white/20'} backdrop-blur-md border-b sticky top-0 z-50 transition-colors duration-200`}>
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl">
              <BarChart3 className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Flexify.ai
              </h1>
              <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                Flex the future of Machine Learning
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* Theme Toggle Button */}
            <button
              onClick={toggleTheme}
              className={`flex items-center justify-center p-2 rounded-xl border transition-all duration-200 ${
                isDark
                  ? 'bg-gray-800/50 hover:bg-gray-800/80 border-gray-700/30 hover:shadow-lg'
                  : 'bg-white/50 hover:bg-white/80 border-white/30 hover:shadow-lg'
              }`}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
            >
              {isDark ? (
                <Sun className="h-4 w-4 text-yellow-400" />
              ) : (
                <Moon className="h-4 w-4 text-gray-600" />
              )}
            </button>

            {/* Reset Button */}
            <button
              onClick={onReset}
              className={`flex items-center space-x-2 px-4 py-2 rounded-xl border transition-all duration-200 ${
                isDark
                  ? 'bg-gray-800/50 hover:bg-gray-800/80 border-gray-700/30 hover:shadow-lg'
                  : 'bg-white/50 hover:bg-white/80 border-white/30 hover:shadow-lg'
              }`}
            >
              <RefreshCw className={`h-4 w-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`} />
              <span className={`text-sm font-medium ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                Reset
              </span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};