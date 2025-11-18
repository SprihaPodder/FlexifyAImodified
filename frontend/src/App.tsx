import React, { useState, useEffect } from 'react';
import { FileUpload } from './components/FileUpload';
import { WorkflowSelector } from './components/WorkflowSelector';
import { EDAWorkflow } from './components/EDAWorkflow';
import { MLWorkflow } from './components/MLWorkflow';
import { Header } from './components/Header';
import { Dataset, WorkflowType, ThemeMode } from './types';
import { TextWorkflow } from './components/TextWorkflow';
import { ImageWorkflow } from './components/ImageWorkflow';
import CsvFeatureEngineer from './components/CsvFeatureEngineer';

function App() {
  const [uploadedDataset, setUploadedDataset] = useState<Dataset | null>(null);
  const [selectedWorkflow, setSelectedWorkflow] = useState<WorkflowType>(null);
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const savedTheme = localStorage.getItem('theme') as ThemeMode | null;
    return savedTheme || 'light';
  });


  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  const handleFileUpload = (dataset: Dataset) => {
    setUploadedDataset(dataset);
    setSelectedWorkflow(null);
  };

  const handleWorkflowSelect = (workflow: WorkflowType) => {
    setSelectedWorkflow(workflow);
  };

  const handleReset = () => {
    setUploadedDataset(null);
    setSelectedWorkflow(null);
  };

  const handleThemeToggle = (newTheme: ThemeMode) => {
    setTheme(newTheme);
  };

  const isDark = theme === 'dark';

  return (
    <div className={`min-h-screen transition-colors duration-200 ${
      isDark
        ? 'bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 relative overflow-hidden'
        : 'bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden'
    }`}>
      {/* Abstract Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {isDark ? (
          <>
            <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-900/20 to-purple-900/20 rounded-full blur-3xl"></div>
            <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-gradient-to-tr from-indigo-900/20 to-cyan-900/20 rounded-full blur-3xl"></div>
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-to-r from-purple-900/10 to-pink-900/10 rounded-full blur-2xl"></div>
          </>
        ) : (
          <>
            <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-purple-600/20 rounded-full blur-3xl"></div>
            <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-gradient-to-tr from-indigo-400/20 to-cyan-600/20 rounded-full blur-3xl"></div>
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-to-r from-purple-400/10 to-pink-600/10 rounded-full blur-2xl"></div>
          </>
        )}
      </div>

      <div className="relative z-10">
        <Header onReset={handleReset} theme={theme} onThemeToggle={handleThemeToggle} />

        <main className={`container mx-auto px-4 py-8 ${isDark ? 'text-gray-50' : 'text-gray-900'}`}>
          {!uploadedDataset ? (
            <FileUpload onFileUpload={handleFileUpload} />
          ) : uploadedDataset.type === 'csv' && !selectedWorkflow ? (
            <WorkflowSelector
              fileName={uploadedDataset.file.name}
              onWorkflowSelect={handleWorkflowSelect}
            />
          ) : uploadedDataset.type === 'csv' && selectedWorkflow === 'eda' ? (
            <EDAWorkflow file={uploadedDataset.file} onBack={() => setSelectedWorkflow(null)} />
          ) : uploadedDataset.type === 'csv' && selectedWorkflow === 'ml' ? (
            <MLWorkflow file={uploadedDataset.file} onBack={() => setSelectedWorkflow(null)} />
          ) : uploadedDataset.type === 'csv' && selectedWorkflow === 'fe' ? (
            <CsvFeatureEngineer initialFile={uploadedDataset.file} onBack={() => setSelectedWorkflow(null)} />
          ) : uploadedDataset.type === 'text' ? (
            <TextWorkflow file={uploadedDataset.file} onBack={handleReset} />
          ) : uploadedDataset.type === 'image' ? (
            <ImageWorkflow file={uploadedDataset.file} onBack={handleReset} />
          ) : (
            <div className={`text-center py-12 rounded-xl p-8 ${
              isDark
                ? 'bg-gray-800/50 border border-gray-700/50'
                : 'bg-white/50 border border-white/50'
            }`}>
              <h2 className={`text-2xl font-bold mb-4 ${
                isDark ? 'text-gray-100' : 'text-gray-800'
              }`}>
                Unsupported File Type
              </h2>
              <p className={`mb-6 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                The selected file type is not supported.
              </p>
              <button
                onClick={handleReset}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  isDark
                    ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-blue-500/50'
                    : 'bg-blue-500 hover:bg-blue-600 text-white shadow-lg hover:shadow-blue-400/50'
                }`}
              >
                Upload Different File
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;



