// ImageWorkflow.tsx
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Eye, Layers, Wand2, Microscope } from 'lucide-react';

interface ImageWorkflowProps {
  file: File;
  onBack: () => void;
}

const BACKEND_URL = "http://127.0.0.1:8000";

export const ImageWorkflow: React.FC<ImageWorkflowProps> = ({ file, onBack }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<any | null>(null); 
  const [features, setFeatures] = useState<any | null>(null); 
  const [explain, setExplain] = useState<any | null>(null); 
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState<string>('basic');

  const [suggestedActions, setSuggestedActions] = useState<any[] | null>(null);
  const [actionResults, setActionResults] = useState<Record<string, any>>({});

  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const postFile = async (endpoint: string, formFields?: Record<string,string|number>) => {
    const formData = new FormData();
    formData.append("file", file);
    if (formFields) {
      for (const k of Object.keys(formFields)) {
        formData.append(k, String((formFields as any)[k]));
      }
    }

    const res = await fetch(`${BACKEND_URL}${endpoint}`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      let txt = await res.text();
      try {
        const j = JSON.parse(txt);
        throw new Error(j.error || j.detail || txt);
      } catch {
        throw new Error(`Server returned ${res.status}: ${txt}`);
      }
    }

    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
  };

  const runBasic = async () => {
    const data = await postFile("/process-image/");
    setResults(data);
    setFeatures(null);
    setExplain(null);
    setSuggestedActions(null);
    setActionResults({});
  };

  const runFeatures = async () => {
    const data = await postFile("/extract-image-features/", { k_colors: 4 });
    setFeatures(data);
    setResults(null);
    setExplain(null);
    setSuggestedActions(null);
    setActionResults({});
  };

  const runExplain = async (level: "short" | "detailed" = "short") => {
    const data = await postFile("/explain-image/", { explanation_level: level });
    setExplain(data);
    setResults(null);
    setFeatures(null);
    setSuggestedActions(null);
    setActionResults({});
  };

  const runAdvanced = async () => {
    const data = await postFile("/advanced-analysis/", { k_colors: 6 });
    if (data.features) setFeatures(data.features);
    if (data.explanation) setExplain(data.explanation);
    if (data.ocr_text) {
      setResults({ dimensions: { width: data.features?.width, height: data.features?.height }, format: data.features?.format || "PNG", mode: data.features?.mode || "RGB", extracted_text: data.ocr_text });
    } else {
      setResults(null);
    }

    setSuggestedActions(Array.isArray(data.suggested_actions) ? data.suggested_actions : []);
    setActionResults({});
  };

  const onAnalyze = async () => {
    setError(null);
    setIsProcessing(true);

    try {
      if (selectedAnalysis === 'basic') {
        await runBasic();
      } else if (selectedAnalysis === 'features') {
        await runFeatures();
      } else if (selectedAnalysis === 'explain') {
        await runExplain("short");
      } else if (selectedAnalysis === 'advanced') {
        await runAdvanced();
      }
    } catch (err: any) {
      setError(err?.message || "Failed to run analysis");
    } finally {
      setIsProcessing(false);
    }
  };

  const runSuggestedAction = async (action: any) => {
    setError(null);
    setIsProcessing(true);
    try {
      const data = await postFile(action.endpoint);
      setActionResults(prev => ({ ...prev, [action.id]: data }));
    } catch (err: any) {
      setError(err?.message || "Suggested action failed");
    } finally {
      setIsProcessing(false);
    }
  };

  const renderAnalysisOptions = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <button
        onClick={() => setSelectedAnalysis('basic')}
        className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${selectedAnalysis === 'basic' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-300'}`}
      >
        <Eye className="h-6 w-6 mb-2" />
        <span className="font-medium">Basic Analysis</span>
      </button>

      <button
        onClick={() => setSelectedAnalysis('features')}
        className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${selectedAnalysis === 'features' ? 'border-purple-500 bg-purple-50' : 'border-gray-200 hover:border-purple-300'}`}
      >
        <Layers className="h-6 w-6 mb-2" />
        <span className="font-medium">Feature Extraction</span>
      </button>

      <button
        onClick={() => setSelectedAnalysis('explain')}
        className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${selectedAnalysis === 'explain' ? 'border-green-500 bg-green-50' : 'border-gray-200 hover:border-green-300'}`}
      >
        <Wand2 className="h-6 w-6 mb-2" />
        <span className="font-medium">Explainable AI</span>
      </button>

      <button
        onClick={() => setSelectedAnalysis('advanced')}
        className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${selectedAnalysis === 'advanced' ? 'border-indigo-500 bg-indigo-50' : 'border-gray-200 hover:border-indigo-300'}`}
      >
        <Microscope className="h-6 w-6 mb-2" />
        <span className="font-medium">Advanced Analysis</span>
      </button>
    </div>
  );

  const renderResults = () => {
    if (!results && !features && !explain && (!suggestedActions || suggestedActions.length === 0)) return null;

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

          {/* Basic results */}
          {results && (
            <>
              <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-white/30 shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Image Properties</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Dimensions</span>
                    <span className="font-medium">{results.dimensions?.width} × {results.dimensions?.height}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Format</span>
                    <span className="font-medium">{results.format}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Color Mode</span>
                    <span className="font-medium">{results.mode}</span>
                  </div>
                </div>
              </div>

              {results.extracted_text && (
                <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-white/30 shadow-sm col-span-2">
                  <h3 className="text-lg font-semibold mb-4">Extracted Text (OCR)</h3>
                  <pre className="whitespace-pre-wrap text-sm">{results.extracted_text}</pre>
                </div>
              )}
            </>
          )}

          {/* Feature extraction UI */}
          {features && (
            <div className="col-span-3 space-y-4">
              <div className="bg-white/80 rounded-xl p-6 border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Feature Summary</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-gray-600">Avg Brightness</div>
                    <div className="font-medium">{features.avg_brightness}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Contrast</div>
                    <div className="font-medium">{features.contrast}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Dimensions</div>
                    <div className="font-medium">{features.width} × {features.height}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Format</div>
                    <div className="font-medium">{features.format}</div>
                  </div>
                </div>
              </div>

              <div className="bg-white/80 rounded-xl p-6 border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Dominant Colors</h3>
                <div className="flex gap-3 items-center">
                  {features.dominant_colors?.map((c: string, i: number) => (
                    <div key={i} className="flex flex-col items-center">
                      <div style={{background: c, width: 48, height: 48, borderRadius: 8, border: "1px solid rgba(0,0,0,0.08)"}}/>
                      <div className="text-xs mt-2">{c}</div>
                    </div>
                  ))}
                </div>

                <div className="mt-6">
                  <h4 className="text-sm text-gray-700 mb-2">Palette</h4>
                  {features.palette_image && <img src={`data:image/png;base64,${features.palette_image}`} alt="palette" className="h-12 rounded-md shadow-sm" />}
                </div>

                <div className="mt-6">
                  <h4 className="text-sm text-gray-700 mb-2">Color Histogram</h4>
                  {features.histogram_image && <img src={`data:image/png;base64,${features.histogram_image}`} alt="hist" className="w-full max-w-xl" />}
                </div>

                <div className="mt-6">
                  <h4 className="text-sm text-gray-700 mb-2">Edge Map</h4>
                  {features.edge_image && <img src={`data:image/png;base64,${features.edge_image}`} alt="edges" className="w-full max-w-xl" />}
                </div>
              </div>
            </div>
          )}

          {/* Explanation UI */}
          {explain && (
            <div className="col-span-3 space-y-4">
              <div className="bg-white/80 rounded-xl p-6 border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Explainable Summary</h3>
                <div className="text-gray-700">
                  {explain.explanation_steps?.map((s: string, i: number) => <p key={i} className="mb-2">{s}</p>)}
                </div>
              </div>

              <div className="bg-white/80 rounded-xl p-6 border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Saliency / Edge Map</h3>
                {explain.saliency_image && <img src={`data:image/png;base64,${explain.saliency_image}`} alt="saliency" className="w-full max-w-xl" />}
                <div className="mt-4">
                  <h4 className="text-sm text-gray-700">Feature Summary</h4>
                  <pre className="text-xs bg-gray-50 p-3 rounded">{JSON.stringify(explain.feature_summary || explain, null, 2)}</pre>
                </div>
              </div>
            </div>
          )}

        </div>

        {/* Advanced: Suggested actions UI */}
        { suggestedActions && suggestedActions.length > 0 && (
          <div className="mt-6 bg-white/80 rounded-xl p-6 border shadow-sm">
            <h3 className="text-lg font-semibold mb-4">Recommended follow-up analyses</h3>
            <div className="grid gap-4">
              {suggestedActions.map((a: any) => (
                <div key={a.id} className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 p-3 border rounded">
                  <div>
                    <div className="font-medium">{a.label}</div>
                    <div className="text-sm text-gray-600">{a.description}</div>
                    <div className="text-xs text-gray-400 mt-1">Endpoint: {a.endpoint}</div>
                  </div>

                  <div className="flex flex-col items-end gap-2">
                    <button
                      onClick={() => runSuggestedAction(a)}
                      disabled={isProcessing}
                      className="px-3 py-2 bg-blue-600 text-white rounded-md"
                    >
                      Run
                    </button>
                    {actionResults[a.id] && (
                      <button
                        onClick={() => setActionResults(prev => { const p = {...prev}; delete p[a.id]; return p; })}
                        className="text-xs mt-2 text-gray-600 underline"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Render action results */}
            <div className="mt-4 space-y-3">
              {Object.keys(actionResults).map(key => (
                <div key={key} className="bg-gray-50 p-3 rounded">
                  <div className="font-medium mb-1">{key}</div>
                  <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(actionResults[key], null, 2)}</pre>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center mb-8">
        <button onClick={onBack} className="flex items-center px-4 py-2 bg-white/50 rounded-xl border mr-6">
          <ArrowLeft className="h-4 w-4 mr-2" /> Back
        </button>
        <div>
          <h1 className="text-3xl font-bold">Image Analysis</h1>
          <p className="text-gray-600">Processing: {file.name}</p>
        </div>
      </div>

      <div className="space-y-8">
        <div className="bg-white/70 rounded-2xl p-8 border">
          {preview && (
            <div className="mb-8">
              <img src={preview} alt="Preview" className="w-full h-auto max-w-2xl mx-auto" />
            </div>
          )}

          {renderAnalysisOptions()}

          <div className="text-center">
            <button
              onClick={onAnalyze}
              disabled={isProcessing}
              className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl"
            >
              {isProcessing ? "Processing..." : "Analyze Image"}
            </button>

            {error && <div className="mt-4 text-red-500 bg-red-50 p-4 rounded-lg">{error}</div>}
          </div>

          {renderResults()}
        </div>
      </div>
    </div>
  );
};