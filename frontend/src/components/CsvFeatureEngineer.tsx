import React, { useState } from "react";
import { ChevronDown, ChevronUp, Download, Zap, TrendingUp } from "lucide-react";

type SummaryType = {
  rows_before: number | null;
  rows_after: number | null;
  rows_removed: number | null;
  missing_values_before: number | null;
  missing_values_after: number | null;
  missing_values_handled: number | null;
  outliers_removed: number | null;
  columns_processed: number | null;
  column_names?: string[];
  applied_methods?: Record<string, any>;
  generated_features?: string[];
};

type FeatureSuggestion = {
  name: string;
  type: string;
  description: string;
  confidence: number;
  columns_involved: string[];
};

type Props = {
  initialFile?: File | null;
  onBack?: () => void;
};

export default function CsvFeatureEngineer({ initialFile = null, onBack }: Props) {
  const [file, setFile] = useState<File | null>(initialFile);
  const [targetColumn, setTargetColumn] = useState("");
  const [returnAsFile, setReturnAsFile] = useState(true);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<SummaryType | null>(null);
  const [handleMissing, setHandleMissing] = useState<'mean'|'drop'|'none'>('mean');
  const [handleOutliers, setHandleOutliers] = useState<'none'|'zscore'|'iqr'>('none');
  const [featureSuggestions, setFeatureSuggestions] = useState<FeatureSuggestion[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<Set<string>>(new Set());
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);

  const backendUrl = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    setFile(e.target.files?.[0] ?? null);
    setMessage(null);
    setError(null);
    setSummary(null);
    setFeatureSuggestions([]);
    setSelectedFeatures(new Set());
  }

  async function fetchFeatureSuggestions() {
    if (!file) {
      setError("Please choose a CSV file first.");
      return;
    }

    setSuggestionsLoading(true);
    setError(null);
    setFeatureSuggestions([]);

    try {
      const form = new FormData();
      form.append("file", file);
      if (targetColumn) form.append("target_column", targetColumn);

      const resp = await fetch(`${backendUrl}/analyze-features/`, {
        method: "POST",
        body: form,
      });

      if (!resp.ok) {
        let jsonErr = null;
        try {
          jsonErr = await resp.json();
        } catch {
          /* ignore */
        }
        const msg = jsonErr?.error || `Server returned ${resp.status}`;
        throw new Error(msg);
      }

      const data = await resp.json();
      setFeatureSuggestions(data.suggestions || []);
      setShowSuggestions(true);
      setMessage(null);
    } catch (err: any) {
      setError(String(err.message || err));
    } finally {
      setSuggestionsLoading(false);
    }
  }

  function toggleFeatureSelection(featureName: string) {
    const updated = new Set(selectedFeatures);
    if (updated.has(featureName)) {
      updated.delete(featureName);
    } else {
      updated.add(featureName);
    }
    setSelectedFeatures(updated);
  }

  function selectAllFeatures() {
    if (selectedFeatures.size === featureSuggestions.length) {
      setSelectedFeatures(new Set());
    } else {
      setSelectedFeatures(new Set(featureSuggestions.map(f => f.name)));
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setMessage(null);
    setSummary(null);

    if (!file) {
      setError("Please choose a CSV file first.");
      return;
    }

    setLoading(true);

    try {
      const form = new FormData();
      form.append("file", file);
      if (targetColumn) form.append("target_column", targetColumn);
      form.append("return_as_file", returnAsFile ? "true" : "false");
      form.append("handle_missing", handleMissing);
      form.append("handle_outliers", handleOutliers);
      form.append("selected_features", JSON.stringify(Array.from(selectedFeatures)));

      const resp = await fetch(`${backendUrl}/feature-engineering/csv`, {
        method: "POST",
        body: form,
      });

      if (!resp.ok) {
        let jsonErr = null;
        try {
          jsonErr = await resp.json();
        } catch {
          /* ignore */
        }
        const msg = jsonErr?.error || jsonErr?.message || `Server returned ${resp.status}`;
        throw new Error(msg);
      }

      const contentType = (resp.headers.get("content-type") || "").toLowerCase();

      if (contentType.includes("application/json") || contentType.includes("text/json")) {
        const data = await resp.json();
        if (data.summary) setSummary(data.summary);
        if (data.processed_csv) {
          const blob = new Blob([data.processed_csv], { type: "text/csv;charset=utf-8;" });
          const filename = `processed_${file.name}`;
          downloadBlob(blob, filename);
          setMessage(`✓ Processed CSV downloaded: ${filename}`);
        } else {
          setMessage(data.message || "Feature engineering completed (no CSV returned).");
        }
      } else {
        const text = await resp.text();
        setMessage("✓ Processed CSV received. Downloading...");
        try {
          const lines = text.split(/\r?\n/).filter(Boolean);
          const header = lines[0] || "";
          const colsCount = header ? header.split(",").length : null;
          const rowsAfter = lines.length - (header ? 1 : 0);
          setSummary({
            rows_before: null,
            rows_after: rowsAfter,
            rows_removed: null,
            missing_values_before: null,
            missing_values_after: null,
            missing_values_handled: null,
            outliers_removed: null,
            columns_processed: colsCount ?? 0,
            column_names: header ? header.split(",").map(s => s.trim()) : [],
            applied_methods: { note: "Server returned raw CSV (not JSON summary)" }
          });
        } catch {}
        const blob = new Blob([text], { type: "text/csv;charset=utf-8;" });
        downloadBlob(blob, `processed_${file.name}`);
      }
    } catch (err: any) {
      setError(String(err.message || err));
    } finally {
      setLoading(false);
    }
  }

  function downloadBlob(blob: Blob, filename: string) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600 bg-green-50";
    if (confidence >= 0.6) return "text-blue-600 bg-blue-50";
    return "text-amber-600 bg-amber-50";
  };

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.8) return "High";
    if (confidence >= 0.6) return "Medium";
    return "Low";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50 p-6">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
              <Zap className="h-8 w-8 text-cyan-600" />
              Feature Engineering Studio
            </h2>
            <p className="text-gray-600 mt-1">Clean, enhance, and intelligently engineer your dataset</p>
          </div>
          {onBack && (
            <button
              onClick={onBack}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              ← Back
            </button>
          )}
        </div>

        {/* Main Form Card */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden mb-6">
          {/* File Input Section */}
          <div className="p-8 border-b border-gray-200">
            <label className="block text-sm font-semibold text-gray-700 mb-3">Dataset File</label>
            <div className="relative">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="w-full px-4 py-3 border-2 border-dashed border-cyan-300 rounded-lg focus:outline-none focus:border-cyan-600 hover:border-cyan-500 transition-colors"
              />
            </div>
            {file && (
              <div className="mt-3 flex items-center gap-2 text-sm text-green-700 bg-green-50 px-4 py-2 rounded-lg">
                <span className="font-medium">✓ Selected:</span>
                <span className="font-semibold">{file.name}</span>
              </div>
            )}
          </div>

          {/* Configuration Section */}
          <div className="p-8 border-b border-gray-200 bg-gradient-to-r from-cyan-50/30 to-teal-50/30">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {/* Target Column */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Target Column (optional)</label>
                <input
                  type="text"
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  placeholder="e.g., price, label, target"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all"
                />
              </div>

              {/* Missing Value Strategy */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Handle Missing Values</label>
                <select
                  value={handleMissing}
                  onChange={(e) => setHandleMissing(e.target.value as any)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all"
                >
                  <option value="mean">Fill with Mean/Mode (recommended)</option>
                  <option value="drop">Drop rows with missing values</option>
                  <option value="none">Keep as is</option>
                </select>
              </div>

              {/* Outlier Strategy */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Outlier Detection</label>
                <select
                  value={handleOutliers}
                  onChange={(e) => setHandleOutliers(e.target.value as any)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all"
                >
                  <option value="none">No detection</option>
                  <option value="zscore">Z-score method (remove outliers)</option>
                  <option value="iqr">IQR method (remove outliers)</option>
                </select>
              </div>

              {/* Return as File */}
              <div className="flex items-end">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={returnAsFile}
                    onChange={(e) => setReturnAsFile(e.target.checked)}
                    className="w-4 h-4 text-cyan-600 rounded focus:ring-cyan-500"
                  />
                  <span className="text-sm font-medium text-gray-700">Download processed CSV</span>
                </label>
              </div>
            </div>

            {/* Suggest Features Button */}
            <button
              onClick={fetchFeatureSuggestions}
              disabled={!file || suggestionsLoading}
              className="w-full px-6 py-3 bg-gradient-to-r from-cyan-500 to-teal-500 text-white font-semibold rounded-lg hover:from-cyan-600 hover:to-teal-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
            >
              <TrendingUp className="h-5 w-5" />
              {suggestionsLoading ? "Analyzing features..." : "Suggest Smart Features"}
            </button>
          </div>

          {/* Feature Suggestions Section */}
          {showSuggestions && featureSuggestions.length > 0 && (
            <div className="p-8 border-b border-gray-200 bg-gradient-to-r from-teal-50/40 to-cyan-50/40">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                  <span className="text-teal-600">💡</span>
                  Recommended Features
                </h3>
                <button
                  onClick={selectAllFeatures}
                  className="text-sm font-medium text-cyan-600 hover:text-cyan-700 bg-cyan-50 px-3 py-1 rounded-full transition-colors"
                >
                  {selectedFeatures.size === featureSuggestions.length ? "Deselect All" : "Select All"}
                </button>
              </div>

              <div className="space-y-3">
                {featureSuggestions.map((feature, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-4 p-4 border border-gray-200 rounded-lg hover:border-cyan-400 hover:bg-cyan-50/30 transition-all cursor-pointer"
                    onClick={() => toggleFeatureSelection(feature.name)}
                  >
                    <input
                      type="checkbox"
                      checked={selectedFeatures.has(feature.name)}
                      onChange={() => toggleFeatureSelection(feature.name)}
                      className="w-5 h-5 text-cyan-600 rounded focus:ring-cyan-500 mt-1 cursor-pointer"
                    />
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold text-gray-800">{feature.name}</span>
                        <span className={`text-xs font-bold px-2 py-1 rounded-full ${getConfidenceColor(feature.confidence)}`}>
                          {getConfidenceBadge(feature.confidence)} ({Math.round(feature.confidence * 100)}%)
                        </span>
                        <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full font-medium">
                          {feature.type}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{feature.description}</p>
                      <div className="text-xs text-gray-500">
                        Based on: <span className="font-medium text-gray-700">{feature.columns_involved.join(", ")}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Processing Section */}
          <div className="p-8 flex gap-4">
            <button
              onClick={handleSubmit}
              disabled={loading || !file}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold rounded-lg hover:from-purple-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-xl"
            >
              {loading ? (
                <>
                  <span className="animate-spin">⚙️</span>
                  Processing...
                </>
              ) : (
                <>
                  <Download className="h-5 w-5" />
                  Run Feature Engineering
                </>
              )}
            </button>
          </div>
        </div>

        {/* Messages */}
        {message && (
          <div className="mb-6 p-4 bg-green-50 border border-green-300 rounded-lg text-green-700 font-medium flex items-center gap-2">
            <span>✓</span>
            {message}
          </div>
        )}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-300 rounded-lg text-red-700 font-medium flex items-center gap-2">
            <span>✕</span>
            Error: {error}
          </div>
        )}

        {/* Summary Panel */}
        {summary && (
          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
            <div className="bg-gradient-to-r from-purple-500 to-indigo-600 px-8 py-6">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                <TrendingUp className="h-6 w-6" />
                Feature Engineering Summary
              </h3>
            </div>

            <div className="p-8">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                {/* Rows Card */}
                <div className="bg-gradient-to-br from-blue-50 to-blue-100/50 border border-blue-200 rounded-lg p-6">
                  <div className="text-sm font-semibold text-blue-700 mb-2">Dataset Rows</div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-blue-900">{summary.rows_after ?? "—"}</span>
                    <span className="text-xs text-blue-600">
                      (was {summary.rows_before ?? "—"})
                    </span>
                  </div>
                  {summary.rows_removed && summary.rows_removed > 0 && (
                    <div className="text-xs text-red-600 mt-2 font-medium">
                      {summary.rows_removed} rows removed
                    </div>
                  )}
                </div>

                {/* Missing Values Card */}
                <div className="bg-gradient-to-br from-cyan-50 to-cyan-100/50 border border-cyan-200 rounded-lg p-6">
                  <div className="text-sm font-semibold text-cyan-700 mb-2">Missing Values</div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-cyan-900">{summary.missing_values_after ?? "—"}</span>
                    <span className="text-xs text-cyan-600">
                      (was {summary.missing_values_before ?? "—"})
                    </span>
                  </div>
                  {summary.missing_values_handled && summary.missing_values_handled > 0 && (
                    <div className="text-xs text-green-600 mt-2 font-medium">
                      {summary.missing_values_handled} handled
                    </div>
                  )}
                </div>

                {/* Columns Card */}
                <div className="bg-gradient-to-br from-purple-50 to-purple-100/50 border border-purple-200 rounded-lg p-6">
                  <div className="text-sm font-semibold text-purple-700 mb-2">Columns</div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-purple-900">{summary.columns_processed ?? "—"}</span>
                    <span className="text-xs text-purple-600">processed</span>
                  </div>
                </div>
              </div>

              {/* Generated Features */}
              {summary.generated_features && summary.generated_features.length > 0 && (
                <div className="mb-8">
                  <h4 className="font-bold text-gray-800 mb-3 flex items-center gap-2">
                    <Zap className="h-5 w-5 text-cyan-600" />
                    Generated Features
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {summary.generated_features.map((feat, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1 bg-gradient-to-r from-cyan-100 to-teal-100 text-cyan-900 text-sm font-medium rounded-full border border-cyan-300"
                      >
                        ✨ {feat}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Column Names */}
              {summary.column_names && summary.column_names.length > 0 && (
                <div className="mb-8">
                  <h4 className="font-bold text-gray-800 mb-3">Final Columns</h4>
                  <div className="bg-gray-50 rounded-lg p-4 max-h-40 overflow-y-auto">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                      {summary.column_names.map((col, idx) => (
                        <span key={idx} className="text-sm text-gray-700 bg-white px-2 py-1 rounded border border-gray-200">
                          {col}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Applied Methods */}
              {summary.applied_methods && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-bold text-gray-800 mb-3">Methods Applied</h4>
                  <ul className="space-y-2">
                    {Object.entries(summary.applied_methods).map(([k, v]) => (
                      <li key={k} className="text-sm text-gray-700">
                        <span className="font-semibold text-gray-900">{k}:</span>{" "}
                        <span className="text-gray-600">
                          {typeof v === "string" ? v : JSON.stringify(v)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}