import { useState } from "react";
import UploadDropzone from "./components/UploadDropzone";
import PredictionCard from "./components/PredictionCard";
import GradCamOverlay from "./components/GradCamOverlay";
import ModelSelector from "./components/ModelSelector";
import HistoryPanel from "./components/HistoryPanel";
import Dashboard from "./components/Dashboard";
import { ApiError, explainImage, predictImage, type PredictionResponse } from "./api/client";

type Tab = "classify" | "history" | "dashboard";
type Status = "idle" | "predicting" | "done" | "error";

const TABS: { id: Tab; label: string }[] = [
  { id: "classify", label: "Classify" },
  { id: "history", label: "History" },
  { id: "dashboard", label: "Dashboard" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("classify");
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [isExplaining, setIsExplaining] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);

  async function handleFileSelected(file: File) {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPrediction(null);
    setOverlayUrl(null);
    setErrorMessage(null);
    setStatus("predicting");

    try {
      const result = await predictImage(file, selectedModel ?? undefined);
      setPrediction(result);
      setStatus("done");
      setHistoryRefreshKey((k) => k + 1);
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
    }
  }

  async function handleRequestExplanation() {
    if (!selectedFile) return;
    setIsExplaining(true);
    try {
      const result = await explainImage(selectedFile, selectedModel ?? undefined);
      setOverlayUrl(result.overlayUrl);
    } catch (err) {
      setErrorMessage(err instanceof ApiError ? err.message : "Could not generate an explanation.");
    } finally {
      setIsExplaining(false);
    }
  }

  return (
    <div className="mx-auto min-h-screen max-w-3xl px-4 py-10">
      <header className="mb-8">
        <div className="mb-1 flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-accent shadow-[0_0_8px_theme(colors.accent.DEFAULT)]" />
          <h1 className="font-mono-ui text-lg font-semibold text-slate-100">cow_vs_sheep_classifier</h1>
        </div>
        <p className="text-sm text-slate-500">
          Upload a photo, pick a model, see the prediction and the reasoning behind it.
        </p>
      </header>

      <nav className="mb-6 flex gap-1 border-b border-slate-800">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={[
              "font-mono-ui -mb-px border-b-2 px-4 py-2 text-xs transition-colors",
              activeTab === tab.id
                ? "border-accent text-accent"
                : "border-transparent text-slate-500 hover:text-slate-300",
            ].join(" ")}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {activeTab === "classify" && (
        <div className="space-y-4">
          <ModelSelector
            selectedModel={selectedModel}
            onSelect={setSelectedModel}
            disabled={status === "predicting"}
          />

          <UploadDropzone onFileSelected={handleFileSelected} disabled={status === "predicting"} />

          {errorMessage && (
            <p role="alert" className="rounded-lg border border-red-900 bg-red-950/40 px-4 py-2 text-sm text-red-400">
              {errorMessage}
            </p>
          )}

          {status === "predicting" && <p className="font-mono-ui text-xs text-slate-500">classifying…</p>}

          {prediction && previewUrl && (
            <div className="space-y-4">
              <img
                src={previewUrl}
                alt="Uploaded animal"
                className="max-h-64 w-full rounded-xl border border-slate-800 object-cover"
              />
              <PredictionCard prediction={prediction} />
              <GradCamOverlay
                originalUrl={previewUrl}
                overlayUrl={overlayUrl}
                isLoading={isExplaining}
                onRequestExplanation={handleRequestExplanation}
              />
            </div>
          )}
        </div>
      )}

      {activeTab === "history" && <HistoryPanel refreshKey={historyRefreshKey} />}

      {activeTab === "dashboard" && <Dashboard />}
    </div>
  );
}
