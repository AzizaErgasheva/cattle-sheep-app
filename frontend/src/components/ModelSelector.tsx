import { useEffect, useState } from "react";
import { getModels, type ModelSummary } from "../api/client";

interface ModelSelectorProps {
  selectedModel: string | null;
  onSelect: (modelName: string) => void;
  disabled?: boolean;
}

export default function ModelSelector({ selectedModel, onSelect, disabled = false }: ModelSelectorProps) {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getModels()
      .then((data) => {
        if (cancelled) return;
        setModels(data.models);
        if (!selectedModel && data.default_model) onSelect(data.default_model);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (error) {
    return <p className="font-mono-ui text-xs text-red-400">Could not load models.</p>;
  }

  if (models.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {models.map((model) => {
        const isActive = model.name === selectedModel;
        return (
          <button
            key={model.name}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(model.name)}
            className={[
              "font-mono-ui flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition-colors",
              isActive
                ? "border-accent bg-accent/10 text-accent"
                : "border-slate-800 bg-slate-900 text-slate-400 hover:border-slate-700 hover:text-slate-200",
              disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer",
            ].join(" ")}
          >
            <span>{model.display_name}</span>
            {model.accuracy != null && (
              <span className={isActive ? "text-accent/80" : "text-slate-500"}>
                {(model.accuracy * 100).toFixed(1)}%
              </span>
            )}
            {model.is_best && <span className="text-[10px] text-amber-400">★</span>}
          </button>
        );
      })}
    </div>
  );
}
