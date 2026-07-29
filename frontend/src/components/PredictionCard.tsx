import type { PredictionResponse } from "../api/client";

interface PredictionCardProps {
  prediction: PredictionResponse;
}

export default function PredictionCard({ prediction }: PredictionCardProps) {
  const sortedEntries = Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <span className="text-lg font-semibold capitalize text-slate-100">{prediction.label}</span>
        <span className="font-mono-ui text-xs text-accent">
          {(prediction.confidence * 100).toFixed(1)}% · {prediction.model_name}
        </span>
      </div>

      <div className="space-y-2">
        {sortedEntries.map(([className, prob]) => (
          <div key={className}>
            <div className="font-mono-ui mb-1 flex justify-between text-xs text-slate-500">
              <span className="capitalize">{className}</span>
              <span>{(prob * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-accent transition-all"
                style={{ width: `${Math.round(prob * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
