import { useEffect, useState } from "react";
import { clearHistory, getHistory, type HistoryEntry } from "../api/client";

interface HistoryPanelProps {
  refreshKey: number;
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export default function HistoryPanel({ refreshKey }: HistoryPanelProps) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    getHistory(50)
      .then((data) => {
        if (!cancelled) setEntries(data);
      })
      .catch(() => {
        if (!cancelled) setError("Could not load history.");
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  async function handleClear() {
    try {
      await clearHistory();
      setEntries([]);
    } catch {
      setError("Could not clear history.");
    }
  }

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <h2 className="font-mono-ui text-sm text-slate-400">
          {entries.length} prediction{entries.length === 1 ? "" : "s"} logged
        </h2>
        {entries.length > 0 && (
          <button
            type="button"
            onClick={handleClear}
            className="font-mono-ui text-xs text-slate-500 hover:text-red-400"
          >
            clear history
          </button>
        )}
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}
      {isLoading && <p className="font-mono-ui text-xs text-slate-500">Loading…</p>}

      {!isLoading && entries.length === 0 && !error && (
        <p className="font-mono-ui text-xs text-slate-600">No predictions yet. Classify something first.</p>
      )}

      <ul className="space-y-2">
        {entries.map((entry) => (
          <li
            key={entry.id}
            className="flex items-center gap-3 rounded-lg border border-slate-800 bg-slate-900/60 p-3"
          >
            <img
              src={entry.thumbnail_data_url}
              alt={entry.label}
              className="h-12 w-12 flex-shrink-0 rounded-md object-cover"
            />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-medium capitalize text-slate-100">{entry.label}</span>
                <span className="font-mono-ui text-xs text-accent">{(entry.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="font-mono-ui mt-0.5 flex gap-2 text-[11px] text-slate-500">
                <span>{entry.model_name}</span>
                <span>·</span>
                <span>{formatTimestamp(entry.created_at)}</span>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
