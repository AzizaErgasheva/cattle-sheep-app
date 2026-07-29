import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getModels, type ModelSummary } from "../api/client";

const METRIC_COLORS: Record<string, string> = {
  accuracy: "#22d3ee",
  precision: "#a78bfa",
  recall: "#34d399",
  f1: "#f472b6",
};

export default function Dashboard() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getModels()
      .then((data) => {
        if (!cancelled) setModels(data.models);
      })
      .catch(() => {
        if (!cancelled) setError("Could not load model metrics.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return <p className="text-sm text-red-400">{error}</p>;
  if (models.length === 0) return <p className="font-mono-ui text-xs text-slate-500">Loading…</p>;

  const chartData = models.map((m) => ({
    name: m.display_name,
    accuracy: m.accuracy != null ? +(m.accuracy * 100).toFixed(2) : null,
    precision: m.precision != null ? +(m.precision * 100).toFixed(2) : null,
    recall: m.recall != null ? +(m.recall * 100).toFixed(2) : null,
    f1: m.f1 != null ? +(m.f1 * 100).toFixed(2) : null,
  }));

  const bestModel = models.find((m) => m.is_best);

  return (
    <div className="space-y-6">
      {bestModel && (
        <div className="rounded-lg border border-accent/30 bg-accent/5 px-4 py-3">
          <p className="font-mono-ui text-xs text-accent">
            best model: {bestModel.display_name}
            {bestModel.accuracy != null && ` — ${(bestModel.accuracy * 100).toFixed(2)}% test accuracy`}
          </p>
        </div>
      )}

      <div className="h-72 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} barGap={4}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
            <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} unit="%" />
            <Tooltip
              contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
              labelStyle={{ color: "#e2e8f0" }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {Object.entries(METRIC_COLORS).map(([key, color]) => (
              <Bar key={key} dataKey={key} fill={color} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="overflow-hidden rounded-xl border border-slate-800">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-900 font-mono-ui text-xs uppercase text-slate-500">
            <tr>
              <th className="px-4 py-2">Model</th>
              <th className="px-4 py-2">Accuracy</th>
              <th className="px-4 py-2">Precision</th>
              <th className="px-4 py-2">Recall</th>
              <th className="px-4 py-2">F1</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.name} className="border-t border-slate-800">
                <td className="px-4 py-2 text-slate-200">
                  {m.display_name} {m.is_best && <span className="text-amber-400">★</span>}
                </td>
                <td className="font-mono-ui px-4 py-2 text-slate-400">
                  {m.accuracy != null ? `${(m.accuracy * 100).toFixed(2)}%` : "—"}
                </td>
                <td className="font-mono-ui px-4 py-2 text-slate-400">
                  {m.precision != null ? `${(m.precision * 100).toFixed(2)}%` : "—"}
                </td>
                <td className="font-mono-ui px-4 py-2 text-slate-400">
                  {m.recall != null ? `${(m.recall * 100).toFixed(2)}%` : "—"}
                </td>
                <td className="font-mono-ui px-4 py-2 text-slate-400">
                  {m.f1 != null ? `${(m.f1 * 100).toFixed(2)}%` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
