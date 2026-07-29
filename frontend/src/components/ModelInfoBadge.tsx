import { useEffect, useState } from "react";
import { getModelInfo, type ModelInfoResponse } from "../api/client";

export default function ModelInfoBadge() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getModelInfo()
      .then((data) => {
        if (!cancelled) setInfo(data);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error || !info) return null;

  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600">
      <span className="font-medium capitalize text-slate-800">{info.model_name}</span>
      {info.test_accuracy != null && <span>· {(info.test_accuracy * 100).toFixed(1)}% test accuracy</span>}
    </span>
  );
}
