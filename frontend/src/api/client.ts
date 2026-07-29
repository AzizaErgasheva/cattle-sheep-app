// Mirrors the backend's Pydantic response schemas exactly
// (app/api/schemas.py) so a shape mismatch is a compile-time error here.

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface PredictionResponse {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_name: string;
}

export interface ModelSummary {
  name: string;
  display_name: string;
  is_best: boolean;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
}

export interface ModelsListResponse {
  models: ModelSummary[];
  default_model: string;
}

export interface HistoryEntry {
  id: string;
  created_at: string;
  model_name: string;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  thumbnail_data_url: string;
}

export interface HistoryListResponse {
  entries: HistoryEntry[];
}

export interface ExplanationResult {
  overlayUrl: string;
  label: string;
  confidence: number;
  modelName: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function assertOk(res: Response, action: string): Promise<Response> {
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new ApiError(`${action} failed (${res.status}): ${detail}`, res.status);
  }
  return res;
}

export async function predictImage(file: File, modelName?: string): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (modelName) formData.append("model_name", modelName);

  const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
  await assertOk(res, "Prediction");
  return (await res.json()) as PredictionResponse;
}

export async function explainImage(file: File, modelName?: string): Promise<ExplanationResult> {
  const formData = new FormData();
  formData.append("file", file);
  if (modelName) formData.append("model_name", modelName);

  const res = await fetch(`${API_URL}/predict/explain`, { method: "POST", body: formData });
  await assertOk(res, "Explanation");

  const blob = await res.blob();
  return {
    overlayUrl: URL.createObjectURL(blob),
    label: res.headers.get("X-Predicted-Label") ?? "",
    confidence: parseFloat(res.headers.get("X-Confidence") ?? "0"),
    modelName: res.headers.get("X-Model-Name") ?? "",
  };
}

export async function getModels(): Promise<ModelsListResponse> {
  const res = await fetch(`${API_URL}/models`);
  await assertOk(res, "Model list");
  return (await res.json()) as ModelsListResponse;
}

export async function getHistory(limit = 20): Promise<HistoryEntry[]> {
  const res = await fetch(`${API_URL}/history?limit=${limit}`);
  await assertOk(res, "History fetch");
  const body = (await res.json()) as HistoryListResponse;
  return body.entries;
}

export async function clearHistory(): Promise<void> {
  const res = await fetch(`${API_URL}/history`, { method: "DELETE" });
  await assertOk(res, "History clear");
}
