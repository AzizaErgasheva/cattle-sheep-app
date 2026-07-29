interface GradCamOverlayProps {
  originalUrl: string;
  overlayUrl: string | null;
  isLoading: boolean;
  onRequestExplanation: () => void;
}

export default function GradCamOverlay({
  originalUrl,
  overlayUrl,
  isLoading,
  onRequestExplanation,
}: GradCamOverlayProps) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <div className="mb-3 flex items-center justify-between">
        <span className="font-mono-ui text-xs text-slate-400">why this prediction?</span>
        {!overlayUrl && (
          <button
            type="button"
            onClick={onRequestExplanation}
            disabled={isLoading}
            className="font-mono-ui rounded-lg border border-accent/40 bg-accent/10 px-3 py-1.5 text-xs text-accent transition-colors hover:bg-accent/20 disabled:opacity-50"
          >
            {isLoading ? "generating…" : "show grad-cam"}
          </button>
        )}
      </div>

      {overlayUrl && (
        <div className="grid grid-cols-2 gap-3">
          <figure>
            <img src={originalUrl} alt="Original upload" className="w-full rounded-lg border border-slate-800" />
            <figcaption className="font-mono-ui mt-1 text-center text-[11px] text-slate-500">original</figcaption>
          </figure>
          <figure>
            <img
              src={overlayUrl}
              alt="Grad-CAM activation overlay"
              className="w-full rounded-lg border border-slate-800"
            />
            <figcaption className="font-mono-ui mt-1 text-center text-[11px] text-slate-500">grad-cam</figcaption>
          </figure>
        </div>
      )}
    </div>
  );
}
