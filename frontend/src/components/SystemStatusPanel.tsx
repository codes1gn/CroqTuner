import { useState } from "react";
import { api, type HealthData } from "../api";

interface Props {
  health: HealthData | null;
  onRefresh: () => Promise<void>;
}

const MODEL_LABELS: Record<string, string> = {
  "opencode/qwen3.6-plus-free": "Qwen3.6 Plus Free",
  "opencode/minimax-m2.5-free": "Minimax M2.5 Free",
  "opencode/big-pickle": "Big Pickle Free",
};

export function SystemStatusPanel({ health, onRefresh }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  if (!health) {
    return (
      <section className="rounded-2xl border border-gray-800 bg-gray-900/80 p-4 text-sm text-gray-500">
        Loading system status...
      </section>
    );
  }

  const queueItems = [
    { label: "Waiting", value: health.task_counts.waiting ?? 0 },
    { label: "Pending", value: health.task_counts.pending ?? 0 },
    { label: "Running", value: health.task_counts.running ?? 0 },
    { label: "Failed", value: health.task_counts.failed ?? 0 },
  ];

  const handleSelectModel = async (model: string) => {
    setSaving(true);
    setError("");
    try {
      await api.setDefaultModel(model);
      await onRefresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update model");
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="rounded-2xl border border-gray-800 bg-gradient-to-br from-gray-900 via-gray-900 to-slate-950 p-4 shadow-[0_20px_80px_rgba(0,0,0,0.35)]">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.3em] text-cyan-500">System</div>
          <div className="mt-1 flex items-center gap-3">
            <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${health.scheduler_running ? "bg-emerald-500/20 text-emerald-300" : "bg-red-500/20 text-red-300"}`}>
              {health.scheduler_running ? "Scheduler running" : "Scheduler stopped"}
            </span>
            <span className="text-sm text-gray-400">
              Active task: {health.active_task_id ?? "none"}
            </span>
          </div>
          <div className="mt-3 text-sm text-gray-300">
            Default model: <span className="font-medium text-white">{MODEL_LABELS[health.default_model] ?? health.default_model}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {queueItems.map((item) => (
            <div key={item.label} className="rounded-xl border border-gray-800 bg-black/20 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.25em] text-gray-500">{item.label}</div>
              <div className="mt-1 text-2xl font-semibold text-gray-100">{item.value}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="rounded-xl border border-gray-800 bg-black/20 px-4 py-3 text-sm text-gray-300">
          <div className="text-[11px] uppercase tracking-[0.25em] text-gray-500">GPU</div>
          <pre className="mt-2 whitespace-pre-wrap font-mono text-xs text-gray-300">{health.gpu_info ?? "Unavailable"}</pre>
        </div>

        <div className="lg:w-[26rem]">
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="flex w-full items-center justify-between rounded-xl border border-cyan-800/60 bg-cyan-950/20 px-4 py-3 text-left transition hover:border-cyan-700 hover:bg-cyan-950/30"
          >
            <div>
              <div className="text-[11px] uppercase tracking-[0.25em] text-cyan-400">Model control</div>
              <div className="mt-1 text-sm text-cyan-50">Choose the default OpenCode Zen model for new and fallback task runs</div>
            </div>
            <span className="text-cyan-300">{expanded ? "Hide" : "Show"}</span>
          </button>

          {expanded && (
            <div className="mt-2 space-y-2 rounded-xl border border-gray-800 bg-black/20 p-3">
              {health.available_models.map((model) => {
                const active = model === health.default_model;
                return (
                  <button
                    key={model}
                    type="button"
                    disabled={saving}
                    onClick={() => handleSelectModel(model)}
                    className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-sm transition ${active ? "border-cyan-500 bg-cyan-500/10 text-cyan-100" : "border-gray-700 bg-gray-900/70 text-gray-300 hover:border-gray-500"}`}
                  >
                    <span>{MODEL_LABELS[model] ?? model}</span>
                    <span className="font-mono text-xs text-gray-400">{model}</span>
                  </button>
                );
              })}
              {error && <p className="rounded-lg bg-red-950/40 px-3 py-2 text-sm text-red-300">{error}</p>}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}