import { useState } from "react";
import { api } from "../api";

interface Props {
  onCreated: () => void;
  onCancel: () => void;
}

export function AddTaskForm({ onCreated, onCancel }: Props) {
  const [dtype, setDtype] = useState("f16");
  const [m, setM] = useState("");
  const [n, setN] = useState("");
  const [k, setK] = useState("");
  const [mode, setMode] = useState("from_current_best");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    const mVal = parseInt(m);
    const nVal = parseInt(n);
    const kVal = parseInt(k);

    if (isNaN(mVal) || mVal < 128) return setError("M must be >= 128");
    if (isNaN(nVal) || nVal < 256) return setError("N must be >= 256");
    if (isNaN(kVal) || kVal < 128) return setError("K must be >= 128");

    setSubmitting(true);
    try {
      await api.createTask({ dtype, m: mVal, n: nVal, k: kVal, mode });
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create task");
    } finally {
      setSubmitting(false);
    }
  };

  const maxIter = mode === "from_current_best" ? 30 : 150;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <form
        onSubmit={handleSubmit}
        className="bg-gray-800 rounded-xl p-6 w-full max-w-md shadow-2xl border border-gray-700"
      >
        <h2 className="text-lg font-bold mb-4 text-gray-100">Add Kernel Tuning Task</h2>

        <div className="space-y-3">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Dtype</label>
            <select
              value={dtype}
              onChange={(e) => setDtype(e.target.value)}
              className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
            >
              <option value="f16">f16</option>
              <option value="e4m3">e4m3</option>
            </select>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "M", value: m, set: setM, min: 128 },
              { label: "N", value: n, set: setN, min: 256 },
              { label: "K", value: k, set: setK, min: 128 },
            ].map(({ label, value, set, min }) => (
              <div key={label}>
                <label className="block text-sm text-gray-400 mb-1">
                  {label} <span className="text-gray-500">({">"}= {min})</span>
                </label>
                <input
                  type="number"
                  value={value}
                  onChange={(e) => set(e.target.value)}
                  min={min}
                  className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
                  required
                />
              </div>
            ))}
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Mode</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
            >
              <option value="from_current_best">From Current Best (30 iter)</option>
              <option value="from_scratch">From Scratch (150 iter)</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">Max iterations: {maxIter}</p>
          </div>
        </div>

        {error && (
          <p className="mt-3 text-sm text-red-400 bg-red-900/30 rounded px-3 py-1.5">{error}</p>
        )}

        <div className="flex justify-end gap-3 mt-5">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 transition"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting}
            className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white font-medium transition disabled:opacity-50"
          >
            {submitting ? "Adding..." : "Add Task"}
          </button>
        </div>
      </form>
    </div>
  );
}
