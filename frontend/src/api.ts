const BASE = "/api";

export interface TaskData {
  id: number;
  shape_key: string;
  dtype: string;
  m: number;
  n: number;
  k: number;
  mode: string;
  max_iterations: number;
  status: string;
  current_iteration: number;
  best_tflops: number | null;
  baseline_tflops: number | null;
  best_kernel: string | null;
  error_message: string | null;
  created_at: string | null;
  updated_at: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface IterationLogData {
  id: number;
  task_id: number;
  iteration: number;
  kernel_path: string | null;
  tflops: number | null;
  decision: string | null;
  bottleneck: string | null;
  idea_summary: string | null;
  logged_at: string | null;
}

export interface AgentLogData {
  id: number;
  task_id: number;
  level: string;
  message: string;
  timestamp: string | null;
}

export interface HealthData {
  status: string;
  scheduler_running: boolean;
  active_task_id: number | null;
  gpu_info: string | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  if (res.status === 204) return undefined as unknown as T;
  return res.json();
}

export const api = {
  listTasks: (status?: string) =>
    request<TaskData[]>(`/tasks${status ? `?status=${status}` : ""}`),

  createTask: (data: { dtype: string; m: number; n: number; k: number; mode: string }) =>
    request<TaskData>("/tasks", { method: "POST", body: JSON.stringify(data) }),

  getTask: (id: number) => request<TaskData>(`/tasks/${id}`),

  cancelTask: (id: number) =>
    request<TaskData>(`/tasks/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ status: "cancelled" }),
    }),

  promoteTask: (id: number) =>
    request<TaskData>(`/tasks/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ status: "pending" }),
    }),

  deleteTask: (id: number) =>
    request<void>(`/tasks/${id}`, { method: "DELETE" }),

  getIterationLogs: (id: number) =>
    request<IterationLogData[]>(`/tasks/${id}/logs`),

  getAgentLogs: (id: number, limit = 100) =>
    request<AgentLogData[]>(`/tasks/${id}/agent-logs?limit=${limit}`),

  getHealth: () => request<HealthData>("/health"),
};
