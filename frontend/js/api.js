async function request(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    const detail = data && data.detail
      ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
      : `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return data;
}

export const api = {
  getSchema: () => request('/schema'),
  listConfigs: () => request('/configs'),
  getConfig: (scope, name) => request(`/configs/${encodeURIComponent(scope)}/${encodeURIComponent(name)}`),
  saveConfig: (payload) => request('/configs/save', { method: 'POST', body: JSON.stringify(payload) }),
  validateConfig: (config) => request('/config/validate', { method: 'POST', body: JSON.stringify(config) }),
  previewRoute: (payload) => request('/graph/preview-route', { method: 'POST', body: JSON.stringify(payload) }),

  createSimSession: (payload) => request('/sim/sessions', { method: 'POST', body: JSON.stringify(payload) }),
  getSimSession: (sid) => request(`/sim/sessions/${encodeURIComponent(sid)}`),
  resetSimSession: (sid, seed = null) => request(`/sim/sessions/${encodeURIComponent(sid)}/reset`, {
    method: 'POST',
    body: JSON.stringify({ seed }),
  }),
  stepSimSession: (sid, action) => request(`/sim/sessions/${encodeURIComponent(sid)}/step`, {
    method: 'POST',
    body: JSON.stringify({ action, validate_mask: true }),
  }),
  autoStepSimSession: (sid, payload) => request(`/sim/sessions/${encodeURIComponent(sid)}/auto-step`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }),

  validateTraining: (payload) => request('/training/validate', { method: 'POST', body: JSON.stringify(payload) }),
  startTraining: (payload) => request('/training/start', { method: 'POST', body: JSON.stringify(payload) }),
  listTrainingJobs: () => request('/training/jobs'),
  listRuns: () => request('/runs'),
  getRunDetail: (runId) => request(`/runs/${encodeURIComponent(runId)}`),
};
