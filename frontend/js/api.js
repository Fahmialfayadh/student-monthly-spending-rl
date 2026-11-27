function formatApiError(detail, status) {
  if (!detail) return `HTTP ${status}`;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .slice(0, 4)
      .map((item) => {
        const loc = Array.isArray(item.loc) ? item.loc.filter((p) => p !== 'body').join('.') : '';
        const msg = item.msg || JSON.stringify(item);
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join(' | ');
  }
  return JSON.stringify(detail);
}

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
    throw new Error(formatApiError(data?.detail, res.status));
  }
  return data;
}

export const api = {
  getSchema: () => request('/schema'),
  listConfigs: () => request('/configs'),
  getConfig: (scope, name) => request(`/configs/${encodeURIComponent(scope)}/${encodeURIComponent(name)}`),
  saveConfig: (payload) => request('/configs/save', { method: 'POST', body: JSON.stringify(payload) }),
  renderYaml: (config) => request('/configs/render-yaml', { method: 'POST', body: JSON.stringify(config) }),
  importYaml: (payload) => request('/configs/import-yaml', { method: 'POST', body: JSON.stringify(payload) }),
  deleteCustomConfig: (name) => request(`/configs/custom/${encodeURIComponent(name)}`, { method: 'DELETE' }),
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
  deleteSimSession: (sid) => request(`/sim/sessions/${encodeURIComponent(sid)}`, { method: 'DELETE' }),

  validateTraining: (payload) => request('/training/validate', { method: 'POST', body: JSON.stringify(payload) }),
  startTraining: (payload) => request('/training/start', { method: 'POST', body: JSON.stringify(payload) }),
  stopTrainingJob: (jobId) => request(`/training/jobs/${encodeURIComponent(jobId)}/stop`, { method: 'POST' }),
  listTrainingJobs: () => request('/training/jobs'),
  listRuns: () => request('/runs'),
  getRunDetail: (runId) => request(`/runs/${encodeURIComponent(runId)}`),
  getRunTrace: (runId) => request(`/runs/${encodeURIComponent(runId)}/trace?limit=300`),
  deleteRun: (runId) => request(`/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' }),
};
