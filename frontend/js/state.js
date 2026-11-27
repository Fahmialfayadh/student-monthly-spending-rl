export const appState = {
  schema: null,
  configsList: [],
  activeMeta: {
    scope: 'preset',
    name: 'surabaya_its',
    sha256: null,
    dirty: false,
  },
  worldConfig: null,
  trainingProfile: null,
  selectedGraphItem: null, // { type: 'node' | 'edge', id: string }
  previewedPath: null,
  simSession: null,
  simLogs: [],
  runs: [],
};

const listeners = new Set();

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function notify(reason = 'update') {
  for (const fn of listeners) {
    fn(appState, reason);
  }
}

export function markDirty() {
  appState.activeMeta.dirty = true;
  notify('dirty');
}

export function showToast(message, durationMs = 3200) {
  const el = document.getElementById('toast-banner');
  if (!el) return;
  el.textContent = message;
  el.classList.add('is-visible');
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => el.classList.remove('is-visible'), durationMs);
}
