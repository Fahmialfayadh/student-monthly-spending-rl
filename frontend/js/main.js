import { api } from './api.js';
import { initEnvConfigPage } from './pages/env_config.js';
import { initGraphStudioPage } from './pages/graph_studio.js';
import { initSimulationPage } from './pages/simulation.js';
import { initTrainingPage } from './pages/training.js';
import { appState, showToast, subscribe } from './state.js';
import { initGlobalTooltips } from './tooltip.js';

const PARTIALS = [
  ['graph', '/partials/graph_studio.html', initGraphStudioPage],
  ['config', '/partials/env_config.html', initEnvConfigPage],
  ['simulation', '/partials/simulation.html', initSimulationPage],
  ['training', '/partials/training.html', initTrainingPage],
];

const controllers = {};

function setStatus(stateClass, text) {
  const dot = document.getElementById('global-status-dot');
  const lbl = document.getElementById('global-status-text');
  dot.className = `status-dot ${stateClass}`;
  lbl.textContent = text;
}

function formatConfigOptionLabel(c) {
  if (c.scope === 'preset' && c.name === 'standard_student') return 'Template: Standard Student';
  if (c.scope === 'preset' && c.name === 'minimal_dorm') return 'Template: Minimal Dorm (2 Nodes)';
  if (c.scope === 'preset' && c.name === 'surabaya_its') return 'Example: Author Personal (Surabaya)';
  if (c.scope === 'active') return 'Active: env_config.yaml';
  return `Custom: ${c.name}`;
}

async function populateConfigSelector() {
  const select = document.getElementById('global-config-select');
  const res = await api.listConfigs();
  appState.configsList = res.configs || [];
  select.innerHTML = appState.configsList.map((c) => (
    `<option value="${c.scope}::${c.name}">${formatConfigOptionLabel(c)}</option>`
  )).join('');
  const currentKey = `${appState.activeMeta.scope}::${appState.activeMeta.name}`;
  if (appState.configsList.some((c) => `${c.scope}::${c.name}` === currentKey)) {
    select.value = currentKey;
  }
}

async function loadSelectedConfig(scope, name) {
  const res = await api.getConfig(scope, name);
  appState.worldConfig = res.config;
  appState.activeMeta = { scope: res.scope, name: res.name, sha256: res.sha256, dirty: false };
  appState.selectedGraphItem = null;
  appState.previewedPath = null;
  appState.simSession = null;
  setStatus('is-ok', `${res.scope}/${res.name}`);
  refreshActiveTab();
}

function activeTabName() {
  const h = (location.hash || '#graph').replace('#', '');
  return PARTIALS.some(([k]) => k === h) ? h : 'graph';
}

function switchTab(tabName) {
  document.querySelectorAll('.tab-button').forEach((btn) => {
    const active = btn.dataset.tab === tabName;
    btn.classList.toggle('is-active', active);
    btn.setAttribute('aria-selected', String(active));
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.hidden = panel.id !== `panel-${tabName}`;
  });
  refreshActiveTab();
}

function refreshActiveTab() {
  const tab = activeTabName();
  if (controllers[tab]?.refresh) controllers[tab].refresh();
}

async function bootstrap() {
  initGlobalTooltips();

  const main = document.getElementById('workspace-main');
  const htmlList = await Promise.all(PARTIALS.map(([, path]) => fetch(path).then((r) => r.text())));
  main.innerHTML = htmlList.join('\n');

  PARTIALS.forEach(([key, , initFn]) => {
    controllers[key] = initFn(main.querySelector(`#panel-${key}`));
  });

  const [schema, runsRes] = await Promise.all([api.getSchema(), api.listRuns()]);
  appState.schema = schema;
  appState.trainingProfile = structuredClone(schema.default_training_profile);
  appState.runs = runsRes.runs || [];

  await populateConfigSelector();
  // Default to universal neutral starter template so new users never see author-specific names first
  const initial = appState.configsList.find((c) => c.scope === 'preset' && c.name === 'standard_student')
    || appState.configsList.find((c) => c.scope === 'active')
    || appState.configsList[0];
  if (initial) {
    await loadSelectedConfig(initial.scope, initial.name);
    document.getElementById('global-config-select').value = `${initial.scope}::${initial.name}`;
  }

  document.getElementById('global-config-select').addEventListener('change', (e) => {
    const [scope, name] = e.target.value.split('::');
    loadSelectedConfig(scope, name).catch((err) => showToast(`Load error: ${err.message}`));
  });

  document.getElementById('btn-global-validate').addEventListener('click', async () => {
    try {
      const res = await api.validateConfig(appState.worldConfig);
      setStatus('is-ok', `Valid (${res.observation_dim}D obs)`);
      showToast(`Config valid: ${res.graph_summary.nodes.length} lokasi, ${res.graph_summary.edges.length} rute.`);
    } catch (err) {
      setStatus('is-err', 'Invalid config');
      showToast(`Validation error: ${err.message}`, 4500);
    }
  });

  document.getElementById('btn-global-save').addEventListener('click', () => {
    location.hash = '#config';
    switchTab('config');
  });

  subscribe((state, reason) => {
    if (reason === 'dirty') {
      setStatus('is-warn', `${state.activeMeta.name} (unsaved)`);
    } else if (reason === 'configs_changed') {
      populateConfigSelector().then(() => {
        setStatus('is-ok', `${state.activeMeta.scope}/${state.activeMeta.name}`);
      });
    }
  });

  window.addEventListener('hashchange', () => switchTab(activeTabName()));
  switchTab(activeTabName());
}

bootstrap().catch((err) => {
  setStatus('is-err', 'API Offline');
  showToast(`Startup failed: ${err.message}`, 5000);
});
