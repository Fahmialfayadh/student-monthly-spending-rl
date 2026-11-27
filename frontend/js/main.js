import { api } from './api.js';
import { initEnvConfigPage } from './pages/env_config.js';
import { initGraphStudioPage } from './pages/graph_studio.js';
import { initSimulationPage } from './pages/simulation.js';
import { initTrainingPage } from './pages/training.js';
import { appState, notify, showToast, subscribe } from './state.js';
import { hideTooltip, initGlobalTooltips } from './tooltip.js';
import { escapeHtml, formatIdr } from './utils.js';

const PARTIALS = [
  ['config', '/partials/env_config.html', initEnvConfigPage],
  ['graph', '/partials/graph_studio.html', initGraphStudioPage],
  ['training', '/partials/training.html', initTrainingPage],
  ['simulation', '/partials/simulation.html', initSimulationPage],
];

const controllers = {};
let currentTab = 'config';

function setStatus(stateClass, text) {
  const dot = document.getElementById('global-status-dot');
  const lbl = document.getElementById('global-status-text');
  dot.className = `status-dot ${stateClass}`;
  lbl.textContent = text;
}

function syncUnsavedBar() {
  const bar = document.getElementById('unsaved-bar');
  const nameEl = document.getElementById('unsaved-bar-name');
  if (!bar) return;
  const dirty = Boolean(appState.activeMeta?.dirty);
  bar.hidden = !dirty;
  if (nameEl) {
    nameEl.textContent = `${appState.activeMeta.scope}/${appState.activeMeta.name}`;
  }
}

function formatConfigOptionLabel(c) {
  if (c.scope === 'preset' && c.name === 'standard_student') return 'Template: Standard Student';
  if (c.scope === 'preset' && c.name === 'minimal_dorm') return 'Template: Minimal Dorm (2 Lokasi)';
  if (c.scope === 'preset' && c.name === 'surabaya_its') return 'Template: Multi-Hub Campus (4 Lokasi)';
  if (c.scope === 'active') return 'Aktif: env_config.yaml';
  return `Custom: ${c.name}`;
}

async function populateConfigSelector() {
  const select = document.getElementById('global-config-select');
  const res = await api.listConfigs();
  appState.configsList = res.configs || [];
  select.innerHTML = appState.configsList.map((c) => (
    `<option value="${escapeHtml(c.scope)}::${escapeHtml(c.name)}">${escapeHtml(formatConfigOptionLabel(c))}</option>`
  )).join('');
  const currentKey = `${appState.activeMeta.scope}::${appState.activeMeta.name}`;
  if (appState.configsList.some((c) => `${c.scope}::${c.name}` === currentKey)) {
    select.value = currentKey;
  }
}

async function loadSelectedConfig(scope, name) {
  if (controllers.simulation?.onLeave) controllers.simulation.onLeave();
  const res = await api.getConfig(scope, name);
  appState.worldConfig = res.config;
  appState.activeMeta = {
    scope: res.scope,
    name: res.name,
    sha256: res.sha256,
    canonicalSha256: res.canonical_sha256 || res.sha256,
    dirty: false,
  };
  appState.selectedGraphItem = null;
  appState.previewedPath = null;
  appState.simSession = null;
  appState.simStale = false;
  appState.simLogs = [];
  setStatus('is-ok', `${res.scope}/${res.name}`);
  syncUnsavedBar();
  notify('config_loaded');
  refreshActiveTab();
}

function activeTabName() {
  const h = (location.hash || '#config').replace('#', '');
  return PARTIALS.some(([k]) => k === h) ? h : 'config';
}

function switchTab(tabName) {
  if (currentTab !== tabName && controllers[currentTab]?.onLeave) {
    controllers[currentTab].onLeave();
  }
  hideTooltip();
  currentTab = tabName;
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

function initSaveDialog() {
  const dialog = document.getElementById('save-config-dialog');
  const nameEl = document.getElementById('dlg-save-name');
  const syncEl = document.getElementById('dlg-save-sync');
  const saveBtn = document.getElementById('dlg-btn-save');
  const valBtn = document.getElementById('dlg-btn-validate');
  const expBtn = document.getElementById('dlg-btn-export');
  const impInput = document.getElementById('dlg-input-import');
  const delBtn = document.getElementById('dlg-btn-delete');
  const closeBtn = document.getElementById('dlg-btn-close');
  const metaBox = document.getElementById('dlg-meta-box');

  function renderDialogMeta() {
    const m = appState.activeMeta;
    metaBox.innerHTML = `
      <div class="kv-row"><span class="kv-key">Config Aktif</span><span class="kv-val code-id">${escapeHtml(m.scope)}/${escapeHtml(m.name)}</span></div>
      <div class="kv-row"><span class="kv-key">Status</span><span class="kv-val">${m.dirty ? 'Belum disimpan' : 'Tersinkron dengan disk'}</span></div>
      <div class="kv-row"><span class="kv-key">SHA-256</span><span class="kv-val code-id">${escapeHtml((m.sha256 || '-').slice(0, 16))}</span></div>
    `;
    delBtn.hidden = m.scope !== 'custom';
  }

  function openDialog() {
    const m = appState.activeMeta;
    nameEl.value = m.scope === 'custom' ? m.name : `${m.name}_custom`;
    renderDialogMeta();
    if (!dialog.open) dialog.showModal();
    if (m.scope !== 'custom') {
      nameEl.focus();
      nameEl.select();
    } else {
      saveBtn.focus();
    }
  }

  closeBtn.addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (ev) => {
    if (ev.target === dialog) dialog.close();
  });

  valBtn.addEventListener('click', async () => {
    try {
      const res = await api.validateConfig(appState.worldConfig);
      const fin = appState.worldConfig.finance || {};
      const fixedTotal = Object.values(fin.fixed_expenses || {}).reduce((a, b) => a + Number(b || 0), 0);
      const net = Math.max(0, Number(fin.monthly_budget || 0) - fixedTotal);
      setStatus('is-ok', 'Konfigurasi valid');
      showToast(
        `Valid: ${res.graph_summary.nodes.length} lokasi, ${res.graph_summary.edges.length} rute, saldo operasional ${formatIdr(net)}.`,
      );
    } catch (err) {
      setStatus('is-err', 'Perlu diperbaiki');
      showToast(`Konfigurasi belum valid: ${err.message}`, 5000);
    }
  });

  saveBtn.addEventListener('click', async () => {
    const name = nameEl.value.trim();
    if (!name) {
      showToast('Masukkan nama config lokal (mis. kos_bandung_v1).');
      nameEl.focus();
      return;
    }
    try {
      const res = await api.saveConfig({
        name,
        config: appState.worldConfig,
        overwrite: true,
        expected_sha256: appState.activeMeta.scope === 'custom' && appState.activeMeta.name === name
          ? appState.activeMeta.sha256
          : null,
        set_active_env_config: syncEl.checked,
      });
      appState.activeMeta = {
        scope: 'custom',
        name: res.name,
        sha256: res.sha256,
        canonicalSha256: res.canonical_sha256 || res.sha256,
        dirty: false,
      };
      syncUnsavedBar();
      showToast(`Tersimpan ke ${res.path}`);
      dialog.close();
      notify('configs_changed');
    } catch (err) {
      showToast(`Gagal simpan: ${err.message}`, 5000);
    }
  });

  expBtn.addEventListener('click', async () => {
    try {
      const name = nameEl.value.trim() || appState.activeMeta.name || 'world_config';
      const res = await api.renderYaml(appState.worldConfig);
      const blob = new Blob([res.yaml_text], { type: 'text/yaml' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${name}.yaml`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (err) {
      showToast(`Gagal unduh YAML: ${err.message}`, 5000);
    }
  });

  impInput.addEventListener('change', async () => {
    const file = impInput.files?.[0];
    if (!file) return;
    const baseName = file.name.replace(/\.(ya?ml)$/i, '').replace(/[^a-zA-Z0-9_-]/g, '_') || 'imported_cfg';
    try {
      const yamlContent = await file.text();
      const res = await api.importYaml({
        name: baseName,
        yaml_content: yamlContent,
        overwrite: true,
        set_active_env_config: false,
      });
      const loaded = await api.getConfig('custom', res.name);
      appState.worldConfig = loaded.config;
      appState.activeMeta = {
        scope: 'custom',
        name: loaded.name,
        sha256: loaded.sha256,
        canonicalSha256: loaded.canonical_sha256 || loaded.sha256,
        dirty: false,
      };
      syncUnsavedBar();
      showToast(`Berhasil impor ke ${res.path}`);
      dialog.close();
      notify('configs_changed');
      refreshActiveTab();
    } catch (err) {
      showToast(`Gagal impor YAML: ${err.message}`, 5000);
    } finally {
      impInput.value = '';
    }
  });

  delBtn.addEventListener('click', async () => {
    if (appState.activeMeta.scope !== 'custom') return;
    if (!window.confirm(`Hapus custom config '${appState.activeMeta.name}' dari disk?`)) return;
    try {
      await api.deleteCustomConfig(appState.activeMeta.name);
      showToast(`Config '${appState.activeMeta.name}' dihapus.`);
      dialog.close();
      notify('config_deleted');
    } catch (err) {
      showToast(`Gagal hapus: ${err.message}`);
    }
  });

  return { openDialog };
}

async function bootstrap() {
  initGlobalTooltips();
  const saveModal = initSaveDialog();

  const main = document.getElementById('workspace-main');
  const htmlList = await Promise.all(PARTIALS.map(([, path]) => fetch(path).then((r) => r.text())));
  main.innerHTML = htmlList.join('\n');

  PARTIALS.forEach(([key, , initFn]) => {
    controllers[key] = initFn(main.querySelector(`#panel-${key}`), {
      openSaveDialog: saveModal.openDialog,
    });
  });

  const [schema, runsRes] = await Promise.all([api.getSchema(), api.listRuns()]);
  appState.schema = schema;
  appState.trainingProfile = structuredClone(schema.default_training_profile);
  appState.runs = runsRes.runs || [];

  await populateConfigSelector();
  const initial = appState.configsList.find((c) => c.scope === 'preset' && c.name === 'standard_student')
    || appState.configsList.find((c) => c.scope === 'active')
    || appState.configsList[0];
  if (initial) {
    await loadSelectedConfig(initial.scope, initial.name);
    document.getElementById('global-config-select').value = `${initial.scope}::${initial.name}`;
  }

  const configSelect = document.getElementById('global-config-select');
  configSelect.addEventListener('change', (e) => {
    const prevKey = `${appState.activeMeta.scope}::${appState.activeMeta.name}`;
    if (appState.activeMeta.dirty) {
      const ok = window.confirm('Ada perubahan yang belum disimpan. Yakin ingin pindah config dan membuang perubahan?');
      if (!ok) {
        configSelect.value = prevKey;
        return;
      }
    }
    const [scope, name] = e.target.value.split('::');
    loadSelectedConfig(scope, name).catch((err) => showToast(`Gagal memuat: ${err.message}`));
  });

  document.getElementById('btn-global-save').addEventListener('click', saveModal.openDialog);
  document.getElementById('btn-unsaved-save').addEventListener('click', saveModal.openDialog);
  document.getElementById('btn-unsaved-discard').addEventListener('click', () => {
    if (!window.confirm('Batalkan seluruh perubahan yang belum disimpan dan muat ulang dari disk?')) return;
    loadSelectedConfig(appState.activeMeta.scope, appState.activeMeta.name)
      .then(() => showToast('Perubahan dibatalkan; konfigurasi dimuat ulang dari disk.'))
      .catch((err) => showToast(`Gagal muat ulang: ${err.message}`));
  });

  window.addEventListener('beforeunload', (ev) => {
    if (appState.activeMeta?.dirty) {
      ev.preventDefault();
      ev.returnValue = '';
    }
  });

  subscribe(async (state, reason) => {
    if (reason === 'dirty') {
      setStatus('is-warn', `${state.activeMeta.name} · belum disimpan`);
      syncUnsavedBar();
    } else if (reason === 'configs_changed') {
      await populateConfigSelector();
      setStatus('is-ok', `${state.activeMeta.scope}/${state.activeMeta.name}`);
      syncUnsavedBar();
    } else if (reason === 'config_deleted') {
      await populateConfigSelector();
      const fallback = appState.configsList.find((c) => c.scope === 'preset' && c.name === 'standard_student')
        || appState.configsList[0];
      if (fallback) {
        await loadSelectedConfig(fallback.scope, fallback.name);
        configSelect.value = `${fallback.scope}::${fallback.name}`;
      }
    }
  });

  window.addEventListener('hashchange', () => switchTab(activeTabName()));
  switchTab(activeTabName());
}

bootstrap().catch((err) => {
  setStatus('is-err', 'API Offline');
  showToast(`Gagal inisialisasi: ${err.message}`, 5000);
});
