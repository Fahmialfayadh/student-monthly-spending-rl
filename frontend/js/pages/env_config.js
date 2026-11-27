import { api } from '../api.js';
import { appState, markDirty, notify, showToast } from '../state.js';

const FIELD_HELP = {
  monthly_budget: 'Total uang saku / anggaran bulanan sebelum dipotong biaya tetap (fixed_expenses).',
  low_balance_threshold: 'Ambang saldo (IDR) ketika fitur peringatan saldo menipis aktif untuk agen.',
  start_day: 'Hari ke-berapa dalam bulan tekanan impulsivitas akhir bulan mulai muncul.',
  weight: 'Bobot kenaikan probabilitas jajan impulsif saat memasuki akhir bulan.',
  meal_prep_batch_size: 'Jumlah porsi maksimal yang dihasilkan dalam 1 kali sesi meal prep.',
  meal_prep_energy_cost: 'Energi yang dipakai untuk melakukan meal prep di dapur.',
  meal_prep_min_energy: 'Energi minimum agar agen diizinkan melakukan meal prep.',
  cook_fresh_energy_cost: 'Energi yang dipakai untuk memasak 1 porsi makanan segar.',
  min_hunger_to_eat: 'Ambang rasa lapar minimum agar aksi makan langsung diizinkan (mencegah makan saat kenyang).',
  starvation_threshold: 'Ambang rasa lapar tinggi yang mulai memicu penalti energi & mood setiap step.',
  room_study_max_stress_gate: 'Batas stres maksimal agar masih bisa belajar mandiri di kamar tanpa harus keluar.',
  min_attendance_rate_for_exam: 'Syarat rasio kehadiran kuliah minimal (mis. 0.80 = 80%) agar terhindar dari penalti ujian.',
  fatal_energy_threshold: 'Jika energi jatuh ke nilai ini atau lebih rendah, episode langsung berhenti (fatal).',
  fatal_hunger_consecutive_steps: 'Berapa step berturut-turut lapar maksimal sebelum episode berhenti fatal.',
  potential_scale: 'Skala reward potential-based shaping berbasis kecukupan saldo terhadap sisa hari.',
  gamma: 'Faktor diskon shaping; harus sama persis dengan Discount Gamma PPO jika shaping aktif.',
  survived_30_days_bonus: 'Bonus reward terminal bila agen berhasil bertahan penuh selama 1 bulan.',
  remaining_balance_reward_per_100k: 'Reward tambahan per Rp 100.000 sisa tabungan di akhir bulan.',
};

const SECTION_DEFS = [
  { key: 'finance', title: 'Finance & Impulsivity', objPath: ['finance'] },
  { key: 'pantry_cooking', title: 'Cooking Gates & Energy Costs', objPath: ['pantry_and_cooking', 'cooking_costs_and_gates'] },
  { key: 'biometrics_init', title: 'Biometrics Initial State', objPath: ['biometrics_and_psychology', 'initial_state'] },
  { key: 'biometrics_dyn', title: 'Biometrics Step Dynamics', objPath: ['biometrics_and_psychology', 'step_dynamics'] },
  { key: 'academics', title: 'Academics & Study Rules', objPath: ['academics'] },
  { key: 'rewards', title: 'Rewards & Potential Shaping', objPath: ['rewards'] },
  { key: 'termination', title: 'Termination Thresholds', objPath: ['termination'] },
];

function getAtPath(obj, path) {
  return path.reduce((acc, k) => (acc && typeof acc === 'object' ? acc[k] : undefined), obj);
}

function renderObjectFields(container, targetObj, prefix = '') {
  Object.entries(targetObj).forEach(([k, v]) => {
    if (k === 'weekly_schedule') return;
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      const sub = document.createElement('div');
      sub.className = 'stack-col';
      sub.style.gap = '6px';
      sub.style.paddingTop = '6px';
      const title = document.createElement('div');
      title.className = 'field-label mono';
      title.style.color = 'var(--ink-primary)';
      title.textContent = prefix ? `${prefix}.${k}` : k;
      sub.appendChild(title);
      renderObjectFields(sub, v, prefix ? `${prefix}.${k}` : k);
      container.appendChild(sub);
      return;
    }

    const row = document.createElement('div');
    row.className = 'field-row-2';
    row.style.alignItems = 'center';
    const label = document.createElement('label');
    label.className = 'field-label mono';
    label.textContent = k;
    if (FIELD_HELP[k]) {
      label.classList.add('tooltip-trigger');
      label.setAttribute('data-tooltip', FIELD_HELP[k]);
    }

    let input;
    if (typeof v === 'boolean') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = v;
      input.addEventListener('change', () => {
        targetObj[k] = input.checked;
        markDirty();
      });
    } else if (Array.isArray(v)) {
      input = document.createElement('input');
      input.className = 'control-input mono';
      input.value = v.join(', ');
      input.addEventListener('change', () => {
        targetObj[k] = input.value.split(',').map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n));
        markDirty();
      });
    } else {
      input = document.createElement('input');
      input.className = 'control-input mono';
      input.type = typeof v === 'number' ? 'number' : 'text';
      input.step = 'any';
      input.value = v ?? '';
      input.addEventListener('change', () => {
        targetObj[k] = typeof v === 'number' ? Number(input.value) : input.value;
        markDirty();
      });
    }

    row.appendChild(label);
    row.appendChild(input);
    container.appendChild(row);
  });
}

function renderExternalFoods(container, worldConfig) {
  const foods = worldConfig.external_foods || {};
  const entries = Object.entries(foods);

  if (entries.length === 0) {
    container.innerHTML = '<p class="panel-sub">Belum ada opsi makanan eksternal.</p>';
    return;
  }

  const table = document.createElement('table');
  table.className = 'data-table';
  table.innerHTML = `
    <thead>
      <tr>
        <th>ID</th>
        <th>Nama Makanan / Kantin</th>
        <th>Tier</th>
        <th>Harga (IDR)</th>
        <th>Lapar (-)</th>
        <th>Energi (+)</th>
        <th>Mood</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      ${entries.map(([fid, f]) => `
        <tr data-fid="${fid}">
          <td class="mono">${fid}</td>
          <td><input class="control-input ef-label" value="${f.label || ''}"></td>
          <td>
            <select class="control-select ef-tier">
              <option value="budget" ${f.tier === 'budget' ? 'selected' : ''}>budget</option>
              <option value="varied" ${f.tier === 'varied' ? 'selected' : ''}>varied</option>
            </select>
          </td>
          <td><input class="control-input mono ef-cost" type="number" step="500" value="${f.cost_idr}"></td>
          <td><input class="control-input mono ef-hunger" type="number" step="1" value="${f.hunger}"></td>
          <td><input class="control-input mono ef-energy" type="number" step="1" value="${f.energy}"></td>
          <td><input class="control-input mono ef-mood" type="number" step="0.5" value="${f.mood}"></td>
          <td><button class="btn btn-sm btn-danger ef-del" type="button">Hapus</button></td>
        </tr>
      `).join('')}
    </tbody>
  `;

  table.querySelectorAll('tbody tr').forEach((tr) => {
    const fid = tr.dataset.fid;
    const item = foods[fid];
    tr.querySelector('.ef-label').addEventListener('change', (e) => { item.label = e.target.value; markDirty(); });
    tr.querySelector('.ef-tier').addEventListener('change', (e) => { item.tier = e.target.value; markDirty(); });
    tr.querySelector('.ef-cost').addEventListener('change', (e) => { item.cost_idr = Number(e.target.value); markDirty(); });
    tr.querySelector('.ef-hunger').addEventListener('change', (e) => { item.hunger = Number(e.target.value); markDirty(); });
    tr.querySelector('.ef-energy').addEventListener('change', (e) => { item.energy = Number(e.target.value); markDirty(); });
    tr.querySelector('.ef-mood').addEventListener('change', (e) => { item.mood = Number(e.target.value); markDirty(); });
    tr.querySelector('.ef-del').addEventListener('click', () => {
      delete foods[fid];
      (worldConfig.world_graph?.locations || []).forEach((n) => {
        n.food_options = (n.food_options || []).filter((x) => x !== fid);
      });
      markDirty();
      renderExternalFoods(container, worldConfig);
    });
  });

  container.innerHTML = '';
  container.appendChild(table);
}

function renderScheduleMatrix(container, worldConfig) {
  const sched = worldConfig?.academics?.weekly_schedule;
  if (!sched) return;
  const days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];
  const slots = worldConfig.environment.steps_per_day || 5;

  const table = document.createElement('table');
  table.className = 'schedule-grid';
  table.innerHTML = `
    <thead>
      <tr>
        <th>Day</th>
        ${Array.from({ length: slots }, (_, i) => `<th class="mono">Slot ${i}</th>`).join('')}
      </tr>
    </thead>
    <tbody>
      ${days.map((d) => `
        <tr>
          <td class="mono" style="text-align: left; font-weight: 500;">${d}</td>
          ${Array.from({ length: slots }, (_, i) => {
            const on = Boolean((sched[d] || [])[i]);
            return `<td><button type="button" class="schedule-cell-btn ${on ? 'is-on' : ''}" data-day="${d}" data-slot="${i}">${on ? 'CLASS' : '-'}</button></td>`;
          }).join('')}
        </tr>
      `).join('')}
    </tbody>
  `;

  table.querySelectorAll('.schedule-cell-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const d = btn.dataset.day;
      const i = Number(btn.dataset.slot);
      const next = sched[d][i] ? 0 : 1;
      sched[d][i] = next;
      btn.classList.toggle('is-on', Boolean(next));
      btn.textContent = next ? 'CLASS' : '-';
      markDirty();
    });
  });

  container.innerHTML = '';
  container.appendChild(table);
}

export function initEnvConfigPage(root) {
  const sectionsEl = root.querySelector('#env-config-sections');
  const foodsEl = root.querySelector('#external-foods-container');
  const scheduleEl = root.querySelector('#weekly-schedule-container');
  const saveNameEl = root.querySelector('#save-config-name');
  const setActiveEl = root.querySelector('#save-set-active');
  const metaBarEl = root.querySelector('#config-meta-bar');

  root.querySelector('#btn-add-food').addEventListener('click', () => {
    const foods = appState.worldConfig.external_foods || {};
    const idx = Object.keys(foods).length + 1;
    const fid = `makanan_${idx}`;
    foods[fid] = {
      id: fid,
      label: `Warung / Kantin ${idx}`,
      tier: 'budget',
      cost_idr: 15000,
      hunger: -55,
      energy: 18,
      energy_cost: 2,
      mood: 4,
    };
    appState.worldConfig.external_foods = foods;
    markDirty();
    renderExternalFoods(foodsEl, appState.worldConfig);
    showToast(`Opsi makanan '${fid}' ditambahkan. Centang di Spatial Graph -> Inspector lokasi untuk mengaktifkannya.`);
  });

  root.querySelector('#btn-save-custom').addEventListener('click', async () => {
    const name = saveNameEl.value.trim();
    if (!name) {
      showToast('Masukkan nama config lokal (mis. my_campus_v1).');
      return;
    }
    try {
      const payload = {
        name,
        config: appState.worldConfig,
        overwrite: true,
        expected_sha256: appState.activeMeta.scope === 'custom' && appState.activeMeta.name === name
          ? appState.activeMeta.sha256
          : null,
        set_active_env_config: setActiveEl.checked,
      };
      const res = await api.saveConfig(payload);
      appState.activeMeta = { scope: 'custom', name: res.name, sha256: res.sha256, dirty: false };
      showToast(`Tersimpan ke ${res.path}`);
      notify('configs_changed');
    } catch (err) {
      showToast(`Gagal simpan: ${err.message}`, 4500);
    }
  });

  root.querySelector('#btn-export-yaml').addEventListener('click', async () => {
    try {
      const name = saveNameEl.value.trim() || appState.activeMeta.name || 'custom_export';
      const res = await api.saveConfig({ name, config: appState.worldConfig, overwrite: true });
      const blob = new Blob([res.yaml_text], { type: 'text/yaml' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${name}.yaml`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (err) {
      showToast(`Gagal export: ${err.message}`);
    }
  });

  return {
    refresh() {
      const cfg = appState.worldConfig;
      if (!cfg) return;
      saveNameEl.value = appState.activeMeta.scope === 'custom' ? appState.activeMeta.name : `${appState.activeMeta.name}_custom`;
      metaBarEl.textContent = `Source: ${appState.activeMeta.scope}/${appState.activeMeta.name} | sha256: ${(appState.activeMeta.sha256 || 'unsaved').slice(0, 12)} | dirty: ${appState.activeMeta.dirty}`;

      renderExternalFoods(foodsEl, cfg);

      sectionsEl.innerHTML = '';
      SECTION_DEFS.forEach((sec, idx) => {
        const obj = getAtPath(cfg, sec.objPath);
        if (!obj) return;
        const details = document.createElement('details');
        details.className = 'section-disclosure';
        if (idx < 2) details.open = true;
        const summary = document.createElement('summary');
        summary.textContent = sec.title;
        const body = document.createElement('div');
        body.className = 'disclosure-body';
        renderObjectFields(body, obj);
        details.appendChild(summary);
        details.appendChild(body);
        sectionsEl.appendChild(details);
      });

      renderScheduleMatrix(scheduleEl, cfg);
    },
  };
}
