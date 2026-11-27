import { appState, markDirty, showToast, subscribe } from '../state.js';
import { escapeHtml, formatIdr, nextUniqueId } from '../utils.js';

const CORE_INGREDIENTS = new Set(['rice', 'egg', 'chicken', 'tempe']);
const DEFAULT_ING_LABELS = {
  rice: 'Beras / Nasi',
  egg: 'Telur',
  chicken: 'Ayam',
  tempe: 'Tempe',
};

const ADVANCED_SECTIONS = [
  { key: 'pantry_cooking_adv', title: 'Gate & Aturan Dapur Lanjutan', objPath: ['pantry_and_cooking', 'cooking_costs_and_gates'] },
  { key: 'biometrics_dyn', title: 'Dinamika Biometrik per Step', objPath: ['biometrics_and_psychology', 'step_dynamics'] },
  { key: 'academics', title: 'Beban Tugas & Stres Akademik', objPath: ['academics'] },
  { key: 'rewards', title: 'Bobot Reward & Potential Shaping', objPath: ['rewards'] },
  { key: 'termination', title: 'Aturan Terminasi Episode', objPath: ['termination'] },
];

function getAtPath(obj, path) {
  return path.reduce((acc, k) => (acc && typeof acc === 'object' ? acc[k] : undefined), obj);
}

function renderFinanceCore(stripEl, fieldsEl, worldConfig) {
  const fin = worldConfig.finance || {};
  const days = Math.max(1, Number(worldConfig.environment?.episode_days || 30));
  const fixedExpenses = fin.fixed_expenses || {};
  const totalFixed = Object.values(fixedExpenses).reduce((a, b) => a + Number(b || 0), 0);
  const netOperational = Math.max(0, Number(fin.monthly_budget || 0) - totalFixed);
  const perDay = netOperational / days;

  stripEl.innerHTML = `
    <div class="metric-box"><div class="metric-box-label">Anggaran Bulanan</div><div class="metric-box-value num">${formatIdr(fin.monthly_budget)}</div></div>
    <div class="metric-box"><div class="metric-box-label">Biaya Tetap</div><div class="metric-box-value num">${formatIdr(totalFixed)}</div></div>
    <div class="metric-box"><div class="metric-box-label">Saldo Operasional Bersih</div><div class="metric-box-value num">${formatIdr(netOperational)}</div></div>
    <div class="metric-box"><div class="metric-box-label">Jatah Rata-rata / Hari</div><div class="metric-box-value num">${formatIdr(perDay)}</div></div>
  `;

  const fixedEntries = Object.entries(fixedExpenses);
  const firstFixedKey = fixedEntries.length > 0 ? fixedEntries[0][0] : 'mobile_data';
  const firstFixedVal = fixedEntries.length > 0 ? fixedEntries[0][1] : 0;

  fieldsEl.innerHTML = `
    <div class="field-group">
      <label for="fin-monthly">Anggaran Bulanan (IDR)</label>
      <input id="fin-monthly" class="control-input num" type="number" min="10000" step="50000" value="${fin.monthly_budget ?? 1500000}">
    </div>
    <div class="field-group">
      <label for="fin-fixed">Biaya Tetap (${escapeHtml(firstFixedKey)})</label>
      <input id="fin-fixed" class="control-input num" type="number" min="0" step="5000" value="${firstFixedVal}">
    </div>
    <div class="field-group">
      <label for="fin-low-alert" class="tooltip-trigger" data-tooltip="Saldo di bawah ambang ini menandai kondisi keuangan menipis untuk agen.">Ambang Saldo Menipis (IDR)</label>
      <input id="fin-low-alert" class="control-input num" type="number" min="0" step="25000" value="${fin.alerts?.low_balance_threshold ?? 250000}">
    </div>
  `;

  fieldsEl.querySelector('#fin-monthly').addEventListener('change', (e) => {
    fin.monthly_budget = Math.max(1000, Number(e.target.value || fin.monthly_budget));
    markDirty();
    renderFinanceCore(stripEl, fieldsEl, worldConfig);
  });
  fieldsEl.querySelector('#fin-fixed').addEventListener('change', (e) => {
    fin.fixed_expenses = fin.fixed_expenses || {};
    fin.fixed_expenses[firstFixedKey] = Math.max(0, Number(e.target.value || 0));
    markDirty();
    renderFinanceCore(stripEl, fieldsEl, worldConfig);
  });
  fieldsEl.querySelector('#fin-low-alert').addEventListener('change', (e) => {
    fin.alerts = fin.alerts || {};
    fin.alerts.low_balance_threshold = Math.max(0, Number(e.target.value || 0));
    markDirty();
  });
}

function renderLifestyleHabits(container, worldConfig) {
  const gates = worldConfig?.pantry_and_cooking?.cooking_costs_and_gates;
  const boredom = worldConfig?.pantry_and_cooking?.boredom_mechanics;
  const bioInit = worldConfig?.biometrics_and_psychology?.initial_state;
  const craving = worldConfig?.biometrics_and_psychology?.craving_and_cafe;
  if (!gates || !boredom || !bioInit || !craving) return;

  container.innerHTML = `
    <div class="stack-col" style="gap: 8px;">
      <div class="field-label" style="color: var(--ink-primary);">Kemalasan &amp; Biaya Energi Memasak</div>
      <div class="field-row-3">
        <div class="field-group">
          <label for="hab-cook-cost" class="tooltip-trigger" data-tooltip="Energi yang terpakai sekali masak segar. Semakin tinggi, semakin malas/berat untuk masak.">Energi Masak Segar</label>
          <input id="hab-cook-cost" class="control-input num" type="number" min="0" step="1" value="${gates.cook_fresh_energy_cost}">
        </div>
        <div class="field-group">
          <label for="hab-cook-min-e" class="tooltip-trigger" data-tooltip="Batas energi minimal (0–100) agar mau masak segar. Di bawah angka ini agen memilih cara makan lain.">Energi Minimal Untuk Masak</label>
          <input id="hab-cook-min-e" class="control-input num" type="number" min="0" max="100" step="1" value="${gates.cook_fresh_min_energy}">
        </div>
        <div class="field-group">
          <label for="hab-cook-min-m" class="tooltip-trigger" data-tooltip="Batas mood minimal (0–100) agar mau masak segar.">Mood Minimal Untuk Masak</label>
          <input id="hab-cook-min-m" class="control-input num" type="number" min="0" max="100" step="1" value="${gates.cook_fresh_min_mood}">
        </div>
      </div>
      <div class="field-row-3">
        <div class="field-group">
          <label for="hab-prep-cost" class="tooltip-trigger" data-tooltip="Energi yang terpakai saat memasak beberapa porsi sekaligus (meal prep).">Energi Meal Prep (Batch)</label>
          <input id="hab-prep-cost" class="control-input num" type="number" min="0" step="1" value="${gates.meal_prep_energy_cost}">
        </div>
        <div class="field-group">
          <label for="hab-reheat-cost" class="tooltip-trigger" data-tooltip="Energi untuk memanaskan stok meal prep dari kulkas.">Energi Panaskan Stok</label>
          <input id="hab-reheat-cost" class="control-input num" type="number" min="0" step="1" value="${gates.reheat_prep_energy_cost}">
        </div>
        <div class="field-group">
          <label for="hab-boredom" class="tooltip-trigger" data-tooltip="Seberapa cepat bosan makan menu yang sama (0 = tidak gampang bosan, 1 = normal, 1.5 = cepat bosan).">Sensitivitas Bosan Menu</label>
          <input id="hab-boredom" class="control-input num" type="number" min="0" max="3" step="0.05" value="${boredom.sensitivity}">
        </div>
      </div>
    </div>

    <div class="stack-col" style="gap: 8px; padding-top: 10px; border-top: 1px solid var(--border-subtle);">
      <div class="field-label" style="color: var(--ink-primary);">Rasa Jenuh (Craving Kafe) &amp; Kondisi Awal</div>
      <div class="field-row-3">
        <div class="field-group">
          <label for="hab-crav-thresh" class="tooltip-trigger" data-tooltip="Level jenuh (0–100) saat agen mulai terdistraksi belajar di kamar dan butuh suasana baru.">Ambang Mulai Jenuh</label>
          <input id="hab-crav-thresh" class="control-input num" type="number" min="0" max="100" step="1" value="${craving.distraction_threshold}">
        </div>
        <div class="field-group">
          <label for="hab-crav-pen" class="tooltip-trigger" data-tooltip="Pengurangan mood per step ketika rasa jenuh melewati ambang (angka negatif).">Penalti Mood Saat Jenuh</label>
          <input id="hab-crav-pen" class="control-input num" type="number" step="0.5" value="${craving.distraction_mood_penalty_per_step}">
        </div>
        <div class="field-group">
          <label for="hab-crav-bonus" class="tooltip-trigger" data-tooltip="Tambahan laju jenuh per step ketika sedang stres tinggi atau cuaca panas terik.">Tambahan Jenuh (Stres/Panas)</label>
          <input id="hab-crav-bonus" class="control-input num" type="number" min="0" step="0.5" value="${craving.stress_or_heat_craving_bonus}">
        </div>
      </div>
      <div class="field-row-3">
        <div class="field-group">
          <label for="hab-init-e">Energi Awal (0–100)</label>
          <input id="hab-init-e" class="control-input num" type="number" min="0" max="100" step="1" value="${bioInit.energy}">
        </div>
        <div class="field-group">
          <label for="hab-init-m">Mood Awal (0–100)</label>
          <input id="hab-init-m" class="control-input num" type="number" min="0" max="100" step="1" value="${bioInit.mood}">
        </div>
        <div class="field-group">
          <label for="hab-init-c">Jenuh / Craving Awal (0–100)</label>
          <input id="hab-init-c" class="control-input num" type="number" min="0" max="100" step="1" value="${bioInit.cafe_craving}">
        </div>
      </div>
    </div>
  `;

  const bindNum = (id, setter, minVal = null, maxVal = null) => {
    const el = container.querySelector(id);
    if (!el) return;
    el.addEventListener('change', () => {
      let n = Number(el.value);
      if (!Number.isFinite(n)) return;
      if (minVal !== null) n = Math.max(minVal, n);
      if (maxVal !== null) n = Math.min(maxVal, n);
      el.value = String(n);
      setter(n);
      markDirty();
    });
  };

  bindNum('#hab-cook-cost', (n) => { gates.cook_fresh_energy_cost = n; }, 0);
  bindNum('#hab-cook-min-e', (n) => { gates.cook_fresh_min_energy = n; }, 0, 100);
  bindNum('#hab-cook-min-m', (n) => { gates.cook_fresh_min_mood = n; }, 0, 100);
  bindNum('#hab-prep-cost', (n) => { gates.meal_prep_energy_cost = n; }, 0);
  bindNum('#hab-reheat-cost', (n) => { gates.reheat_prep_energy_cost = n; }, 0);
  bindNum('#hab-boredom', (n) => { boredom.sensitivity = n; }, 0, 5);
  bindNum('#hab-crav-thresh', (n) => { craving.distraction_threshold = n; }, 0, 100);
  bindNum('#hab-crav-pen', (n) => { craving.distraction_mood_penalty_per_step = n; });
  bindNum('#hab-crav-bonus', (n) => { craving.stress_or_heat_craving_bonus = n; }, 0);
  bindNum('#hab-init-e', (n) => { bioInit.energy = n; }, 0, 100);
  bindNum('#hab-init-m', (n) => { bioInit.mood = n; }, 0, 100);
  bindNum('#hab-init-c', (n) => { bioInit.cafe_craving = n; }, 0, 100);
}

function renderIngredientsInDays(container, worldConfig) {
  const ings = worldConfig?.pantry_and_cooking?.ingredients;
  const initPantry = worldConfig?.pantry_and_cooking?.initial_pantry || {};
  const stepsPerDay = Math.max(1, Number(worldConfig?.environment?.steps_per_day || 5));
  if (!ings) return;

  container.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Nama Bahan</th>
          <th>Harga / Pak</th>
          <th>Unit / Pak</th>
          <th class="tooltip-trigger" data-tooltip="Otomatis dikonversi ke step simulasi (${stepsPerDay} step = 1 hari).">Tahan (Hari)</th>
          <th>Stok Awal</th>
          <th class="tooltip-trigger" data-tooltip="Tingkat kenyang per 1 unit (0–100, positif). Untuk minuman seperti kopi bisa 0–10.">Kenyang (0–100)</th>
          <th>Energi (+)</th>
          <th>Mood</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        ${Object.entries(ings).map(([name, spec]) => {
          const daysVal = Math.round((Number(spec.shelf_life_steps || 1) / stepsPerDay) * 10) / 10;
          const isCore = CORE_INGREDIENTS.has(name);
          const displayLabel = spec.label || DEFAULT_ING_LABELS[name] || name;
          const nut = spec.nutrition_per_unit || { hunger: -20, energy: 8, mood: 0 };
          return `
            <tr data-ing="${escapeHtml(name)}">
              <td>
                <input class="control-input ing-label" value="${escapeHtml(displayLabel)}">
                <div class="hint-inline code-id" style="margin-top: 2px;">${escapeHtml(name)}</div>
              </td>
              <td><input class="control-input num ing-price" type="number" min="1000" step="500" value="${spec.pack_price_idr}"></td>
              <td><input class="control-input num ing-units" type="number" min="1" step="1" value="${spec.units_per_pack}"></td>
              <td><input class="control-input num ing-days" type="number" min="0.5" step="0.5" value="${daysVal}"></td>
              <td><input class="control-input num ing-init" type="number" min="0" step="1" value="${initPantry[name] ?? 0}"></td>
              <td><input class="control-input num ing-satiety" type="number" min="0" max="100" step="1" value="${Math.abs(Number(nut.hunger ?? 0))}"></td>
              <td><input class="control-input num ing-energy" type="number" step="1" value="${nut.energy ?? 0}"></td>
              <td><input class="control-input num ing-mood" type="number" step="0.5" value="${nut.mood ?? 0}"></td>
              <td>
                ${isCore
                  ? '<span class="hint-inline">Dasar</span>'
                  : '<button class="btn btn-sm btn-danger ing-del" type="button">Hapus</button>'
                }
              </td>
            </tr>
          `;
        }).join('')}
      </tbody>
    </table>
  `;

  container.querySelectorAll('tbody tr').forEach((tr) => {
    const key = tr.dataset.ing;
    const item = ings[key];
    item.nutrition_per_unit = item.nutrition_per_unit || { hunger: -20, energy: 8, mood: 0 };
    tr.querySelector('.ing-label').addEventListener('change', (e) => {
      item.label = e.target.value.trim() || key;
      markDirty();
    });
    tr.querySelector('.ing-price').addEventListener('change', (e) => {
      item.pack_price_idr = Math.max(100, Number(e.target.value || item.pack_price_idr));
      markDirty();
    });
    tr.querySelector('.ing-units').addEventListener('change', (e) => {
      item.units_per_pack = Math.max(1, Math.round(Number(e.target.value || item.units_per_pack)));
      markDirty();
    });
    tr.querySelector('.ing-days').addEventListener('change', (e) => {
      const days = Math.max(0.5, Number(e.target.value || 1));
      item.shelf_life_steps = Math.max(1, Math.round(days * stepsPerDay));
      markDirty();
    });
    tr.querySelector('.ing-init').addEventListener('change', (e) => {
      worldConfig.pantry_and_cooking.initial_pantry[key] = Math.max(0, Math.round(Number(e.target.value || 0)));
      markDirty();
    });
    tr.querySelector('.ing-satiety').addEventListener('change', (e) => {
      const sat = Math.max(0, Math.min(100, Number(e.target.value || 0)));
      item.nutrition_per_unit.hunger = -sat;
      e.target.value = String(sat);
      markDirty();
    });
    tr.querySelector('.ing-energy').addEventListener('change', (e) => {
      item.nutrition_per_unit.energy = Number(e.target.value || 0);
      markDirty();
    });
    tr.querySelector('.ing-mood').addEventListener('change', (e) => {
      item.nutrition_per_unit.mood = Number(e.target.value || 0);
      markDirty();
    });
    const delBtn = tr.querySelector('.ing-del');
    if (delBtn) {
      delBtn.addEventListener('click', () => {
        delete ings[key];
        delete worldConfig.pantry_and_cooking.initial_pantry[key];
        markDirty();
        renderIngredientsInDays(container, worldConfig);
        showToast(`Bahan '${key}' dihapus dari dapur.`);
      });
    }
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
        <th>Nama Makanan / Kantin</th>
        <th>Kategori</th>
        <th>Harga (IDR)</th>
        <th class="tooltip-trigger" data-tooltip="Seberapa mengenyangkan makanan ini (0–100, angka positif).">Tingkat Kenyang (0–100)</th>
        <th>Energi (+)</th>
        <th>Mood</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      ${entries.map(([fid, f]) => `
        <tr data-fid="${escapeHtml(fid)}">
          <td><input class="control-input ef-label" value="${escapeHtml(f.label || '')}"></td>
          <td>
            <select class="control-select ef-tier">
              <option value="budget" ${f.tier === 'budget' ? 'selected' : ''}>Hemat</option>
              <option value="varied" ${f.tier === 'varied' ? 'selected' : ''}>Variasi</option>
            </select>
          </td>
          <td><input class="control-input num ef-cost" type="number" min="0" step="500" value="${f.cost_idr}"></td>
          <td><input class="control-input num ef-satiety" type="number" min="1" max="100" step="1" value="${Math.abs(Number(f.hunger || 50))}"></td>
          <td><input class="control-input num ef-energy" type="number" step="1" value="${f.energy}"></td>
          <td><input class="control-input num ef-mood" type="number" step="0.5" value="${f.mood}"></td>
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
    tr.querySelector('.ef-cost').addEventListener('change', (e) => { item.cost_idr = Math.max(0, Number(e.target.value || 0)); markDirty(); });
    tr.querySelector('.ef-satiety').addEventListener('change', (e) => {
      const sat = Math.max(1, Math.min(100, Number(e.target.value || 50)));
      item.hunger = -sat;
      e.target.value = String(sat);
      markDirty();
    });
    tr.querySelector('.ef-energy').addEventListener('change', (e) => { item.energy = Number(e.target.value || 0); markDirty(); });
    tr.querySelector('.ef-mood').addEventListener('change', (e) => { item.mood = Number(e.target.value || 0); markDirty(); });
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
  const days = [
    ['monday', 'Senin'],
    ['tuesday', 'Selasa'],
    ['wednesday', 'Rabu'],
    ['thursday', 'Kamis'],
    ['friday', 'Jumat'],
    ['saturday', 'Sabtu'],
    ['sunday', 'Minggu'],
  ];
  const slots = worldConfig.environment.steps_per_day || 5;
  const defs = worldConfig.environment.step_definitions || {};

  const table = document.createElement('table');
  table.className = 'schedule-grid';
  table.innerHTML = `
    <thead>
      <tr>
        <th>Hari</th>
        ${Array.from({ length: slots }, (_, i) => {
          const sd = defs[i] || defs[String(i)] || {};
          const slotTitle = sd.name || `Slot ${i}`;
          const hours = sd.hours ? `<div class="hint-inline num">${escapeHtml(sd.hours)}</div>` : '';
          return `<th>${escapeHtml(slotTitle)}${hours}</th>`;
        }).join('')}
      </tr>
    </thead>
    <tbody>
      ${days.map(([key, label]) => `
        <tr>
          <td style="text-align: left; font-weight: 600;">${label}</td>
          ${Array.from({ length: slots }, (_, i) => {
            const on = Boolean((sched[key] || [])[i]);
            return `<td><button type="button" class="schedule-cell-btn ${on ? 'is-on' : ''}" data-day="${key}" data-slot="${i}">${on ? 'KULIAH' : '-'}</button></td>`;
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
      if (next === 1 && i === slots - 1) {
        showToast('Catatan: slot terakhir adalah jam malam (curfew untuk lokasi luar yang tutup malam).');
      }
      sched[d][i] = next;
      btn.classList.toggle('is-on', Boolean(next));
      btn.textContent = next ? 'KULIAH' : '-';
      markDirty();
    });
  });

  container.innerHTML = '';
  container.appendChild(table);
}

function renderAdvancedFields(container, targetObj) {
  Object.entries(targetObj).forEach(([k, v]) => {
    if (k === 'weekly_schedule') return;
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      const sub = document.createElement('div');
      sub.className = 'stack-col';
      sub.style.gap = '6px';
      sub.style.paddingTop = '6px';
      const title = document.createElement('div');
      title.className = 'field-label';
      title.style.color = 'var(--ink-primary)';
      title.textContent = k;
      sub.appendChild(title);
      renderAdvancedFields(sub, v);
      container.appendChild(sub);
      return;
    }
    const row = document.createElement('div');
    row.className = 'field-row-2';
    row.style.alignItems = 'center';
    const label = document.createElement('label');
    label.className = 'field-label code-id';
    label.textContent = k;

    if (typeof v === 'boolean') {
      const wrap = document.createElement('label');
      wrap.className = 'checkbox-row';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = v;
      const txt = document.createElement('span');
      txt.textContent = v ? 'Aktif' : 'Nonaktif';
      input.addEventListener('change', () => {
        targetObj[k] = input.checked;
        txt.textContent = input.checked ? 'Aktif' : 'Nonaktif';
        markDirty();
      });
      wrap.appendChild(input);
      wrap.appendChild(txt);
      row.appendChild(label);
      row.appendChild(wrap);
      container.appendChild(row);
      return;
    }

    const input = document.createElement('input');
    input.className = 'control-input num';
    if (Array.isArray(v)) {
      input.value = v.join(', ');
      input.addEventListener('change', () => {
        const parsed = input.value.split(',').map((s) => Number(s.trim())).filter((n) => Number.isFinite(n));
        if (parsed.length > 0) {
          targetObj[k] = parsed;
          markDirty();
        }
      });
    } else {
      input.type = typeof v === 'number' ? 'number' : 'text';
      input.step = 'any';
      input.value = v ?? '';
      input.addEventListener('change', () => {
        if (typeof v === 'number') {
          if (input.value.trim() === '') {
            input.value = String(targetObj[k]);
            return;
          }
          targetObj[k] = Number(input.value);
        } else {
          targetObj[k] = input.value;
        }
        markDirty();
      });
    }
    row.appendChild(label);
    row.appendChild(input);
    container.appendChild(row);
  });
}

export function initEnvConfigPage(root, hostActions = {}) {
  const finStripEl = root.querySelector('#finance-summary-strip');
  const finFieldsEl = root.querySelector('#finance-core-fields');
  const habitsEl = root.querySelector('#lifestyle-habits-container');
  const sectionsEl = root.querySelector('#env-config-sections');
  const foodsEl = root.querySelector('#external-foods-container');
  const ingsEl = root.querySelector('#ingredients-days-container');
  const scheduleEl = root.querySelector('#weekly-schedule-container');
  const metaBarEl = root.querySelector('#config-meta-bar');
  const openSaveBtn = root.querySelector('#btn-open-save-dialog');
  const addIngBox = root.querySelector('#add-ingredient-box');

  function renderMetaBar() {
    const m = appState.activeMeta;
    metaBarEl.innerHTML = `
      <div class="kv-row"><span class="kv-key">Sumber Config</span><span class="kv-val code-id">${escapeHtml(m.scope)}/${escapeHtml(m.name)}</span></div>
      <div class="kv-row"><span class="kv-key">Status</span><span class="kv-val">${m.dirty ? 'Ada perubahan belum disimpan' : 'Tersinkron dengan disk'}</span></div>
      <div class="kv-row"><span class="kv-key">SHA-256</span><span class="kv-val code-id">${escapeHtml((m.sha256 || 'belum tersimpan').slice(0, 16))}</span></div>
    `;
  }

  subscribe((_st, reason) => {
    if (reason === 'dirty' || reason === 'configs_changed') renderMetaBar();
  });

  if (openSaveBtn && hostActions.openSaveDialog) {
    openSaveBtn.addEventListener('click', hostActions.openSaveDialog);
  }

  root.querySelector('#btn-add-food').addEventListener('click', () => {
    const foods = appState.worldConfig.external_foods || {};
    const fid = nextUniqueId('makanan', Object.keys(foods));
    foods[fid] = {
      id: fid,
      label: `Warung / Kantin ${fid.split('_')[1] || ''}`.trim(),
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
    showToast(`Opsi makanan '${fid}' ditambahkan. Aktifkan per lokasi di tab Peta & Mobilitas.`);
  });

  root.querySelector('#btn-toggle-add-ing').addEventListener('click', () => {
    addIngBox.hidden = !addIngBox.hidden;
    if (!addIngBox.hidden) root.querySelector('#new-ing-label').focus();
  });
  root.querySelector('#btn-cancel-add-ing').addEventListener('click', () => {
    addIngBox.hidden = true;
  });
  root.querySelector('#btn-confirm-add-ing').addEventListener('click', () => {
    const labelText = root.querySelector('#new-ing-label').value.trim();
    if (!labelText) {
      showToast('Masukkan nama bahan dapur terlebih dahulu (mis. Kopi Sachet).');
      root.querySelector('#new-ing-label').focus();
      return;
    }
    const ings = appState.worldConfig.pantry_and_cooking.ingredients;
    const slugBase = labelText.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'bahan';
    const key = Object.prototype.hasOwnProperty.call(ings, slugBase)
      ? nextUniqueId(slugBase, Object.keys(ings))
      : slugBase;
    const stepsPerDay = Math.max(1, Number(appState.worldConfig?.environment?.steps_per_day || 5));
    const daysVal = Math.max(0.5, Number(root.querySelector('#new-ing-days').value || 7));
    const satVal = Math.max(0, Math.min(100, Number(root.querySelector('#new-ing-satiety').value || 0)));

    ings[key] = {
      label: labelText,
      pack_price_idr: Math.max(500, Number(root.querySelector('#new-ing-price').value || 15000)),
      units_per_pack: Math.max(1, Math.round(Number(root.querySelector('#new-ing-units').value || 5))),
      shelf_life_steps: Math.max(1, Math.round(daysVal * stepsPerDay)),
      nutrition_per_unit: {
        hunger: -satVal,
        energy: Number(root.querySelector('#new-ing-energy').value || 10),
        mood: Number(root.querySelector('#new-ing-mood').value || 4),
      },
    };
    appState.worldConfig.pantry_and_cooking.initial_pantry[key] = Math.max(
      0,
      Math.round(Number(root.querySelector('#new-ing-init').value || 0)),
    );
    root.querySelector('#new-ing-label').value = '';
    addIngBox.hidden = true;
    markDirty();
    renderIngredientsInDays(ingsEl, appState.worldConfig);
    showToast(`Bahan dapur '${labelText}' ditambahkan. Agen kini bisa membeli dan menggunakannya di dapur.`);
  });

  return {
    refresh() {
      const cfg = appState.worldConfig;
      if (!cfg) return;
      renderMetaBar();
      renderFinanceCore(finStripEl, finFieldsEl, cfg);
      renderLifestyleHabits(habitsEl, cfg);
      renderScheduleMatrix(scheduleEl, cfg);
      renderExternalFoods(foodsEl, cfg);
      renderIngredientsInDays(ingsEl, cfg);

      sectionsEl.innerHTML = '';
      ADVANCED_SECTIONS.forEach((sec) => {
        const obj = getAtPath(cfg, sec.objPath);
        if (!obj) return;
        const details = document.createElement('details');
        details.className = 'section-disclosure';
        const summary = document.createElement('summary');
        summary.textContent = sec.title;
        const body = document.createElement('div');
        body.className = 'disclosure-body';
        renderAdvancedFields(body, obj);
        details.appendChild(summary);
        details.appendChild(body);
        sectionsEl.appendChild(details);
      });
    },
  };
}
