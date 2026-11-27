import { api } from '../api.js';
import { computePixelDistanceKm, renderSpatialGraph } from '../graph_canvas.js';
import { appState, markDirty, showToast } from '../state.js';

function numOrNull(val) {
  if (val === '' || val === null || val === undefined) return null;
  const n = Number(val);
  return Number.isFinite(n) ? n : null;
}

function formatIdr(n) {
  return `Rp ${Math.round(Number(n || 0)).toLocaleString('id-ID')}`;
}

function describeAmbientDelta(val, positiveDesc, negativeDesc) {
  const n = Number(val || 0);
  if (n > 0) return `+${n}/step (${positiveDesc})`;
  if (n < 0) return `${n}/step (${negativeDesc})`;
  return '0/step (netral)';
}

function estimateRideHailingBounds(edge, preset) {
  if (!preset) return null;
  const p = edge.pricing || {};
  const minKm = p.min_distance_km ?? preset.min_distance_km;
  const minFare = p.min_fare_idr || preset.min_fare_idr;
  const perKm = p.per_km_idr || preset.per_km_idr;
  const appFee = p.app_fee_idr ?? preset.app_fee_idr;
  const peakMult = p.peak_multiplier ?? preset.peak_multiplier;
  const rainMult = p.rain_surge_multiplier ?? preset.rain_surge_multiplier;

  const extraKm = Math.max(0, Number(edge.distance_km || 0) - Number(minKm));
  const normLow = Math.round(minFare[0] + extraKm * perKm[0] + appFee[0]);
  const normHigh = Math.round(minFare[1] + extraKm * perKm[1] + appFee[1]);
  const surgeLow = Math.round((minFare[0] + extraKm * perKm[0]) * peakMult * rainMult + appFee[0]);
  const surgeHigh = Math.round((minFare[1] + extraKm * perKm[1]) * peakMult * rainMult + appFee[1]);
  return { minKm, minFare, perKm, appFee, peakMult, rainMult, normLow, normHigh, surgeLow, surgeHigh };
}

const LOCATION_ARCHETYPES = {
  home: {
    label: 'Kos / Tempat Tinggal',
    zone: 'home',
    ambient_mood_delta: 0.0,
    ambient_stress_delta: 0.0,
    craving_gain_per_step: 4.0,
    can_sleep: true,
    has_kitchen: true,
    has_grocery: true,
    is_class_venue: false,
    can_study: true,
    closes_at_night: false,
    study_profile: { energy_cost: 10, stress_reduction: 12, mood_delta: 0, affected_by_craving_distraction: true },
  },
  campus: {
    label: 'Gedung Kuliah / Fakultas',
    zone: 'campus',
    ambient_mood_delta: 0.0,
    ambient_stress_delta: 0.0,
    craving_gain_per_step: 3.0,
    can_sleep: false,
    has_kitchen: false,
    has_grocery: false,
    is_class_venue: true,
    can_study: false,
    closes_at_night: true,
    study_profile: { energy_cost: 6, stress_reduction: 10, mood_delta: 0, affected_by_craving_distraction: false },
  },
  study_hub: {
    label: 'Perpustakaan / Study Hub',
    zone: 'campus',
    ambient_mood_delta: 0.5,
    ambient_stress_delta: -0.5,
    craving_gain_per_step: 2.0,
    can_sleep: false,
    has_kitchen: false,
    has_grocery: false,
    is_class_venue: false,
    can_study: true,
    closes_at_night: true,
    study_profile: {
      energy_cost: 5,
      stress_reduction: 16,
      mood_delta: 2,
      affected_by_craving_distraction: false,
      craving_relief: 24,
      craving_floor: 15,
      buff_duration_steps: 4,
      buff_craving_rate_multiplier: 0.65,
    },
  },
  cafe: {
    label: 'Kafe / Working Space',
    zone: 'third_place',
    ambient_mood_delta: 1.0,
    ambient_stress_delta: -1.0,
    craving_gain_per_step: 0.0,
    can_sleep: false,
    has_kitchen: false,
    has_grocery: false,
    is_class_venue: false,
    can_study: true,
    closes_at_night: true,
    study_profile: {
      energy_cost: 5,
      stress_reduction: 16,
      mood_delta: 4,
      affected_by_craving_distraction: false,
      craving_reset: true,
      buff_duration_steps: 10,
      buff_craving_rate_multiplier: 0.35,
    },
    leisure_spend_idr: [25000, 42000],
    leisure_hunger_delta: -15,
    leisure_mood_gain: 26,
    leisure_stress_reduction: 26,
    leisure_energy_cost: 5,
    leisure_buff_duration_steps: 10,
    leisure_buff_craving_rate_multiplier: 0.35,
  },
  custom: {
    label: 'Lokasi Baru',
    zone: 'custom',
    ambient_mood_delta: 0.0,
    ambient_stress_delta: 0.0,
    craving_gain_per_step: 2.0,
    can_sleep: false,
    has_kitchen: false,
    has_grocery: false,
    is_class_venue: false,
    can_study: true,
    closes_at_night: true,
    study_profile: { energy_cost: 6, stress_reduction: 12, mood_delta: 0, affected_by_craving_distraction: false },
  },
};

function renderAddLocationForm(body, worldConfig, pxPerKm, onCreated, onCancel) {
  const wg = worldConfig.world_graph;
  const defaultIdx = wg.locations.length + 1;
  body.innerHTML = `
    <div class="stack-col" style="gap: 10px;">
      <div class="info-box">
        Pilih template tipe lokasi agar kapabilitas belajar, mood, dan jam malam terisi otomatis. Kamu tetap bisa ubah semua angkanya setelah dibuat.
      </div>
      <div class="field-group">
        <label>Tipe Lokasi (Template Cepat)</label>
        <select id="new-loc-type" class="control-select">
          <option value="study_hub">Perpustakaan / Study Hub (bisa belajar, redakan jenuh)</option>
          <option value="cafe">Kafe / Third Place (bisa belajar &amp; reset craving, berbayar)</option>
          <option value="campus">Gedung Kuliah / Kampus (lokasi menghadiri kelas)</option>
          <option value="home">Kos / Tempat Tinggal (bisa tidur &amp; masak)</option>
          <option value="custom">Custom Kosong (atur manual)</option>
        </select>
      </div>
      <div class="field-row-2">
        <div class="field-group">
          <label class="tooltip-trigger" data-tooltip="ID unik tanpa spasi (huruf kecil, angka, underscore). Contoh: perpus_pusat atau kafe_selatan.">Node ID</label>
          <input id="new-loc-id" class="control-input mono" value="lokasi_${defaultIdx}">
        </div>
        <div class="field-group">
          <label>Nama Tampilan</label>
          <input id="new-loc-label" class="control-input" value="Perpustakaan / Study Hub">
        </div>
      </div>
      <label class="checkbox-row">
        <input id="new-loc-connect-home" type="checkbox" checked>
        <span>Hubungkan otomatis ke Home (${wg.home_node_id}) sesuai jarak peta</span>
      </label>
      <div class="field-group" id="new-loc-mode-wrap">
        <label>Moda Rute Awal ke Home</label>
        <select id="new-loc-edge-mode" class="control-select">
          <option value="goride">GoRide (tarif otomatis berbasis jarak Zona I)</option>
          <option value="walk">Jalan Kaki (gratis, beban energi/mood dari km)</option>
          <option value="public_transit">Angkutan Umum / Tarif Manual</option>
        </select>
      </div>
      <div style="display: flex; gap: 8px; justify-content: flex-end;">
        <button id="btn-cancel-new-loc" class="btn btn-sm" type="button">Batal</button>
        <button id="btn-confirm-new-loc" class="btn btn-sm btn-primary" type="button">Tambah Lokasi</button>
      </div>
    </div>
  `;

  const typeSel = body.querySelector('#new-loc-type');
  const labelInput = body.querySelector('#new-loc-label');
  const idInput = body.querySelector('#new-loc-id');
  const connChk = body.querySelector('#new-loc-connect-home');
  const modeWrap = body.querySelector('#new-loc-mode-wrap');

  typeSel.addEventListener('change', () => {
    const tpl = LOCATION_ARCHETYPES[typeSel.value];
    if (tpl) {
      labelInput.value = tpl.label;
      idInput.value = `${typeSel.value}_${defaultIdx}`;
    }
  });
  connChk.addEventListener('change', () => {
    modeWrap.style.display = connChk.checked ? 'flex' : 'none';
  });
  body.querySelector('#btn-cancel-new-loc').addEventListener('click', onCancel);
  body.querySelector('#btn-confirm-new-loc').addEventListener('click', () => {
    const rawId = idInput.value.trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_');
    if (!rawId || wg.locations.some((n) => n.id === rawId)) {
      showToast('ID lokasi kosong atau sudah dipakai. Gunakan ID unik.');
      return;
    }
    const tpl = structuredClone(LOCATION_ARCHETYPES[typeSel.value] || LOCATION_ARCHETYPES.custom);
    const pos = { x: 240 + ((defaultIdx * 85) % 420), y: 110 + ((defaultIdx * 70) % 280) };
    const newNode = {
      id: rawId,
      ...tpl,
      label: labelInput.value.trim() || tpl.label,
      food_options: [],
      ui_position: pos,
    };
    wg.locations.push(newNode);

    if (connChk.checked) {
      const homeNode = wg.locations.find((n) => n.id === wg.home_node_id);
      const distKm = homeNode ? computePixelDistanceKm(homeNode.ui_position, pos, pxPerKm) : 2.0;
      const mode = body.querySelector('#new-loc-edge-mode').value;
      const pricing = mode === 'walk'
        ? { pricing_model: 'free' }
        : mode === 'public_transit'
          ? { pricing_model: 'manual_range', cost_normal_idr: [5000, 7000], cost_peak_idr: [6000, 8000] }
          : { pricing_model: 'ride_hailing_zone', zone_preset: 'goride_zona_1' };

      wg.edges.push({
        id: `route_${wg.home_node_id}_${rawId}`,
        label: `${wg.home_node_id} <-> ${newNode.label}`,
        source: wg.home_node_id,
        target: rawId,
        bidirectional: true,
        mode,
        distance_km: distKm,
        pricing,
      });
    }
    onCreated(rawId);
  });
}

function renderNodeInspector(body, node, worldConfig, onRerenderGraph) {
  const sp = node.study_profile || {};
  const foodEntries = Object.entries(worldConfig.external_foods || {});
  const selectedFoods = new Set(node.food_options || []);
  const isHome = worldConfig.world_graph.home_node_id === node.id;

  body.innerHTML = `
    <div class="stack-col" style="gap: 11px;">
      <div class="field-row-2">
        <div class="field-group">
          <label>Node ID</label>
          <input class="control-input mono" value="${node.id}" disabled>
        </div>
        <div class="field-group">
          <label>Nama Lokasi</label>
          <input id="nd-label" class="control-input" value="${node.label || ''}">
        </div>
      </div>

      <div class="field-row-2" style="align-items: end;">
        <div class="field-group">
          <label class="tooltip-trigger" data-tooltip="Kategori warna dan peran tempat: home (tempat tinggal), campus (area kuliah/hub), third_place (kafe/tempat santai).">Zona / Kategori</label>
          <select id="nd-zone" class="control-select">
            ${[
              ['home', 'Home (Tempat Tinggal / Kos)'],
              ['campus', 'Campus (Area Kuliah / Hub)'],
              ['third_place', 'Third Place (Kafe / Santai)'],
              ['commercial', 'Commercial'],
              ['custom', 'Custom'],
            ].map(([val, lbl]) => `<option value="${val}" ${node.zone === val ? 'selected' : ''}>${lbl}</option>`).join('')}
          </select>
        </div>
        <div>
          ${!isHome && node.can_sleep
            ? '<button id="btn-set-home" class="btn btn-sm" style="width: 100%;" type="button">Jadikan Home Utama</button>'
            : `<div class="hint-inline mono" style="padding: 6px 0;">${isHome ? 'Status: Home utama (titik awal & pulang malam)' : 'Aktifkan can_sleep untuk jadi Home'}</div>`
          }
        </div>
      </div>

      <div class="info-box">
        <strong>Efek Suasana Pasif (Per Step)</strong> — Berlaku otomatis tiap 1 sesi agen berada di lokasi ini:<br>
        <span id="nd-ambient-summary" class="mono">
          Mood ${describeAmbientDelta(node.ambient_mood_delta, 'naik tiap step', 'turun tiap step')} ·
          Stress ${describeAmbientDelta(node.ambient_stress_delta, 'bertambah', 'mereda')} ·
          Jenuh +${node.craving_gain_per_step ?? 0}/step
        </span>
      </div>

      <div class="field-row-3">
        <div class="field-group">
          <label class="tooltip-trigger" data-tooltip="Delta mood konstan yang ditambahkan ke biometrik agen setiap 1 step di lokasi ini. Positif = suasana nyaman; negatif = cepat lelah/bosan.">Mood / Step</label>
          <input id="nd-mood" class="control-input mono" type="number" step="0.5" value="${node.ambient_mood_delta ?? 0}">
        </div>
        <div class="field-group">
          <label class="tooltip-trigger" data-tooltip="Delta stres akademik konstan tiap 1 step berada di lokasi ini. Negatif = tempat menenangkan; positif = menambah tekanan.">Stress / Step</label>
          <input id="nd-stress" class="control-input mono" type="number" step="0.5" value="${node.ambient_stress_delta ?? 0}">
        </div>
        <div class="field-group">
          <label class="tooltip-trigger" data-tooltip="Seberapa cepat muncul keinginan ganti suasana (cafe craving) tiap step di sini. Kamar biasanya +3..4; kafe +0.">Jenuh / Step</label>
          <input id="nd-craving" class="control-input mono" type="number" step="0.5" min="0" value="${node.craving_gain_per_step ?? 3}">
        </div>
      </div>

      <div class="field-group">
        <label>Kapabilitas Lokasi (Apa yang bisa dilakukan di sini?)</label>
        <div class="field-row-2" style="gap: 6px;">
          <label class="checkbox-row" data-tooltip="Agen bisa tidur/istirahat malam di lokasi ini. Home utama wajib mencentang ini.">
            <input id="nd-sleep" type="checkbox" ${node.can_sleep ? 'checked' : ''}><span class="tooltip-trigger">Bisa Tidur (Sleep)</span>
          </label>
          <label class="checkbox-row" data-tooltip="Agen bisa masak segar, meal prep, dan memanaskan stok kulkas di sini.">
            <input id="nd-kitchen" type="checkbox" ${node.has_kitchen ? 'checked' : ''}><span class="tooltip-trigger">Ada Dapur (Masak)</span>
          </label>
          <label class="checkbox-row" data-tooltip="Agen bisa melakukan aktivitas belanja bahan baku dapur dari lokasi ini.">
            <input id="nd-grocery" type="checkbox" ${node.has_grocery ? 'checked' : ''}><span class="tooltip-trigger">Bisa Belanja Bahan</span>
          </label>
          <label class="checkbox-row" data-tooltip="Menandai lokasi ini sebagai ruang kuliah untuk aktivitas attend_schedule.">
            <input id="nd-class" type="checkbox" ${node.is_class_venue ? 'checked' : ''}><span class="tooltip-trigger">Lokasi Kuliah</span>
          </label>
          <label class="checkbox-row" data-tooltip="Mengizinkan sesi belajar (menurunkan stres akademik) di tempat ini.">
            <input id="nd-study" type="checkbox" ${node.can_study ? 'checked' : ''}><span class="tooltip-trigger">Bisa Dipakai Belajar</span>
          </label>
          <label class="checkbox-row" data-tooltip="Jika aktif, tempat ini tutup di slot malam sehingga agen harus pulang ke Home.">
            <input id="nd-curfew" type="checkbox" ${node.closes_at_night ? 'checked' : ''}><span class="tooltip-trigger">Tutup Malam (Curfew)</span>
          </label>
        </div>
      </div>

      ${node.can_study ? `
        <div class="stack-col" style="gap: 8px; padding-top: 8px; border-top: 1px solid var(--border-subtle);">
          <div class="field-label">Profil Belajar di Lokasi Ini</div>
          <div class="field-row-3">
            <div class="field-group">
              <label class="tooltip-trigger" data-tooltip="Jumlah poin stres akademik yang berkurang setiap 1 sesi belajar di sini.">Stres Turun (-)</label>
              <input id="nd-sp-stress" class="control-input mono" type="number" step="1" value="${sp.stress_reduction ?? 12}">
            </div>
            <div class="field-group">
              <label class="tooltip-trigger" data-tooltip="Energi yang dihabiskan untuk 1 sesi belajar di lokasi ini.">Biaya Energi</label>
              <input id="nd-sp-energy" class="control-input mono" type="number" step="1" value="${sp.energy_cost ?? 6}">
            </div>
            <div class="field-group">
              <label class="tooltip-trigger" data-tooltip="Tambahan/pengurangan mood langsung saat selesai belajar di sini.">Efek Mood Belajar</label>
              <input id="nd-sp-mood" class="control-input mono" type="number" step="1" value="${sp.mood_delta ?? 0}">
            </div>
          </div>
          <div class="field-row-2">
            <label class="checkbox-row" data-tooltip="Jika aktif (seperti di kamar kos), efektivitas belajar turun saat agen sedang jenuh/craving tinggi.">
              <input id="nd-sp-distract" type="checkbox" ${sp.affected_by_craving_distraction ? 'checked' : ''}>
              <span class="tooltip-trigger">Terpengaruh Distraksi Jenuh</span>
            </label>
            <label class="checkbox-row" data-tooltip="Jika aktif (seperti kafe), belajar di sini langsung me-reset rasa jenuh ke 0.">
              <input id="nd-sp-reset" type="checkbox" ${sp.craving_reset ? 'checked' : ''}>
              <span class="tooltip-trigger">Reset Craving (Setara Kafe)</span>
            </label>
          </div>
          <div class="field-row-2">
            <div class="field-group">
              <label class="tooltip-trigger" data-tooltip="Pengurangan nilai craving sekali kunjungan belajar (misal 24 di perpustakaan/hub).">Craving Relief</label>
              <input id="nd-sp-relief" class="control-input mono" type="number" step="1" value="${sp.craving_relief ?? 0}">
            </div>
            <div class="field-group">
              <label class="tooltip-trigger" data-tooltip="Berapa step ke depan laju kenaikan jenuh diperlambat setelah belajar di sini.">Durasi Buff (Step)</label>
              <input id="nd-sp-buff" class="control-input mono" type="number" step="1" value="${sp.buff_duration_steps ?? 0}">
            </div>
          </div>
        </div>
      ` : ''}

      ${node.zone === 'third_place' || (node.leisure_mood_gain ?? 0) > 0 ? `
        <div class="stack-col" style="gap: 8px; padding-top: 8px; border-top: 1px solid var(--border-subtle);">
          <div class="field-label">Sesi Kafe / Leisure Reset</div>
          <div class="field-row-3">
            <div class="field-group">
              <label class="tooltip-trigger" data-tooltip="Kisaran biaya jajan/kopi saat sesi santai di sini (Min–Max IDR).">Biaya Min (IDR)</label>
              <input id="nd-leis-min" class="control-input mono" type="number" step="1000" value="${(node.leisure_spend_idr || [20000, 35000])[0]}">
            </div>
            <div class="field-group">
              <label>Biaya Max (IDR)</label>
              <input id="nd-leis-max" class="control-input mono" type="number" step="1000" value="${(node.leisure_spend_idr || [20000, 35000])[1]}">
            </div>
            <div class="field-group">
              <label class="tooltip-trigger" data-tooltip="Lonjakan mood langsung ketika menyelesaikan sesi santai di tempat ini.">Mood Gain (+)</label>
              <input id="nd-leis-mood" class="control-input mono" type="number" step="1" value="${node.leisure_mood_gain ?? 20}">
            </div>
          </div>
        </div>
      ` : ''}

      <div class="field-group" style="padding-top: 8px; border-top: 1px solid var(--border-subtle);">
        <label class="tooltip-trigger" data-tooltip="Pilihan makanan eksternal yang bisa dibeli saat agen berada di lokasi ini. Daftar makanan bisa ditambah/diubah di tab Env & Rules.">Opsi Makanan di Lokasi Ini</label>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
          ${foodEntries.length === 0
            ? '<span class="hint-inline">Belum ada katalog makanan. Tambah di tab Env &amp; Rules.</span>'
            : foodEntries.map(([fk, fspec]) => `
              <label class="checkbox-row" data-tooltip="${fspec.label} (${fspec.tier}) · ${formatIdr(fspec.cost_idr)} · Lapar ${fspec.hunger}, Mood ${fspec.mood > 0 ? '+' : ''}${fspec.mood}">
                <input class="nd-food-opt" type="checkbox" value="${fk}" ${selectedFoods.has(fk) ? 'checked' : ''}>
                <span class="tooltip-trigger">${fspec.label || fk}</span>
              </label>
            `).join('')
          }
        </div>
      </div>
    </div>
  `;

  const updateSummary = () => {
    const el = body.querySelector('#nd-ambient-summary');
    if (el) {
      el.textContent =
        `Mood ${describeAmbientDelta(node.ambient_mood_delta, 'naik tiap step', 'turun tiap step')} · ` +
        `Stress ${describeAmbientDelta(node.ambient_stress_delta, 'bertambah', 'mereda')} · ` +
        `Jenuh +${node.craving_gain_per_step ?? 0}/step`;
    }
  };

  const bind = (id, fn, rerenderInspector = false) => {
    const el = body.querySelector(id);
    if (!el) return;
    el.addEventListener('change', () => {
      fn(el);
      markDirty();
      updateSummary();
      onRerenderGraph();
      if (rerenderInspector) renderNodeInspector(body, node, worldConfig, onRerenderGraph);
    });
  };

  bind('#nd-label', (el) => { node.label = el.value; });
  bind('#nd-zone', (el) => { node.zone = el.value; }, true);
  bind('#nd-mood', (el) => { node.ambient_mood_delta = Number(el.value); });
  bind('#nd-stress', (el) => { node.ambient_stress_delta = Number(el.value); });
  bind('#nd-craving', (el) => { node.craving_gain_per_step = Math.max(0, Number(el.value)); });
  bind('#nd-sleep', (el) => { node.can_sleep = el.checked; }, true);
  bind('#nd-kitchen', (el) => { node.has_kitchen = el.checked; });
  bind('#nd-grocery', (el) => { node.has_grocery = el.checked; });
  bind('#nd-class', (el) => { node.is_class_venue = el.checked; });
  bind('#nd-study', (el) => { node.can_study = el.checked; }, true);
  bind('#nd-curfew', (el) => { node.closes_at_night = el.checked; });

  node.study_profile = sp;
  bind('#nd-sp-stress', (el) => { sp.stress_reduction = Number(el.value); });
  bind('#nd-sp-energy', (el) => { sp.energy_cost = Number(el.value); });
  bind('#nd-sp-mood', (el) => { sp.mood_delta = Number(el.value); });
  bind('#nd-sp-distract', (el) => { sp.affected_by_craving_distraction = el.checked; });
  bind('#nd-sp-reset', (el) => { sp.craving_reset = el.checked; });
  bind('#nd-sp-relief', (el) => { sp.craving_relief = Number(el.value); });
  bind('#nd-sp-buff', (el) => { sp.buff_duration_steps = Number(el.value); });

  bind('#nd-leis-min', (el) => {
    const cur = node.leisure_spend_idr || [20000, 35000];
    node.leisure_spend_idr = [Number(el.value), cur[1]];
  });
  bind('#nd-leis-max', (el) => {
    const cur = node.leisure_spend_idr || [20000, 35000];
    node.leisure_spend_idr = [cur[0], Number(el.value)];
  });
  bind('#nd-leis-mood', (el) => { node.leisure_mood_gain = Number(el.value); });

  const setHomeBtn = body.querySelector('#btn-set-home');
  if (setHomeBtn) {
    setHomeBtn.addEventListener('click', () => {
      worldConfig.world_graph.home_node_id = node.id;
      markDirty();
      onRerenderGraph();
      renderNodeInspector(body, node, worldConfig, onRerenderGraph);
      showToast(`Home utama dipindah ke '${node.label || node.id}'.`);
    });
  }

  body.querySelectorAll('.nd-food-opt').forEach((cb) => {
    cb.addEventListener('change', () => {
      node.food_options = Array.from(body.querySelectorAll('.nd-food-opt:checked')).map((c) => c.value);
      markDirty();
    });
  });
}

function renderEdgeInspector(body, edge, worldConfig, onRerenderGraph) {
  const nodes = worldConfig.world_graph.locations || [];
  const p = edge.pricing || { pricing_model: 'free' };
  const zonePresets = appState.schema?.ride_hailing_zone_presets || {};
  const presetKeys = Object.keys(zonePresets);
  const activePresetKey = p.zone_preset || 'goride_zona_1';
  const activePreset = zonePresets[activePresetKey];
  const est = p.pricing_model === 'ride_hailing_zone' ? estimateRideHailingBounds(edge, activePreset) : null;

  body.innerHTML = `
    <div class="stack-col" style="gap: 10px;">
      <div class="field-row-2">
        <div class="field-group">
          <label>Route ID</label>
          <input class="control-input mono" value="${edge.id}" disabled>
        </div>
        <div class="field-group">
          <label>Label Rute</label>
          <input id="ed-label" class="control-input" value="${edge.label || ''}">
        </div>
      </div>

      <div class="field-row-3">
        <div class="field-group">
          <label>Dari</label>
          <select id="ed-src" class="control-select mono">
            ${nodes.map((n) => `<option value="${n.id}" ${edge.source === n.id ? 'selected' : ''}>${n.label || n.id}</option>`).join('')}
          </select>
        </div>
        <div class="field-group">
          <label>Ke</label>
          <select id="ed-dst" class="control-select mono">
            ${nodes.map((n) => `<option value="${n.id}" ${edge.target === n.id ? 'selected' : ''}>${n.label || n.id}</option>`).join('')}
          </select>
        </div>
        <div class="field-group">
          <label class="tooltip-trigger" data-tooltip="Otomatis terhitung dari posisi peta saat Auto-Sync aktif, namun tetap bisa diubah manual.">Jarak (km)</label>
          <input id="ed-dist" class="control-input mono" type="number" step="0.1" min="0.1" value="${edge.distance_km}">
        </div>
      </div>

      <div class="field-row-2">
        <label class="checkbox-row">
          <input id="ed-bidir" type="checkbox" ${edge.bidirectional !== false ? 'checked' : ''}>
          <span>Dua arah (PP)</span>
        </label>
        <label class="checkbox-row" data-tooltip="Kunci jarak manual agar tidak berubah otomatis saat node digeser di peta.">
          <input id="ed-lock-dist" type="checkbox" ${edge._lockDistance ? 'checked' : ''}>
          <span class="tooltip-trigger">Kunci Jarak Manual</span>
        </label>
      </div>

      <div class="field-row-2">
        <div class="field-group">
          <label>Moda Transportasi</label>
          <select id="ed-mode" class="control-select">
            ${[
              ['walk', 'Jalan Kaki (walk)'],
              ['goride', 'GoRide / Ojol Motor'],
              ['gocar', 'GoCar / Mobil'],
              ['ojol', 'Ojol Umum'],
              ['public_transit', 'Angkutan Umum / Bus'],
              ['personal', 'Kendaraan Pribadi'],
              ['custom', 'Custom'],
            ].map(([m, lbl]) => `<option value="${m}" ${edge.mode === m ? 'selected' : ''}>${lbl}</option>`).join('')}
          </select>
        </div>
        <div class="field-group">
          <label class="tooltip-trigger" data-tooltip="free: Rp 0. manual_range: isi rentang harga tetap A->B. per_km_custom: tarif dasar + per km buatan sendiri. ride_hailing_zone: pakai regulasi GoRide Zona I/II/III atau GoCar.">Model Tarif</label>
          <select id="ed-pmodel" class="control-select">
            ${[
              ['free', 'Gratis (Rp 0)'],
              ['manual_range', 'Rentang Manual (A -> B)'],
              ['per_km_custom', 'Per Km Kustom (User)'],
              ['ride_hailing_zone', 'Preset GoRide / GoCar Zona'],
            ].map(([m, lbl]) => `<option value="${m}" ${p.pricing_model === m ? 'selected' : ''}>${lbl}</option>`).join('')}
          </select>
        </div>
      </div>

      ${p.pricing_model === 'free' ? `
        <div class="info-box">
          <strong>Tarif Gratis (Rp 0)</strong> — Jika moda <code>walk</code>, beban energi &amp; mood dihitung otomatis dari <strong>${Number(edge.distance_km).toFixed(1)} km</strong> dan dibatasi oleh aturan <em>Max Walk (km)</em>.
        </div>
      ` : ''}

      ${p.pricing_model === 'manual_range' ? `
        <div class="info-box">Cocok untuk bus kampus, angkot, KRL, atau rute ojol yang harganya sudah pasti kamu hafal.</div>
        <div class="field-row-2">
          <div class="field-group">
            <label>Normal Min/Max (IDR)</label>
            <div class="field-row-2">
              <input id="ed-norm-min" class="control-input mono" type="number" step="500" value="${(p.cost_normal_idr || [5000, 7000])[0]}">
              <input id="ed-norm-max" class="control-input mono" type="number" step="500" value="${(p.cost_normal_idr || [5000, 7000])[1]}">
            </div>
          </div>
          <div class="field-group">
            <label>Jam Sibuk Min/Max (IDR)</label>
            <div class="field-row-2">
              <input id="ed-peak-min" class="control-input mono" type="number" step="500" value="${(p.cost_peak_idr || p.cost_normal_idr || [5000, 7000])[0]}">
              <input id="ed-peak-max" class="control-input mono" type="number" step="500" value="${(p.cost_peak_idr || p.cost_normal_idr || [5000, 7000])[1]}">
            </div>
          </div>
        </div>
      ` : ''}

      ${p.pricing_model === 'per_km_custom' ? `
        <div class="info-box">Rumus: <strong>Tarif Dasar + (${Number(edge.distance_km).toFixed(1)} km × Tarif/km)</strong>. Cocok untuk bensin motor/mobil atau estimasi rata-rata pribadi.</div>
        <div class="field-row-2">
          <div class="field-group">
            <label>Tarif Dasar Min/Max (IDR)</label>
            <div class="field-row-2">
              <input id="ed-base-min" class="control-input mono" type="number" step="500" value="${(p.base_fare_idr || [0, 0])[0]}">
              <input id="ed-base-max" class="control-input mono" type="number" step="500" value="${(p.base_fare_idr || [0, 0])[1]}">
            </div>
          </div>
          <div class="field-group">
            <label>Tarif per Km Min/Max (IDR)</label>
            <div class="field-row-2">
              <input id="ed-km-min" class="control-input mono" type="number" step="100" value="${(p.per_km_idr || [2000, 2500])[0]}">
              <input id="ed-km-max" class="control-input mono" type="number" step="100" value="${(p.per_km_idr || [2000, 2500])[1]}">
            </div>
          </div>
        </div>
      ` : ''}

      ${p.pricing_model === 'ride_hailing_zone' ? `
        <div class="field-group">
          <label>Zona Tarif GoRide / GoCar (Regulasi Indonesia)</label>
          <select id="ed-zone-preset" class="control-select">
            ${presetKeys.map((z) => `<option value="${z}" ${activePresetKey === z ? 'selected' : ''}>${zonePresets[z].label || z}</option>`).join('')}
          </select>
        </div>
        ${est ? `
          <div class="info-box">
            <strong>Rincian ${activePreset.label}</strong><br>
            Tarif minimal (0–${est.minKm} km): <span class="mono">${formatIdr(est.minFare[0])}–${formatIdr(est.minFare[1])}</span><br>
            Per km berikutnya: <span class="mono">${formatIdr(est.perKm[0])}–${formatIdr(est.perKm[1])}/km</span> · Jasa app: <span class="mono">${formatIdr(est.appFee[0])}–${formatIdr(est.appFee[1])}</span><br>
            <strong>Estimasi rute ini (${Number(edge.distance_km).toFixed(1)} km):</strong>
            Normal <span class="mono">${formatIdr(est.normLow)}–${formatIdr(est.normHigh)}</span> ·
            Peak+Hujan <span class="mono">${formatIdr(est.surgeLow)}–${formatIdr(est.surgeHigh)}</span>
          </div>
        ` : ''}
        <details class="section-disclosure" style="border-radius: var(--radius-sm);">
          <summary style="padding: 7px 10px; font-size: 11px;">Override Tarif Dasar / Per-Km (Opsional)</summary>
          <div class="disclosure-body" style="padding: 8px 10px;">
            <div class="field-row-2">
              <div class="field-group">
                <label>Min Fare Override (Min/Max)</label>
                <div class="field-row-2">
                  <input id="ed-ov-min0" class="control-input mono" type="number" placeholder="${est ? est.minFare[0] : ''}" value="${p.min_fare_idr ? p.min_fare_idr[0] : ''}">
                  <input id="ed-ov-min1" class="control-input mono" type="number" placeholder="${est ? est.minFare[1] : ''}" value="${p.min_fare_idr ? p.min_fare_idr[1] : ''}">
                </div>
              </div>
              <div class="field-group">
                <label>Per-Km Override (Min/Max)</label>
                <div class="field-row-2">
                  <input id="ed-ov-km0" class="control-input mono" type="number" placeholder="${est ? est.perKm[0] : ''}" value="${p.per_km_idr ? p.per_km_idr[0] : ''}">
                  <input id="ed-ov-km1" class="control-input mono" type="number" placeholder="${est ? est.perKm[1] : ''}" value="${p.per_km_idr ? p.per_km_idr[1] : ''}">
                </div>
              </div>
            </div>
          </div>
        </details>
      ` : ''}
    </div>
  `;

  edge.pricing = p;
  const bind = (id, fn, rerenderInspector = false) => {
    const el = body.querySelector(id);
    if (!el) return;
    el.addEventListener('change', () => {
      fn(el);
      markDirty();
      onRerenderGraph();
      if (rerenderInspector) renderEdgeInspector(body, edge, worldConfig, onRerenderGraph);
    });
  };

  bind('#ed-label', (el) => { edge.label = el.value; });
  bind('#ed-src', (el) => { edge.source = el.value; });
  bind('#ed-dst', (el) => { edge.target = el.value; });
  bind('#ed-dist', (el) => { edge.distance_km = Math.max(0.1, Number(el.value)); }, true);
  bind('#ed-bidir', (el) => { edge.bidirectional = el.checked; });
  bind('#ed-lock-dist', (el) => { edge._lockDistance = el.checked; });
  bind('#ed-mode', (el) => {
    edge.mode = el.value;
    if (el.value === 'walk') p.pricing_model = 'free';
    else if (el.value === 'gocar') { p.pricing_model = 'ride_hailing_zone'; p.zone_preset = 'gocar'; }
    else if (el.value === 'goride') { p.pricing_model = 'ride_hailing_zone'; p.zone_preset = p.zone_preset || 'goride_zona_1'; }
  }, true);
  bind('#ed-pmodel', (el) => {
    p.pricing_model = el.value;
    if (el.value === 'manual_range' && !p.cost_normal_idr) p.cost_normal_idr = [5000, 7000];
    if (el.value === 'per_km_custom' && !p.per_km_idr) p.per_km_idr = [2000, 2500];
    if (el.value === 'ride_hailing_zone' && !p.zone_preset) p.zone_preset = 'goride_zona_1';
  }, true);
  bind('#ed-norm-min', (el) => { p.cost_normal_idr = [Number(el.value), (p.cost_normal_idr || [5000, 7000])[1]]; });
  bind('#ed-norm-max', (el) => { p.cost_normal_idr = [(p.cost_normal_idr || [5000, 7000])[0], Number(el.value)]; });
  bind('#ed-peak-min', (el) => { p.cost_peak_idr = [Number(el.value), (p.cost_peak_idr || p.cost_normal_idr || [5000, 7000])[1]]; });
  bind('#ed-peak-max', (el) => { p.cost_peak_idr = [(p.cost_peak_idr || p.cost_normal_idr || [5000, 7000])[0], Number(el.value)]; });
  bind('#ed-base-min', (el) => { p.base_fare_idr = [Number(el.value), (p.base_fare_idr || [0, 0])[1]]; });
  bind('#ed-base-max', (el) => { p.base_fare_idr = [(p.base_fare_idr || [0, 0])[0], Number(el.value)]; });
  bind('#ed-km-min', (el) => { p.per_km_idr = [Number(el.value), (p.per_km_idr || [2000, 2500])[1]]; });
  bind('#ed-km-max', (el) => { p.per_km_idr = [(p.per_km_idr || [2000, 2500])[0], Number(el.value)]; });
  bind('#ed-zone-preset', (el) => { p.zone_preset = el.value; }, true);

  const bindOverridePair = (id0, id1, field, fallbackPair) => {
    const e0 = body.querySelector(id0);
    const e1 = body.querySelector(id1);
    if (!e0 || !e1) return;
    const handler = () => {
      if (e0.value === '' && e1.value === '') {
        p[field] = null;
      } else {
        p[field] = [
          e0.value !== '' ? Number(e0.value) : fallbackPair[0],
          e1.value !== '' ? Number(e1.value) : fallbackPair[1],
        ];
      }
      markDirty();
      renderEdgeInspector(body, edge, worldConfig, onRerenderGraph);
    };
    e0.addEventListener('change', handler);
    e1.addEventListener('change', handler);
  };
  if (est) {
    bindOverridePair('#ed-ov-min0', '#ed-ov-min1', 'min_fare_idr', est.minFare);
    bindOverridePair('#ed-ov-km0', '#ed-ov-km1', 'per_km_idr', est.perKm);
  }
}

export function initGraphStudioPage(root) {
  const svg = root.querySelector('#spatial-graph-svg');
  const srcSel = root.querySelector('#preview-src');
  const dstSel = root.querySelector('#preview-dst');
  const prefSel = root.querySelector('#preview-pref');
  const tariffSel = root.querySelector('#preview-tariff');
  const rainChk = root.querySelector('#preview-rain');
  const previewOut = root.querySelector('#route-preview-output');
  const walkPrefEl = root.querySelector('#walk-pref-select');
  const walkMaxEl = root.querySelector('#walk-max-km');
  const walkRainEl = root.querySelector('#walk-max-rain-km');
  const scaleSel = root.querySelector('#graph-scale-select');
  const autoDistChk = root.querySelector('#chk-auto-distance');
  const scaleBarLine = root.querySelector('#scale-bar-line');
  const connectBtn = root.querySelector('#btn-connect-mode');
  const modeBanner = root.querySelector('#graph-mode-banner');
  const inspTitle = root.querySelector('#inspector-title');
  const inspSub = root.querySelector('#inspector-subtitle');
  const inspBody = root.querySelector('#inspector-body');
  const delBtn = root.querySelector('#btn-delete-selected');

  let connectMode = false;
  let connectSourceId = null;
  let addingLocation = false;

  function currentPxPerKm() {
    return Number(scaleSel.value || 120);
  }

  function setConnectMode(active) {
    connectMode = active;
    connectSourceId = null;
    connectBtn.classList.toggle('is-active-tool', active);
    if (active) {
      modeBanner.hidden = false;
      modeBanner.textContent = 'Connect Mode aktif: Klik lokasi ASAL di peta, lalu klik lokasi TUJUAN untuk membuat rute otomatis.';
    } else {
      modeBanner.hidden = true;
    }
    redrawGraph();
  }

  function syncWalkControls() {
    const wp = appState.worldConfig?.mobility_and_climate?.walk_physics;
    if (!wp) return;
    walkPrefEl.value = wp.walk_preference || 'balanced';
    walkMaxEl.value = wp.max_walk_distance_km ?? '';
    walkRainEl.value = wp.max_walk_distance_rain_km ?? '';
  }

  function syncPreviewSelectors() {
    const nodes = appState.worldConfig?.world_graph?.locations || [];
    const curSrc = srcSel.value;
    const curDst = dstSel.value;
    const opts = nodes.map((n) => `<option value="${n.id}">${n.label || n.id}</option>`).join('');
    srcSel.innerHTML = opts;
    dstSel.innerHTML = opts;
    if (nodes.some((n) => n.id === curSrc)) srcSel.value = curSrc;
    else if (nodes.length > 0) srcSel.value = appState.worldConfig.world_graph.home_node_id;
    if (nodes.some((n) => n.id === curDst)) dstSel.value = curDst;
    else if (nodes.length > 1) dstSel.value = nodes.find((n) => n.id !== srcSel.value)?.id || nodes[0].id;
  }

  function renderInspector() {
    const wg = appState.worldConfig?.world_graph;
    if (!wg) return;

    if (addingLocation) {
      inspTitle.textContent = 'Tambah Lokasi Baru';
      inspSub.textContent = 'Pilih jenis tempat dan hubungkan ke graf';
      delBtn.hidden = true;
      renderAddLocationForm(
        inspBody,
        appState.worldConfig,
        currentPxPerKm(),
        (createdId) => {
          addingLocation = false;
          appState.selectedGraphItem = { type: 'node', id: createdId };
          markDirty();
          syncPreviewSelectors();
          redrawGraph();
          renderInspector();
          runPreview();
          showToast(`Lokasi '${createdId}' ditambahkan.`);
        },
        () => {
          addingLocation = false;
          renderInspector();
        },
      );
      return;
    }

    const sel = appState.selectedGraphItem;
    if (!sel) {
      inspTitle.textContent = 'Inspector';
      inspSub.textContent = 'Klik lokasi atau garis rute di peta untuk mengubah aturannya';
      inspBody.innerHTML =
        '<p class="panel-sub">Geser node di peta untuk memperbarui jarak antar-lokasi secara real-time, atau klik node/garis rute untuk mengedit mood, profil belajar, dan tarif transport.</p>';
      delBtn.hidden = true;
      return;
    }
    if (sel.type === 'node') {
      const node = wg.locations.find((n) => n.id === sel.id);
      if (!node) return;
      inspTitle.textContent = `Lokasi: ${node.label || node.id}`;
      inspSub.textContent = 'Atur suasana, efek mood/stress per step, kapabilitas, dan profil belajar';
      delBtn.hidden = node.id === wg.home_node_id;
      renderNodeInspector(inspBody, node, appState.worldConfig, redrawGraph);
    } else {
      const edge = wg.edges.find((e) => e.id === sel.id);
      if (!edge) return;
      inspTitle.textContent = `Rute: ${edge.label || edge.id}`;
      inspSub.textContent = 'Atur jarak, moda transportasi, dan skema tarif';
      delBtn.hidden = false;
      renderEdgeInspector(inspBody, edge, appState.worldConfig, () => {
        redrawGraph();
        runPreview();
      });
    }
  }

  function redrawGraph() {
    if (!appState.worldConfig) return;
    scaleBarLine.style.width = `${currentPxPerKm()}px`;
    renderSpatialGraph(svg, appState.worldConfig, {
      interactive: true,
      selectedItem: appState.selectedGraphItem,
      connectSourceId,
      highlightedEdgeIds: appState.previewedPath?.edge_ids || [],
      highlightMode: appState.previewedPath?.preference || 'fastest',
      pxPerKm: currentPxPerKm(),
      autoSyncDistance: autoDistChk.checked,
      onSelect: (item) => {
        if (connectMode && item.type === 'node') {
          if (!connectSourceId) {
            connectSourceId = item.id;
            modeBanner.textContent = `Dari '${item.id}' -> sekarang klik lokasi TUJUAN di peta.`;
            redrawGraph();
          } else if (connectSourceId !== item.id) {
            const wg = appState.worldConfig.world_graph;
            const srcNode = wg.locations.find((n) => n.id === connectSourceId);
            const dstNode = wg.locations.find((n) => n.id === item.id);
            const distKm = computePixelDistanceKm(srcNode.ui_position, dstNode.ui_position, currentPxPerKm());
            const newId = `route_${connectSourceId}_${item.id}_${wg.edges.length + 1}`;
            wg.edges.push({
              id: newId,
              label: `${srcNode.label || srcNode.id} <-> ${dstNode.label || dstNode.id}`,
              source: connectSourceId,
              target: item.id,
              bidirectional: true,
              mode: distKm <= 1.2 ? 'walk' : 'goride',
              distance_km: distKm,
              pricing: distKm <= 1.2
                ? { pricing_model: 'free' }
                : { pricing_model: 'ride_hailing_zone', zone_preset: 'goride_zona_1' },
            });
            appState.selectedGraphItem = { type: 'edge', id: newId };
            addingLocation = false;
            setConnectMode(false);
            markDirty();
            renderInspector();
            runPreview();
            showToast(`Rute dibuat (${distKm.toFixed(1)} km). Pilih tarif di Inspector.`);
          }
          return;
        }

        addingLocation = false;
        appState.selectedGraphItem = item;
        redrawGraph();
        renderInspector();
      },
      onNodeMove: (_nodeId, _x, _y, committed) => {
        if (committed) {
          markDirty();
          redrawGraph();
          if (appState.selectedGraphItem?.type === 'edge') renderInspector();
          runPreview();
        }
      },
    });
  }

  async function runPreview() {
    if (!appState.worldConfig || !srcSel.value || !dstSel.value) return;
    try {
      const res = await api.previewRoute({
        config: appState.worldConfig,
        source_node_id: srcSel.value,
        target_node_id: dstSel.value,
        preference: prefSel.value,
        ojol_tariff: tariffSel.value,
        is_raining: rainChk.checked,
      });
      appState.previewedPath = res;
      previewOut.innerHTML =
        `<strong>Jalur:</strong> ${res.edge_ids.join(' -> ') || 'di tempat'}<br>` +
        `Total <strong>${res.total_distance_km.toFixed(1)} km</strong> (jalan kaki ${res.walk_distance_km.toFixed(1)} km) · ` +
        `Ongkos <strong>${formatIdr(res.min_fare_idr)}–${formatIdr(res.max_fare_idr)}</strong> · ` +
        `Energi -${res.total_energy_cost.toFixed(1)} · Mood ${res.total_mood_delta >= 0 ? '+' : ''}${res.total_mood_delta.toFixed(1)}`;
      redrawGraph();
    } catch (err) {
      appState.previewedPath = null;
      previewOut.textContent = `Rute tidak tersedia: ${err.message}`;
      redrawGraph();
    }
  }

  walkPrefEl.addEventListener('change', () => {
    appState.worldConfig.mobility_and_climate.walk_physics.walk_preference = walkPrefEl.value;
    markDirty();
    runPreview();
  });
  walkMaxEl.addEventListener('change', () => {
    appState.worldConfig.mobility_and_climate.walk_physics.max_walk_distance_km = numOrNull(walkMaxEl.value);
    markDirty();
    runPreview();
  });
  walkRainEl.addEventListener('change', () => {
    appState.worldConfig.mobility_and_climate.walk_physics.max_walk_distance_rain_km = numOrNull(walkRainEl.value);
    markDirty();
    runPreview();
  });

  scaleSel.addEventListener('change', redrawGraph);
  autoDistChk.addEventListener('change', redrawGraph);
  connectBtn.addEventListener('click', () => setConnectMode(!connectMode));

  [srcSel, dstSel, prefSel, tariffSel, rainChk].forEach((el) => el.addEventListener('change', runPreview));
  root.querySelector('#btn-preview-route').addEventListener('click', runPreview);

  root.querySelector('#btn-add-node').addEventListener('click', () => {
    setConnectMode(false);
    addingLocation = true;
    renderInspector();
  });

  delBtn.addEventListener('click', () => {
    const sel = appState.selectedGraphItem;
    const wg = appState.worldConfig.world_graph;
    if (!sel) return;
    if (sel.type === 'node') {
      wg.locations = wg.locations.filter((n) => n.id !== sel.id);
      wg.edges = wg.edges.filter((e) => e.source !== sel.id && e.target !== sel.id);
    } else {
      wg.edges = wg.edges.filter((e) => e.id !== sel.id);
    }
    appState.selectedGraphItem = null;
    markDirty();
    syncPreviewSelectors();
    redrawGraph();
    renderInspector();
    runPreview();
  });

  return {
    refresh() {
      addingLocation = false;
      setConnectMode(false);
      syncWalkControls();
      syncPreviewSelectors();
      redrawGraph();
      renderInspector();
      runPreview();
    },
  };
}
