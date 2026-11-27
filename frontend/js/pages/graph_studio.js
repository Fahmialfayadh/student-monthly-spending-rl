import { api } from '../api.js';
import { computeWorldDistanceKm, createSpatialGraphViewport } from '../graph_canvas.js';
import { appState, markDirty, showToast } from '../state.js';
import { escapeHtml, formatIdr, nextUniqueId, syncActivitiesWithGraph } from '../utils.js';

function numOrNull(val) {
  if (val === '' || val === null || val === undefined) return null;
  const n = Number(val);
  return Number.isFinite(n) ? n : null;
}

function clampPair(low, high) {
  const a = Math.max(0, Number(low || 0));
  const b = Math.max(a, Number(high || 0));
  return [a, b];
}

function buildCleanPricing(model, zonePreset = 'goride_zona_1') {
  if (model === 'free') return { pricing_model: 'free' };
  if (model === 'manual_range') {
    return {
      pricing_model: 'manual_range',
      cost_normal_idr: [5000, 7000],
      cost_peak_idr: [6000, 8000],
      rain_surge_multiplier: 1.2,
    };
  }
  if (model === 'per_km_custom') {
    return {
      pricing_model: 'per_km_custom',
      base_fare_idr: [0, 0],
      per_km_idr: [1500, 2200],
      peak_multiplier: 1.0,
      rain_surge_multiplier: 1.0,
    };
  }
  return { pricing_model: 'ride_hailing_zone', zone_preset: zonePreset };
}

function applyTransportModeDefaults(edge, mode, zonePreset = null) {
  edge.mode = mode;
  if (mode === 'walk') {
    edge.pricing = buildCleanPricing('free');
  } else if (mode === 'goride') {
    edge.pricing = buildCleanPricing('ride_hailing_zone', zonePreset && zonePreset !== 'gocar' ? zonePreset : 'goride_zona_1');
  } else if (mode === 'gocar') {
    edge.pricing = buildCleanPricing('ride_hailing_zone', 'gocar');
  } else if (mode === 'ojol') {
    edge.pricing = buildCleanPricing('ride_hailing_zone', zonePreset || 'goride_zona_1');
  } else if (mode === 'public_transit') {
    edge.pricing = buildCleanPricing('manual_range');
  } else if (mode === 'personal') {
    edge.pricing = buildCleanPricing('per_km_custom');
    edge.pricing.per_km_idr = [600, 1000];
  } else {
    edge.pricing = buildCleanPricing('per_km_custom');
  }
}

function applyGlobalProfileToEdge(edge, profileValue) {
  const [mode, detail] = (profileValue || 'goride:goride_zona_1').split(':');
  applyTransportModeDefaults(edge, mode, detail === 'custom' || detail === 'manual' ? null : detail);
}

function inferGlobalProfileValue(worldConfig) {
  const edges = worldConfig?.world_graph?.edges || [];
  const motorized = edges.find((e) => e.mode !== 'walk');
  if (!motorized) return 'goride:goride_zona_1';
  if (motorized.mode === 'gocar' || motorized.pricing?.zone_preset === 'gocar') return 'gocar:gocar';
  if (motorized.pricing?.pricing_model === 'ride_hailing_zone') {
    return `goride:${motorized.pricing.zone_preset || 'goride_zona_1'}`;
  }
  if (motorized.mode === 'personal') return 'personal:custom';
  if (motorized.mode === 'public_transit') return 'public_transit:manual';
  return 'goride:goride_zona_1';
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
  return { minKm, minFare, perKm, appFee, normLow, normHigh, surgeLow, surgeHigh };
}

const LOCATION_ARCHETYPES = {
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

function renderAddLocationForm(body, worldConfig, centerPos, globalProfileVal, onCreated, onCancel) {
  const wg = worldConfig.world_graph;
  const existingNodes = wg.locations || [];
  body.innerHTML = `
    <div class="stack-col" style="gap: 10px;">
      <p class="panel-sub">Beri nama dan pilih jenis lokasi. Secara default lokasi baru otomatis terhubung ke semua tempat di peta sesuai jaraknya.</p>
      <div class="field-row-2">
        <div class="field-group">
          <label>Nama Tempat</label>
          <input id="new-loc-label" class="control-input" value="Kafe Belajar Baru">
        </div>
        <div class="field-group">
          <label>Jenis Tempat</label>
          <select id="new-loc-type" class="control-select">
            <option value="cafe">Kafe / Working Space</option>
            <option value="study_hub">Perpustakaan / Study Hub</option>
            <option value="campus">Gedung Kuliah / Kampus</option>
            <option value="home">Kos / Tempat Tinggal</option>
            <option value="custom">Custom</option>
          </select>
        </div>
      </div>

      <div class="field-group">
        <label>Konektivitas Rute Otomatis</label>
        <select id="new-loc-mesh-mode" class="control-select">
          <option value="all">Terhubung otomatis ke SEMUA lokasi (${existingNodes.length} tempat)</option>
          <option value="home">Hanya terhubung ke Home (${escapeHtml(wg.home_node_id)})</option>
          <option value="pick">Pilih lokasi tujuan tertentu saja</option>
        </select>
      </div>

      <div id="new-loc-pick-box" style="display: none; flex-wrap: wrap; gap: 8px;">
        ${existingNodes.map((n) => `
          <label class="checkbox-row">
            <input class="new-loc-target" type="checkbox" value="${escapeHtml(n.id)}" checked>
            <span>${escapeHtml(n.label || n.id)}</span>
          </label>
        `).join('')}
      </div>

      <div class="hint-inline">
        Tarif rute otomatis mengikuti Profil Mobilitas Utama yang sedang aktif.
      </div>

      <div style="display: flex; gap: 8px; justify-content: flex-end;">
        <button id="btn-cancel-new-loc" class="btn btn-sm" type="button">Batal</button>
        <button id="btn-confirm-new-loc" class="btn btn-sm btn-primary" type="button">Tambah ke Peta</button>
      </div>
    </div>
  `;

  const typeSel = body.querySelector('#new-loc-type');
  const labelInput = body.querySelector('#new-loc-label');
  const meshSel = body.querySelector('#new-loc-mesh-mode');
  const pickBox = body.querySelector('#new-loc-pick-box');

  typeSel.addEventListener('change', () => {
    const tpl = LOCATION_ARCHETYPES[typeSel.value];
    if (tpl) labelInput.value = tpl.label;
  });
  meshSel.addEventListener('change', () => {
    pickBox.style.display = meshSel.value === 'pick' ? 'flex' : 'none';
  });
  body.querySelector('#btn-cancel-new-loc').addEventListener('click', onCancel);

  body.querySelector('#btn-confirm-new-loc').addEventListener('click', () => {
    const labelText = labelInput.value.trim() || 'Lokasi Baru';
    const slugBase = labelText.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || typeSel.value;
    const rawId = wg.locations.some((n) => n.id === slugBase)
      ? nextUniqueId(slugBase, wg.locations.map((n) => n.id))
      : slugBase;

    const tpl = structuredClone(LOCATION_ARCHETYPES[typeSel.value] || LOCATION_ARCHETYPES.custom);
    const idx = wg.locations.length;
    const angle = (idx * 1.15) % (Math.PI * 2);
    const pos = {
      x: Math.round((centerPos?.x ?? 380) + Math.cos(angle) * 90),
      y: Math.round((centerPos?.y ?? 240) + Math.sin(angle) * 75),
    };
    const newNode = {
      id: rawId,
      ...tpl,
      label: labelText,
      food_options: [],
      ui_position: pos,
    };
    wg.locations.push(newNode);

    const targets = meshSel.value === 'all'
      ? existingNodes
      : meshSel.value === 'home'
        ? existingNodes.filter((n) => n.id === wg.home_node_id)
        : existingNodes.filter((n) => Array.from(body.querySelectorAll('.new-loc-target:checked')).some((c) => c.value === n.id));

    const wp = worldConfig?.mobility_and_climate?.walk_physics || {};
    const allowWalk = wp.walk_preference !== 'avoid_walk';
    const maxWalk = wp.max_walk_distance_km;
    targets.forEach((other) => {
      const distKm = computeWorldDistanceKm(other.ui_position, pos);
      const newEdge = {
        id: nextUniqueId(`route_${other.id}_${rawId}`, wg.edges.map((e) => e.id)),
        label: `${other.label || other.id} ↔ ${newNode.label}`,
        source: other.id,
        target: rawId,
        bidirectional: true,
        mode: 'goride',
        distance_km: distKm,
        pricing: { pricing_model: 'free' },
      };
      applyGlobalProfileToEdge(newEdge, globalProfileVal);
      wg.edges.push(newEdge);

      if (allowWalk && (maxWalk === null || maxWalk === undefined || (maxWalk > 0 && distKm <= maxWalk))) {
        wg.edges.push({
          id: nextUniqueId(`walk_${other.id}_${rawId}`, wg.edges.map((e) => e.id)),
          label: `Jalan Kaki (${other.label || other.id} ↔ ${newNode.label})`,
          source: other.id,
          target: rawId,
          bidirectional: true,
          mode: 'walk',
          distance_km: distKm,
          pricing: { pricing_model: 'free' },
        });
      }
    });

    syncActivitiesWithGraph(worldConfig);
    onCreated(rawId);
  });
}

function renderNodeInspector(body, node, worldConfig, onUpdated, editorState = {}) {
  // Preserve open state of disclosure sections so editing fields inside never snaps them shut
  const openSections = new Set(editorState.openSections || []);
  body.querySelectorAll('details[data-sec]').forEach((det) => {
    if (det.open) openSections.add(det.dataset.sec);
    else openSections.delete(det.dataset.sec);
  });
  const editingFoodId = editorState.editingFoodId || null;
  const showFoodForm = Boolean(editorState.showFoodForm || editingFoodId);
  const editFoodSpec = editingFoodId ? (worldConfig.external_foods || {})[editingFoodId] : null;

  const sp = node.study_profile || {};
  const foodEntries = Object.entries(worldConfig.external_foods || {});
  const selectedFoods = new Set(node.food_options || []);
  const isHome = worldConfig.world_graph.home_node_id === node.id;
  const fmtSign = (v) => (Number(v || 0) > 0 ? `+${v}` : `${v ?? 0}`);

  body.innerHTML = `
    <div class="stack-col" style="gap: 11px;">
      <div class="field-row-2">
        <div class="field-group">
          <label>Nama Lokasi</label>
          <input id="nd-label" class="control-input" value="${escapeHtml(node.label || '')}">
        </div>
        <div class="field-group">
          <label>Zona Warna</label>
          <select id="nd-zone" class="control-select">
            ${[
              ['home', 'Home (Tempat Tinggal)'],
              ['campus', 'Kampus / Study Hub'],
              ['third_place', 'Kafe / Santai'],
              ['commercial', 'Komersial'],
              ['custom', 'Custom'],
            ].map(([val, lbl]) => `<option value="${val}" ${node.zone === val ? 'selected' : ''}>${lbl}</option>`).join('')}
          </select>
        </div>
      </div>

      <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
        <span class="hint-inline code-id">id: ${escapeHtml(node.id)}</span>
        ${!isHome && node.can_sleep
          ? '<button id="btn-set-home" class="btn btn-sm" type="button">Jadikan Home Utama</button>'
          : `<span class="hint-inline">${isHome ? 'Home utama (awal & pulang malam)' : ''}</span>`
        }
      </div>

      <div class="field-group" style="padding-top: 8px; border-top: 1px solid var(--border-subtle);">
        <label>Kegiatan &amp; Aturan di Tempat Ini</label>
        <div class="field-row-2" style="gap: 7px;">
          <label class="checkbox-row">
            <input id="nd-sleep" type="checkbox" ${node.can_sleep ? 'checked' : ''} ${isHome ? 'disabled' : ''}>
            <span>Bisa Tidur${isHome ? ' (wajib)' : ''}</span>
          </label>
          <label class="checkbox-row">
            <input id="nd-kitchen" type="checkbox" ${node.has_kitchen ? 'checked' : ''}>
            <span>Ada Dapur (Masak)</span>
          </label>
          <label class="checkbox-row">
            <input id="nd-grocery" type="checkbox" ${node.has_grocery ? 'checked' : ''}>
            <span>Bisa Belanja Bahan</span>
          </label>
          <label class="checkbox-row">
            <input id="nd-class" type="checkbox" ${node.is_class_venue ? 'checked' : ''}>
            <span>Lokasi Kuliah</span>
          </label>
          <label class="checkbox-row">
            <input id="nd-study" type="checkbox" ${node.can_study ? 'checked' : ''}>
            <span>Bisa Dipakai Belajar</span>
          </label>
          <label class="checkbox-row">
            <input id="nd-curfew" type="checkbox" ${node.closes_at_night ? 'checked' : ''}>
            <span>Tutup Malam (Curfew)</span>
          </label>
        </div>
      </div>

      <div class="field-group" style="padding-top: 8px; border-top: 1px solid var(--border-subtle);">
        <div style="display: flex; align-items: center; justify-content: space-between;">
          <label>Makanan yang Tersedia di Sini</label>
          <button id="btn-inline-add-food" class="btn btn-sm" type="button">+ Tambah Makanan</button>
        </div>
        <div id="nd-inline-food-box" ${showFoodForm ? '' : 'hidden'} class="stack-col" style="padding: 10px; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); background: var(--bg-subtle); gap: 8px; margin-top: 4px;">
          <div class="field-label" style="color: var(--ink-primary);">
            ${editFoodSpec ? `Edit Makanan: ${escapeHtml(editFoodSpec.label || editingFoodId)}` : 'Tambah Opsi Makanan Baru'}
          </div>
          <div class="field-row-2">
            <div class="field-group">
              <label for="in-food-label">Nama Warung / Menu</label>
              <input id="in-food-label" class="control-input" placeholder="mis. Kantin / Foodcourt" value="${escapeHtml(editFoodSpec?.label || '')}">
            </div>
            <div class="field-group">
              <label for="in-food-tier">Kategori</label>
              <select id="in-food-tier" class="control-select">
                <option value="budget" ${(editFoodSpec?.tier || 'budget') === 'budget' ? 'selected' : ''}>Hemat (budget)</option>
                <option value="varied" ${editFoodSpec?.tier === 'varied' ? 'selected' : ''}>Variasi (varied)</option>
              </select>
            </div>
          </div>
          <div class="field-row-2">
            <div class="field-group">
              <label for="in-food-cost">Harga (IDR)</label>
              <input id="in-food-cost" class="control-input num" type="number" min="0" step="500" value="${editFoodSpec?.cost_idr ?? 15000}">
            </div>
            <div class="field-group">
              <label for="in-food-satiety" class="tooltip-trigger" data-tooltip="Seberapa mengenyangkan makanan ini (1–100, semakin tinggi semakin kenyang).">Tingkat Kenyang (1–100)</label>
              <input id="in-food-satiety" class="control-input num" type="number" min="1" max="100" step="1" value="${Math.abs(Number(editFoodSpec?.hunger ?? 55))}">
            </div>
          </div>
          <div class="field-row-2">
            <div class="field-group">
              <label for="in-food-energy" class="tooltip-trigger" data-tooltip="Tambahan energi setelah makan menu ini.">Energi (+)</label>
              <input id="in-food-energy" class="control-input num" type="number" step="1" value="${editFoodSpec?.energy ?? 18}">
            </div>
            <div class="field-group">
              <label for="in-food-mood" class="tooltip-trigger" data-tooltip="Perubahan mood setelah makan (positif = bikin senang).">Efek Mood (+/-)</label>
              <input id="in-food-mood" class="control-input num" type="number" step="0.5" value="${editFoodSpec?.mood ?? 4}">
            </div>
          </div>
          <div style="display: flex; justify-content: flex-end; gap: 6px;">
            <button id="btn-inline-food-cancel" class="btn btn-sm" type="button">Batal</button>
            <button id="btn-inline-food-save" class="btn btn-sm btn-primary" type="button">${editFoodSpec ? 'Simpan Perubahan' : 'Tambahkan'}</button>
          </div>
        </div>
        <div class="stack-col" style="gap: 5px; margin-top: 6px;">
          ${foodEntries.length === 0
            ? '<span class="hint-inline">Belum ada makanan. Klik + Tambah Makanan di atas.</span>'
            : foodEntries.map(([fk, fspec]) => `
              <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 5px 8px; border: 1px solid var(--border-subtle); border-radius: var(--radius-xs); background: var(--bg-surface);">
                <label class="checkbox-row" style="min-width: 0;">
                  <input class="nd-food-opt" type="checkbox" value="${escapeHtml(fk)}" ${selectedFoods.has(fk) ? 'checked' : ''}>
                  <span style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(fspec.label || fk)}</span>
                </label>
                <div style="display: inline-flex; align-items: center; gap: 8px; flex-shrink: 0;">
                  <span class="hint-inline num">${formatIdr(fspec.cost_idr)} · Kenyang ${Math.abs(Number(fspec.hunger || 0))}</span>
                  <button class="inline-link-btn btn-edit-food-item" data-fid="${escapeHtml(fk)}" type="button">Edit</button>
                </div>
              </div>
            `).join('')
          }
        </div>
      </div>

      <div class="stack-col" style="gap: 8px; padding-top: 8px; border-top: 1px solid var(--border-subtle);">
        <div class="field-label">Dampak Suasana Lokasi (Skala Biometrik 0–100)</div>
        <div class="stat-pills">
          <div class="stat-pill">
            <div class="stat-pill-label">Mood / Step</div>
            <div id="pill-mood" class="stat-pill-value">${fmtSign(node.ambient_mood_delta)}</div>
          </div>
          <div class="stat-pill">
            <div class="stat-pill-label">Stres / Step</div>
            <div id="pill-stress" class="stat-pill-value">${fmtSign(node.ambient_stress_delta)}</div>
          </div>
          <div class="stat-pill">
            <div class="stat-pill-label">Jenuh / Step</div>
            <div id="pill-craving" class="stat-pill-value">+${node.craving_gain_per_step ?? 0}</div>
          </div>
        </div>
        <div class="field-row-3">
          <div class="field-group">
            <label class="tooltip-trigger" data-tooltip="Konstan ditambahkan ke mood tiap 1 step berada di sini.">Ubah Mood</label>
            <input id="nd-mood" class="control-input num" type="number" step="0.5" value="${node.ambient_mood_delta ?? 0}">
          </div>
          <div class="field-group">
            <label class="tooltip-trigger" data-tooltip="Konstan ditambahkan ke stres akademik tiap 1 step berada di sini.">Ubah Stres</label>
            <input id="nd-stress" class="control-input num" type="number" step="0.5" value="${node.ambient_stress_delta ?? 0}">
          </div>
          <div class="field-group">
            <label class="tooltip-trigger" data-tooltip="Kecepatan muncul rasa bosan/jenuh (0–100) per step.">Laju Jenuh</label>
            <input id="nd-craving" class="control-input num" type="number" step="0.5" min="0" value="${node.craving_gain_per_step ?? 3}">
          </div>
        </div>
      </div>

      ${node.can_study ? `
        <details class="section-disclosure" data-sec="study" ${openSections.has('study') ? 'open' : ''} style="border-radius: var(--radius-sm);">
          <summary style="padding: 8px 10px; font-size: 11.5px;">Profil Belajar di Lokasi Ini</summary>
          <div class="disclosure-body" style="padding: 9px 10px;">
            <div class="field-row-3">
              <div class="field-group">
                <label>Reduksi Stres</label>
                <input id="nd-sp-stress" class="control-input num" type="number" min="0" step="1" value="${sp.stress_reduction ?? 12}">
              </div>
              <div class="field-group">
                <label>Biaya Energi</label>
                <input id="nd-sp-energy" class="control-input num" type="number" min="0" step="1" value="${sp.energy_cost ?? 6}">
              </div>
              <div class="field-group">
                <label>Delta Mood</label>
                <input id="nd-sp-mood" class="control-input num" type="number" step="1" value="${sp.mood_delta ?? 0}">
              </div>
            </div>
            <div class="field-row-2">
              <label class="checkbox-row">
                <input id="nd-sp-distract" type="checkbox" ${sp.affected_by_craving_distraction ? 'checked' : ''}>
                <span>Kena Distraksi Jenuh</span>
              </label>
              <label class="checkbox-row">
                <input id="nd-sp-reset" type="checkbox" ${sp.craving_reset ? 'checked' : ''}>
                <span>Reset Jenuh (Setara Kafe)</span>
              </label>
            </div>
            <div class="field-row-2">
              <div class="field-group">
                <label>Reduksi Jenuh</label>
                <input id="nd-sp-relief" class="control-input num" type="number" min="0" step="1" value="${sp.craving_relief ?? 0}">
              </div>
              <div class="field-group">
                <label>Durasi Buff (Step)</label>
                <input id="nd-sp-buff" class="control-input num" type="number" min="0" step="1" value="${sp.buff_duration_steps ?? 0}">
              </div>
            </div>
          </div>
        </details>
      ` : ''}

      ${node.zone === 'third_place' || (node.leisure_mood_gain ?? 0) > 0 ? `
        <details class="section-disclosure" data-sec="leisure" ${openSections.has('leisure') ? 'open' : ''} style="border-radius: var(--radius-sm);">
          <summary style="padding: 8px 10px; font-size: 11.5px;">Biaya &amp; Efek Sesi Kafe / Santai</summary>
          <div class="disclosure-body" style="padding: 9px 10px;">
            <div class="field-row-3">
              <div class="field-group">
                <label>Biaya Min (IDR)</label>
                <input id="nd-leis-min" class="control-input num" type="number" min="0" step="1000" value="${(node.leisure_spend_idr || [20000, 35000])[0]}">
              </div>
              <div class="field-group">
                <label>Biaya Max (IDR)</label>
                <input id="nd-leis-max" class="control-input num" type="number" min="0" step="1000" value="${(node.leisure_spend_idr || [20000, 35000])[1]}">
              </div>
              <div class="field-group">
                <label>Mood Gain (+)</label>
                <input id="nd-leis-mood" class="control-input num" type="number" min="0" step="1" value="${node.leisure_mood_gain ?? 20}">
              </div>
            </div>
          </div>
        </details>
      ` : ''}
    </div>
  `;

  const updatePills = () => {
    const pm = body.querySelector('#pill-mood');
    const ps = body.querySelector('#pill-stress');
    const pc = body.querySelector('#pill-craving');
    if (pm) pm.textContent = fmtSign(node.ambient_mood_delta);
    if (ps) ps.textContent = fmtSign(node.ambient_stress_delta);
    if (pc) pc.textContent = `+${node.craving_gain_per_step ?? 0}`;
  };

  const rerenderKeepOpen = (nextEditorState = {}) => {
    const curOpen = [];
    body.querySelectorAll('details[data-sec]').forEach((det) => {
      if (det.open) curOpen.push(det.dataset.sec);
    });
    renderNodeInspector(body, node, worldConfig, onUpdated, { openSections: curOpen, ...nextEditorState });
  };

  const bind = (id, fn, rerenderInspector = false, syncActs = false) => {
    const el = body.querySelector(id);
    if (!el) return;
    el.addEventListener('change', () => {
      fn(el);
      if (syncActs) syncActivitiesWithGraph(worldConfig);
      markDirty();
      updatePills();
      onUpdated();
      if (rerenderInspector) rerenderKeepOpen();
    });
  };

  bind('#nd-label', (el) => { node.label = el.value; });
  bind('#nd-zone', (el) => {
    node.zone = el.value;
    if (el.value === 'third_place') {
      if (!node.leisure_spend_idr) node.leisure_spend_idr = [20000, 35000];
      if (!node.leisure_mood_gain) node.leisure_mood_gain = 22;
      if (!node.leisure_stress_reduction) node.leisure_stress_reduction = 22;
      if (!node.leisure_energy_cost) node.leisure_energy_cost = 5;
      if (!node.leisure_buff_duration_steps) node.leisure_buff_duration_steps = 8;
    }
  }, true, true);
  bind('#nd-mood', (el) => { node.ambient_mood_delta = Number(el.value); });
  bind('#nd-stress', (el) => { node.ambient_stress_delta = Number(el.value); });
  bind('#nd-craving', (el) => { node.craving_gain_per_step = Math.max(0, Number(el.value)); });
  bind('#nd-sleep', (el) => { node.can_sleep = el.checked; }, true, true);
  bind('#nd-kitchen', (el) => { node.has_kitchen = el.checked; });
  bind('#nd-grocery', (el) => { node.has_grocery = el.checked; }, false, true);
  bind('#nd-class', (el) => { node.is_class_venue = el.checked; }, false, true);
  bind('#nd-study', (el) => { node.can_study = el.checked; }, true, true);
  bind('#nd-curfew', (el) => { node.closes_at_night = el.checked; }, false, true);

  node.study_profile = sp;
  bind('#nd-sp-stress', (el) => { sp.stress_reduction = Math.max(0, Number(el.value)); });
  bind('#nd-sp-energy', (el) => { sp.energy_cost = Math.max(0, Number(el.value)); });
  bind('#nd-sp-mood', (el) => { sp.mood_delta = Number(el.value); });
  bind('#nd-sp-distract', (el) => { sp.affected_by_craving_distraction = el.checked; });
  bind('#nd-sp-reset', (el) => { sp.craving_reset = el.checked; }, false, true);
  bind('#nd-sp-relief', (el) => { sp.craving_relief = Math.max(0, Number(el.value)); }, false, true);
  bind('#nd-sp-buff', (el) => { sp.buff_duration_steps = Math.max(0, Number(el.value)); });

  // Clamp min/max in-place WITHOUT rerendering inspector so <details> never collapses!
  bind('#nd-leis-min', (el) => {
    const cur = node.leisure_spend_idr || [20000, 35000];
    node.leisure_spend_idr = clampPair(el.value, cur[1]);
    el.value = String(node.leisure_spend_idr[0]);
    const maxEl = body.querySelector('#nd-leis-max');
    if (maxEl) maxEl.value = String(node.leisure_spend_idr[1]);
  });
  bind('#nd-leis-max', (el) => {
    const cur = node.leisure_spend_idr || [20000, 35000];
    node.leisure_spend_idr = clampPair(cur[0], el.value);
    const minEl = body.querySelector('#nd-leis-min');
    if (minEl) minEl.value = String(node.leisure_spend_idr[0]);
    el.value = String(node.leisure_spend_idr[1]);
  });
  bind('#nd-leis-mood', (el) => { node.leisure_mood_gain = Math.max(0, Number(el.value)); });

  const setHomeBtn = body.querySelector('#btn-set-home');
  if (setHomeBtn) {
    setHomeBtn.addEventListener('click', () => {
      worldConfig.world_graph.home_node_id = node.id;
      syncActivitiesWithGraph(worldConfig);
      markDirty();
      onUpdated();
      rerenderKeepOpen();
      showToast(`Home utama dipindahkan ke '${node.label || node.id}'.`);
    });
  }

  // Inline Add & Edit Food right inside Node Inspector
  body.querySelector('#btn-inline-add-food').addEventListener('click', () => {
    rerenderKeepOpen({ showFoodForm: !showFoodForm || Boolean(editingFoodId), editingFoodId: null });
  });
  body.querySelector('#btn-inline-food-cancel').addEventListener('click', () => {
    rerenderKeepOpen({ showFoodForm: false, editingFoodId: null });
  });
  body.querySelectorAll('.btn-edit-food-item').forEach((btn) => {
    btn.addEventListener('click', () => {
      rerenderKeepOpen({ showFoodForm: true, editingFoodId: btn.dataset.fid });
    });
  });

  body.querySelector('#btn-inline-food-save').addEventListener('click', () => {
    const lbl = body.querySelector('#in-food-label').value.trim();
    if (!lbl) {
      showToast('Masukkan nama warung atau menu makanan.');
      body.querySelector('#in-food-label').focus();
      return;
    }
    const foods = worldConfig.external_foods || {};
    const sat = Math.max(1, Math.min(100, Number(body.querySelector('#in-food-satiety').value || 55)));
    const payload = {
      label: lbl,
      tier: body.querySelector('#in-food-tier').value,
      cost_idr: Math.max(0, Number(body.querySelector('#in-food-cost').value || 15000)),
      hunger: -sat,
      energy: Number(body.querySelector('#in-food-energy').value || 18),
      energy_cost: editFoodSpec?.energy_cost ?? 2,
      mood: Number(body.querySelector('#in-food-mood').value || 4),
    };

    if (editingFoodId && foods[editingFoodId]) {
      foods[editingFoodId] = { ...foods[editingFoodId], ...payload };
      markDirty();
      rerenderKeepOpen({ showFoodForm: false, editingFoodId: null });
      showToast(`Makanan '${lbl}' diperbarui.`);
    } else {
      const fid = nextUniqueId('makanan', Object.keys(foods));
      foods[fid] = { id: fid, ...payload };
      worldConfig.external_foods = foods;
      node.food_options = [...(node.food_options || []), fid];
      markDirty();
      rerenderKeepOpen({ showFoodForm: false, editingFoodId: null });
      showToast(`Makanan '${lbl}' ditambahkan ke ${node.label || node.id}.`);
    }
  });

  body.querySelectorAll('.nd-food-opt').forEach((cb) => {
    cb.addEventListener('change', () => {
      node.food_options = Array.from(body.querySelectorAll('.nd-food-opt:checked')).map((c) => c.value);
      markDirty();
    });
  });
}

function renderEdgeInspector(body, edge, worldConfig, onUpdated) {
  const nodes = worldConfig.world_graph.locations || [];
  const srcNode = nodes.find((n) => n.id === edge.source);
  const dstNode = nodes.find((n) => n.id === edge.target);
  const mapDistKm = srcNode && dstNode ? computeWorldDistanceKm(srcNode.ui_position, dstNode.ui_position) : null;
  const differsFromMap = mapDistKm !== null && Math.abs(Number(edge.distance_km || 0) - mapDistKm) >= 0.15;

  const p = edge.pricing || { pricing_model: 'free' };
  const zonePresets = appState.schema?.ride_hailing_zone_presets || {};
  const isWalk = edge.mode === 'walk';
  const isRideHailingMode = ['goride', 'gocar', 'ojol'].includes(edge.mode);
  const availableModels = isWalk
    ? [['free', 'Gratis (Rp 0)']]
    : isRideHailingMode
      ? [
          ['ride_hailing_zone', 'Preset GoRide / GoCar Zona'],
          ['manual_range', 'Rentang Harga Manual (A → B)'],
          ['per_km_custom', 'Tarif Per Km Kustom'],
          ['free', 'Gratis (Rp 0)'],
        ]
      : [
          ['per_km_custom', 'Tarif Per Km Kustom'],
          ['manual_range', 'Rentang Harga Manual (A → B)'],
          ['free', 'Gratis (Rp 0)'],
        ];

  const allowedPresets = edge.mode === 'gocar'
    ? Object.keys(zonePresets).filter((k) => k === 'gocar')
    : edge.mode === 'goride'
      ? Object.keys(zonePresets).filter((k) => k !== 'gocar')
      : Object.keys(zonePresets);
  const activePresetKey = allowedPresets.includes(p.zone_preset) ? p.zone_preset : (allowedPresets[0] || 'goride_zona_1');
  const activePreset = zonePresets[activePresetKey];
  const est = p.pricing_model === 'ride_hailing_zone' ? estimateRideHailingBounds(edge, activePreset) : null;

  const customPerKmEst = p.pricing_model === 'per_km_custom'
    ? {
        low: Math.round((p.base_fare_idr?.[0] || 0) + Number(edge.distance_km || 0) * (p.per_km_idr?.[0] || 0)),
        high: Math.round((p.base_fare_idr?.[1] || 0) + Number(edge.distance_km || 0) * (p.per_km_idr?.[1] || 0)),
      }
    : null;

  body.innerHTML = `
    <div class="stack-col" style="gap: 10px;">
      <div class="field-row-2">
        <div class="field-group">
          <label>Label Rute</label>
          <input id="ed-label" class="control-input" value="${escapeHtml(edge.label || '')}">
        </div>
        <div class="field-group">
          <label>Jarak (km)</label>
          <input id="ed-dist" class="control-input num" type="number" step="0.1" min="0.1" value="${edge.distance_km}">
          ${differsFromMap ? `<button id="ed-snap-map-dist" class="inline-link-btn" type="button">Sesuaikan ke jarak peta (${mapDistKm.toFixed(1)} km)</button>` : ''}
        </div>
      </div>

      <div class="field-row-2">
        <div class="field-group">
          <label>Dari</label>
          <select id="ed-src" class="control-select">
            ${nodes.map((n) => `<option value="${escapeHtml(n.id)}" ${edge.source === n.id ? 'selected' : ''} ${n.id === edge.target ? 'disabled' : ''}>${escapeHtml(n.label || n.id)}</option>`).join('')}
          </select>
        </div>
        <div class="field-group">
          <label>Ke</label>
          <select id="ed-dst" class="control-select">
            ${nodes.map((n) => `<option value="${escapeHtml(n.id)}" ${edge.target === n.id ? 'selected' : ''} ${n.id === edge.source ? 'disabled' : ''}>${escapeHtml(n.label || n.id)}</option>`).join('')}
          </select>
        </div>
      </div>

      <label class="checkbox-row">
        <input id="ed-bidir" type="checkbox" ${edge.bidirectional !== false ? 'checked' : ''}>
        <span>Dua arah (PP)</span>
      </label>

      <div class="field-row-2">
        <div class="field-group">
          <label>Moda Transportasi</label>
          <select id="ed-mode" class="control-select">
            ${[
              ['walk', 'Jalan Kaki'],
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
          <label>Skema Tarif</label>
          <select id="ed-pmodel" class="control-select" ${isWalk ? 'disabled' : ''}>
            ${availableModels.map(([m, lbl]) => `<option value="${m}" ${p.pricing_model === m ? 'selected' : ''}>${lbl}</option>`).join('')}
          </select>
        </div>
      </div>

      ${p.pricing_model === 'free' ? `
        <div class="kv-panel">
          <div class="kv-row"><span class="kv-key">Biaya Perjalanan</span><span class="kv-val">Rp 0</span></div>
          <div class="kv-row"><span class="kv-key">Aturan Jarak</span><span class="kv-val">${Number(edge.distance_km).toFixed(1)} km · mengikuti preferensi jalan kaki</span></div>
        </div>
      ` : ''}

      ${p.pricing_model === 'manual_range' ? `
        <div class="field-row-2">
          <div class="field-group">
            <label>Tarif Normal Min–Max (IDR)</label>
            <div class="field-row-2">
              <input id="ed-norm-min" class="control-input num" type="number" min="0" step="500" value="${(p.cost_normal_idr || [5000, 7000])[0]}">
              <input id="ed-norm-max" class="control-input num" type="number" min="0" step="500" value="${(p.cost_normal_idr || [5000, 7000])[1]}">
            </div>
          </div>
          <div class="field-group">
            <label>Tarif Sibuk Min–Max (IDR)</label>
            <div class="field-row-2">
              <input id="ed-peak-min" class="control-input num" type="number" min="0" step="500" value="${(p.cost_peak_idr || p.cost_normal_idr || [5000, 7000])[0]}">
              <input id="ed-peak-max" class="control-input num" type="number" min="0" step="500" value="${(p.cost_peak_idr || p.cost_normal_idr || [5000, 7000])[1]}">
            </div>
          </div>
        </div>
      ` : ''}

      ${p.pricing_model === 'per_km_custom' ? `
        <div class="field-row-2">
          <div class="field-group">
            <label>${edge.mode === 'personal' ? 'Biaya Parkir / Awal (Min–Max)' : 'Tarif Dasar (Min–Max IDR)'}</label>
            <div class="field-row-2">
              <input id="ed-base-min" class="control-input num" type="number" min="0" step="500" value="${(p.base_fare_idr || [0, 0])[0]}">
              <input id="ed-base-max" class="control-input num" type="number" min="0" step="500" value="${(p.base_fare_idr || [0, 0])[1]}">
            </div>
          </div>
          <div class="field-group">
            <label>${edge.mode === 'personal' ? 'Estimasi Bensin / Km (Min–Max)' : 'Tarif per Km (Min–Max IDR)'}</label>
            <div class="field-row-2">
              <input id="ed-km-min" class="control-input num" type="number" min="0" step="100" value="${(p.per_km_idr || [1000, 1500])[0]}">
              <input id="ed-km-max" class="control-input num" type="number" min="0" step="100" value="${(p.per_km_idr || [1000, 1500])[1]}">
            </div>
          </div>
        </div>
        ${customPerKmEst ? `
          <div class="kv-panel">
            <div class="kv-row"><span class="kv-key">Estimasi Biaya (${Number(edge.distance_km).toFixed(1)} km)</span><span class="kv-val">${formatIdr(customPerKmEst.low)} – ${formatIdr(customPerKmEst.high)}</span></div>
          </div>
        ` : ''}
      ` : ''}

      ${p.pricing_model === 'ride_hailing_zone' && isRideHailingMode ? `
        <div class="field-group">
          <label>Zona Layanan</label>
          <select id="ed-zone-preset" class="control-select">
            ${allowedPresets.map((z) => `<option value="${z}" ${activePresetKey === z ? 'selected' : ''}>${escapeHtml(zonePresets[z].label || z)}</option>`).join('')}
          </select>
        </div>
        ${est ? `
          <div class="kv-panel">
            <div class="kv-panel-head">${escapeHtml(activePreset.label)}</div>
            <div class="kv-row"><span class="kv-key">Tarif Minimal (0–${est.minKm} km)</span><span class="kv-val">${formatIdr(est.minFare[0])} – ${formatIdr(est.minFare[1])}</span></div>
            <div class="kv-row"><span class="kv-key">Per Km Berikutnya</span><span class="kv-val">${formatIdr(est.perKm[0])} – ${formatIdr(est.perKm[1])}</span></div>
            <div class="kv-row"><span class="kv-key">Biaya Jasa Aplikasi</span><span class="kv-val">${formatIdr(est.appFee[0])} – ${formatIdr(est.appFee[1])}</span></div>
            <div class="kv-row"><span class="kv-key">Estimasi Normal (${Number(edge.distance_km).toFixed(1)} km)</span><span class="kv-val">${formatIdr(est.normLow)} – ${formatIdr(est.normHigh)}</span></div>
            <div class="kv-row"><span class="kv-key">Estimasi Peak + Hujan</span><span class="kv-val">${formatIdr(est.surgeLow)} – ${formatIdr(est.surgeHigh)}</span></div>
          </div>
        ` : ''}
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
      onUpdated();
      if (rerenderInspector) renderEdgeInspector(body, edge, worldConfig, onUpdated);
    });
  };

  const snapBtn = body.querySelector('#ed-snap-map-dist');
  if (snapBtn && mapDistKm !== null) {
    snapBtn.addEventListener('click', () => {
      edge.distance_km = mapDistKm;
      markDirty();
      onUpdated();
      renderEdgeInspector(body, edge, worldConfig, onUpdated);
    });
  }

  bind('#ed-label', (el) => { edge.label = el.value; });
  bind('#ed-src', (el) => {
    edge.source = el.value;
    const a = nodes.find((n) => n.id === edge.source);
    const b = nodes.find((n) => n.id === edge.target);
    if (a && b) edge.distance_km = computeWorldDistanceKm(a.ui_position, b.ui_position);
  }, true);
  bind('#ed-dst', (el) => {
    edge.target = el.value;
    const a = nodes.find((n) => n.id === edge.source);
    const b = nodes.find((n) => n.id === edge.target);
    if (a && b) edge.distance_km = computeWorldDistanceKm(a.ui_position, b.ui_position);
  }, true);
  bind('#ed-dist', (el) => { edge.distance_km = Math.max(0.1, Number(el.value || 0.1)); }, true);
  bind('#ed-bidir', (el) => { edge.bidirectional = el.checked; });
  bind('#ed-mode', (el) => { applyTransportModeDefaults(edge, el.value); }, true);
  bind('#ed-pmodel', (el) => {
    edge.pricing = buildCleanPricing(el.value, edge.mode === 'gocar' ? 'gocar' : 'goride_zona_1');
  }, true);
  bind('#ed-norm-min', (el) => { p.cost_normal_idr = clampPair(el.value, (p.cost_normal_idr || [5000, 7000])[1]); }, true);
  bind('#ed-norm-max', (el) => { p.cost_normal_idr = clampPair((p.cost_normal_idr || [5000, 7000])[0], el.value); }, true);
  bind('#ed-peak-min', (el) => { p.cost_peak_idr = clampPair(el.value, (p.cost_peak_idr || p.cost_normal_idr || [5000, 7000])[1]); }, true);
  bind('#ed-peak-max', (el) => { p.cost_peak_idr = clampPair((p.cost_peak_idr || p.cost_normal_idr || [5000, 7000])[0], el.value); }, true);
  bind('#ed-base-min', (el) => { p.base_fare_idr = clampPair(el.value, (p.base_fare_idr || [0, 0])[1]); }, true);
  bind('#ed-base-max', (el) => { p.base_fare_idr = clampPair((p.base_fare_idr || [0, 0])[0], el.value); }, true);
  bind('#ed-km-min', (el) => { p.per_km_idr = clampPair(el.value, (p.per_km_idr || [1000, 1500])[1]); }, true);
  bind('#ed-km-max', (el) => { p.per_km_idr = clampPair((p.per_km_idr || [1000, 1500])[0], el.value); }, true);
  bind('#ed-zone-preset', (el) => { edge.pricing = buildCleanPricing('ride_hailing_zone', el.value); }, true);
}

const MOBILITY_SHORT_LABELS = {
  'goride:goride_zona_1': 'GoRide Zona I',
  'goride:goride_zona_2': 'GoRide Zona II',
  'goride:goride_zona_3': 'GoRide Zona III',
  'gocar:gocar': 'GoCar',
  'personal:custom': 'Kendaraan Pribadi',
  'public_transit:manual': 'Angkutan Umum',
};

export function initGraphStudioPage(root) {
  const svg = root.querySelector('#spatial-graph-svg');
  const mobSummaryEl = root.querySelector('#mobility-summary-value');
  const toggleMobBtn = root.querySelector('#btn-toggle-mobility');
  const mobDrawer = root.querySelector('#mobility-drawer');
  const globalMobSel = root.querySelector('#global-mobility-mode');
  const applyMobBtn = root.querySelector('#btn-apply-mobility-all');
  const srcSel = root.querySelector('#preview-src');
  const dstSel = root.querySelector('#preview-dst');
  const prefSel = root.querySelector('#preview-pref');
  const climateSel = root.querySelector('#preview-climate');
  const tariffSel = root.querySelector('#preview-tariff');
  const rainChk = root.querySelector('#preview-rain');
  const previewOut = root.querySelector('#route-preview-output');
  const walkPrefEl = root.querySelector('#walk-pref-select');
  const walkLimitsRow = root.querySelector('#walk-limits-row');
  const walkAvoidNote = root.querySelector('#walk-avoid-note');
  const walkMaxEl = root.querySelector('#walk-max-km');
  const walkRainEl = root.querySelector('#walk-max-rain-km');
  const scaleBarLine = root.querySelector('#scale-bar-line');
  const scaleBarLabel = root.querySelector('#scale-bar-label');
  const connectBtn = root.querySelector('#btn-connect-mode');
  const modeBanner = root.querySelector('#graph-mode-banner');
  const inspTitle = root.querySelector('#inspector-title');
  const inspSub = root.querySelector('#inspector-subtitle');
  const inspBody = root.querySelector('#inspector-body');
  const delBtn = root.querySelector('#btn-delete-selected');

  let connectMode = false;
  let connectSourceId = null;
  let addingLocation = false;
  let lastLoadedConfigRef = null;

  const viewport = createSpatialGraphViewport(svg, {
    onCameraChange: ({ widthPx, label }) => {
      scaleBarLine.style.width = `${widthPx}px`;
      scaleBarLabel.textContent = label;
    },
  });

  function updateWalkVisibilityAndSummary() {
    const wp = appState.worldConfig?.mobility_and_climate?.walk_physics;
    if (!wp) return;
    const isAvoid = wp.walk_preference === 'avoid_walk';
    walkLimitsRow.hidden = isAvoid;
    walkAvoidNote.hidden = !isAvoid;

    const modeLabel = MOBILITY_SHORT_LABELS[globalMobSel.value] || 'Kustom';
    let walkLabel = 'Jalan seimbang';
    if (isAvoid) {
      walkLabel = 'Hindari jalan kaki';
    } else if (wp.walk_preference === 'prefer_walk') {
      walkLabel = wp.max_walk_distance_km ? `Suka jalan ≤ ${wp.max_walk_distance_km} km` : 'Suka jalan kaki';
    } else if (wp.max_walk_distance_km !== null && wp.max_walk_distance_km !== undefined) {
      walkLabel = `Jalan maks ${wp.max_walk_distance_km} km`;
    }
    mobSummaryEl.textContent = `${modeLabel} · ${walkLabel}`;
  }

  function setConnectMode(active) {
    connectMode = active;
    connectSourceId = null;
    connectBtn.classList.toggle('is-active-tool', active);
    if (active) {
      modeBanner.hidden = false;
      modeBanner.textContent = 'Mode Hubungkan aktif — klik lokasi asal lalu lokasi tujuan di peta (Esc untuk batal).';
    } else {
      modeBanner.hidden = true;
    }
    redrawGraph();
  }

  function syncMobilityControls() {
    const wp = appState.worldConfig?.mobility_and_climate?.walk_physics;
    if (!wp) return;
    walkPrefEl.value = wp.walk_preference || 'balanced';
    walkMaxEl.value = wp.max_walk_distance_km ?? '';
    walkRainEl.value = wp.max_walk_distance_rain_km ?? '';
    globalMobSel.value = inferGlobalProfileValue(appState.worldConfig);
    updateWalkVisibilityAndSummary();
  }

  function syncPreviewSelectors() {
    const nodes = appState.worldConfig?.world_graph?.locations || [];
    const curSrc = srcSel.value;
    const curDst = dstSel.value;
    const opts = nodes.map((n) => `<option value="${escapeHtml(n.id)}">${escapeHtml(n.label || n.id)}</option>`).join('');
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
      inspSub.textContent = 'Otomatis terhubung ke graf sesuai Profil Mobilitas Utama';
      delBtn.hidden = true;
      renderAddLocationForm(
        inspBody,
        appState.worldConfig,
        viewport.getCenterWorldPos(),
        globalMobSel.value,
        (createdId) => {
          addingLocation = false;
          appState.selectedGraphItem = { type: 'node', id: createdId };
          markDirty();
          syncPreviewSelectors();
          redrawGraph();
          renderInspector();
          runPreview();
          showToast(`Lokasi '${createdId}' ditambahkan dan dihubungkan ke peta.`);
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
      inspTitle.textContent = 'Inspektor Lokasi & Rute';
      inspSub.textContent = 'Klik lokasi atau garis rute di peta untuk mengedit detailnya';
      inspBody.innerHTML =
        '<p class="panel-sub">Seret titik lokasi ke mana saja di peta untuk menyesuaikan jarak secara otomatis tanpa batas area. Klik garis rute bila ingin mengubah moda, jarak manual, atau tarif khusus.</p>';
      delBtn.hidden = true;
      return;
    }
    if (sel.type === 'node') {
      const node = wg.locations.find((n) => n.id === sel.id);
      if (!node) return;
      inspTitle.textContent = `Lokasi: ${node.label || node.id}`;
      inspSub.textContent = 'Kegiatan, makanan lokal, dampak suasana, dan profil belajar';
      delBtn.hidden = node.id === wg.home_node_id;
      renderNodeInspector(inspBody, node, appState.worldConfig, () => {
        inspTitle.textContent = `Lokasi: ${node.label || node.id}`;
        syncPreviewSelectors();
        redrawGraph();
      });
    } else {
      const edge = wg.edges.find((e) => e.id === sel.id);
      if (!edge) return;
      inspTitle.textContent = `Rute: ${edge.label || edge.id}`;
      inspSub.textContent = 'Atur jarak, moda transportasi, atau skema tarif khusus';
      delBtn.hidden = false;
      renderEdgeInspector(inspBody, edge, appState.worldConfig, () => {
        inspTitle.textContent = `Rute: ${edge.label || edge.id}`;
        redrawGraph();
        runPreview();
      });
    }
  }

  function redrawGraph() {
    if (!appState.worldConfig) return;
    viewport.render(appState.worldConfig, {
      interactive: true,
      selectedItem: appState.selectedGraphItem,
      connectSourceId,
      highlightedEdgeIds: appState.previewedPath?.edge_ids || [],
      highlightMode: appState.previewedPath?.preference || 'fastest',
      onCanvasClick: () => {
        if (connectMode) setConnectMode(false);
      },
      onSelect: (item) => {
        if (connectMode) {
          if (item.type !== 'node') return;
          if (!connectSourceId) {
            connectSourceId = item.id;
            modeBanner.textContent = `Asal: ${item.id} — sekarang klik lokasi tujuan di peta.`;
            redrawGraph();
          } else if (connectSourceId !== item.id) {
            const wg = appState.worldConfig.world_graph;
            const srcNode = wg.locations.find((n) => n.id === connectSourceId);
            const dstNode = wg.locations.find((n) => n.id === item.id);
            const distKm = computeWorldDistanceKm(srcNode.ui_position, dstNode.ui_position);
            const newId = nextUniqueId(`route_${connectSourceId}_${item.id}`, wg.edges.map((e) => e.id));
            const newEdge = {
              id: newId,
              label: `${srcNode.label || srcNode.id} ↔ ${dstNode.label || dstNode.id}`,
              source: connectSourceId,
              target: item.id,
              bidirectional: true,
              mode: 'goride',
              distance_km: distKm,
              pricing: { pricing_model: 'free' },
            };
            applyGlobalProfileToEdge(newEdge, globalMobSel.value);
            wg.edges.push(newEdge);
            appState.selectedGraphItem = { type: 'edge', id: newId };
            addingLocation = false;
            setConnectMode(false);
            markDirty();
            renderInspector();
            runPreview();
            showToast(`Rute dibuat (${distKm.toFixed(1)} km) mengikuti Profil Mobilitas Utama.`);
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
        climate: climateSel.value,
        ojol_tariff: tariffSel.value,
        is_raining: rainChk.checked,
      });
      appState.previewedPath = res;
      const routeLabels = (res.edges || []).map((e) => e.label || e.edge_id).join(' → ') || 'Tetap di lokasi';
      previewOut.innerHTML = `
        <div class="kv-row"><span class="kv-key">Jalur Terpilih</span><span class="kv-val">${escapeHtml(routeLabels)}</span></div>
        <div class="kv-row"><span class="kv-key">Total Jarak</span><span class="kv-val">${res.total_distance_km.toFixed(1)} km (jalan ${res.walk_distance_km.toFixed(1)} km)</span></div>
        <div class="kv-row"><span class="kv-key">Estimasi Ongkos</span><span class="kv-val">${formatIdr(res.min_fare_idr)} – ${formatIdr(res.max_fare_idr)}</span></div>
        <div class="kv-row"><span class="kv-key">Dampak Fisik</span><span class="kv-val">Energi -${res.total_energy_cost.toFixed(1)} · Mood ${res.total_mood_delta >= 0 ? '+' : ''}${res.total_mood_delta.toFixed(1)}</span></div>
      `;
      redrawGraph();
    } catch (err) {
      appState.previewedPath = null;
      previewOut.innerHTML = `<div class="kv-row"><span class="kv-key">Status</span><span class="kv-val" style="color: var(--status-err);">${escapeHtml(err.message)}</span></div>`;
      redrawGraph();
    }
  }

  toggleMobBtn.addEventListener('click', () => {
    const nextOpen = mobDrawer.hidden;
    mobDrawer.hidden = !nextOpen;
    toggleMobBtn.setAttribute('aria-expanded', String(nextOpen));
    toggleMobBtn.textContent = nextOpen ? 'Tutup Mobilitas' : 'Atur Mobilitas';
  });

  root.querySelector('#btn-map-zoom-in').addEventListener('click', () => viewport.zoomBy(0.75));
  root.querySelector('#btn-map-zoom-out').addEventListener('click', () => viewport.zoomBy(1.33));
  root.querySelector('#btn-map-fit').addEventListener('click', () => viewport.fitToNodes(appState.worldConfig));

  window.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && connectMode) setConnectMode(false);
  });
  window.addEventListener('resize', () => {
    if (appState.worldConfig && !root.hidden) redrawGraph();
  });

  globalMobSel.addEventListener('change', updateWalkVisibilityAndSummary);

  applyMobBtn.addEventListener('click', () => {
    const edges = appState.worldConfig?.world_graph?.edges || [];
    let count = 0;
    edges.forEach((e) => {
      if (e.mode !== 'walk') {
        applyGlobalProfileToEdge(e, globalMobSel.value);
        count += 1;
      }
    });
    markDirty();
    updateWalkVisibilityAndSummary();
    redrawGraph();
    if (appState.selectedGraphItem?.type === 'edge') renderInspector();
    runPreview();
    showToast(`Profil mobilitas diterapkan ke ${count} rute kendaraan (rute jalan kaki dipertahankan).`);
  });

  walkPrefEl.addEventListener('change', () => {
    appState.worldConfig.mobility_and_climate.walk_physics.walk_preference = walkPrefEl.value;
    markDirty();
    updateWalkVisibilityAndSummary();
    runPreview();
  });
  walkMaxEl.addEventListener('change', () => {
    appState.worldConfig.mobility_and_climate.walk_physics.max_walk_distance_km = numOrNull(walkMaxEl.value);
    markDirty();
    updateWalkVisibilityAndSummary();
    runPreview();
  });
  walkRainEl.addEventListener('change', () => {
    appState.worldConfig.mobility_and_climate.walk_physics.max_walk_distance_rain_km = numOrNull(walkRainEl.value);
    markDirty();
    updateWalkVisibilityAndSummary();
    runPreview();
  });

  connectBtn.addEventListener('click', () => setConnectMode(!connectMode));

  [srcSel, dstSel, prefSel, climateSel, tariffSel, rainChk].forEach((el) => el.addEventListener('change', runPreview));

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
      syncActivitiesWithGraph(appState.worldConfig);
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
      syncMobilityControls();
      syncPreviewSelectors();
      redrawGraph();
      if (lastLoadedConfigRef !== appState.worldConfig) {
        lastLoadedConfigRef = appState.worldConfig;
        requestAnimationFrame(() => viewport.fitToNodes(appState.worldConfig));
      }
      renderInspector();
      runPreview();
    },
    onLeave() {
      if (connectMode) setConnectMode(false);
    },
  };
}
