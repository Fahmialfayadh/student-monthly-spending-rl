import { api } from '../api.js';
import { createSpatialGraphViewport } from '../graph_canvas.js';
import { appState, showToast } from '../state.js';
import { escapeHtml, formatIdr, formatRunDisplayLabel } from '../utils.js';

export function initSimulationPage(root) {
  const svg = root.querySelector('#sim-graph-svg');
  const scaleLine = root.querySelector('#sim-scale-line');
  const scaleLabel = root.querySelector('#sim-scale-label');
  const subtitle = root.querySelector('#sim-step-subtitle');
  const staleBanner = root.querySelector('#sim-stale-banner');
  const contextStrip = root.querySelector('#sim-context-strip');
  const spendingStrip = root.querySelector('#sim-spending-strip');
  const pantryStrip = root.querySelector('#sim-pantry-strip');
  const balanceVal = root.querySelector('#sim-balance-val');
  const attendVal = root.querySelector('#sim-attend-val');
  const pantryVal = root.querySelector('#sim-pantry-val');
  const metersEl = root.querySelector('#sim-meters');
  const actionFormEl = root.querySelector('#sim-action-form');
  const logTbody = root.querySelector('#sim-log-tbody');
  const seedInput = root.querySelector('#sim-seed-input');
  const policyModeSel = root.querySelector('#sim-policy-mode');
  const runSel = root.querySelector('#sim-run-select');
  const ckptSel = root.querySelector('#sim-ckpt-select');
  const stepBtn = root.querySelector('#btn-sim-step');
  const autoStepBtn = root.querySelector('#btn-sim-autostep');
  const playBtn = root.querySelector('#btn-sim-play');

  let playRunning = false;
  let playTimeout = null;
  let isStepping = false;
  let lastConfigRef = null;

  const viewport = createSpatialGraphViewport(svg, {
    onCameraChange: ({ widthPx, label }) => {
      if (scaleLine) scaleLine.style.width = `${widthPx}px`;
      if (scaleLabel) scaleLabel.textContent = label;
    },
  });

  root.querySelector('#btn-sim-zoom-in').addEventListener('click', () => viewport.zoomBy(0.75));
  root.querySelector('#btn-sim-zoom-out').addEventListener('click', () => viewport.zoomBy(1.33));
  root.querySelector('#btn-sim-fit').addEventListener('click', () => viewport.fitToNodes(appState.worldConfig));

  function stopPlay() {
    playRunning = false;
    if (playTimeout) {
      clearTimeout(playTimeout);
      playTimeout = null;
    }
    playBtn.textContent = 'Play';
  }

  function updateContextualActionVisibility(sess) {
    const actSel = actionFormEl.querySelector('#sim-act-activity');
    const mealSel = actionFormEl.querySelector('#sim-act-meal_mode');
    const cookWrap = actionFormEl.querySelector('#wrap-act-cook_combo');
    const bekalWrap = actionFormEl.querySelector('#wrap-act-bring_bekal');
    const grocWrap = actionFormEl.querySelector('#wrap-act-grocery_bundle');
    if (!actSel || !mealSel) return;

    const mealIdx = Number(mealSel.value);
    const actIdx = Number(actSel.value);
    const actSpec = (appState.worldConfig?.activities || [])[actIdx];
    const handler = actSpec?.handler || '';

    // Cook combo only relevant when cooking fresh (1) or meal_prep (2)
    if (cookWrap) cookWrap.hidden = !(mealIdx === 1 || mealIdx === 2);
    // Grocery bundle only relevant when activity is grocery_shop
    if (grocWrap) grocWrap.hidden = handler !== 'grocery_shop';
    // Bring bekal only relevant when mask allows bring_bekal=1
    const bekalAllowed = Boolean(sess?.action_masks?.bring_bekal?.[1]);
    if (bekalWrap) bekalWrap.hidden = !bekalAllowed;
  }

  function renderSession() {
    staleBanner.hidden = !appState.simStale;
    const sess = appState.simSession;
    if (!sess) return;

    const obs = sess.observation || {};
    const gs = sess.graph_state || {};
    const done = Boolean(sess.terminated || sess.truncated);
    const termReason = sess.info?.step_events?.termination_reason || '';

    subtitle.textContent = done
      ? `Episode selesai di step ${gs.current_step ?? 0} (${termReason || (sess.truncated ? 'truncated' : 'terminated')})`
      : `Step ${gs.current_step ?? 0}/${gs.total_steps ?? 150} · Lokasi: ${gs.current_node_label || gs.current_node_id || '-'}`;

    stepBtn.disabled = done;
    autoStepBtn.disabled = done;
    playBtn.disabled = done;

    contextStrip.innerHTML = `
      <div class="stat-pill">
        <div class="stat-pill-label">Hari &amp; Slot</div>
        <div class="stat-pill-value">Hari ${gs.current_day ?? 1} · ${escapeHtml(gs.time_slot_name || '-')}</div>
      </div>
      <div class="stat-pill">
        <div class="stat-pill-label">Cuaca</div>
        <div class="stat-pill-value">${gs.is_raining ? 'Hujan' : 'Cerah'}</div>
      </div>
      <div class="stat-pill">
        <div class="stat-pill-label">Jadwal Kuliah</div>
        <div class="stat-pill-value">${gs.is_class_scheduled_now ? 'Ada Kelas' : 'Kosong'}</div>
      </div>
    `;

    balanceVal.textContent = formatIdr(obs.balance_idr ?? 0);
    attendVal.textContent = `${((obs.attendance_rate_so_far ?? 1) * 100).toFixed(0)}%`;
    pantryVal.textContent = formatIdr(obs.pantry_inventory_value_idr ?? 0);

    const meterDefs = [
      ['Energi', obs.energy ?? 0],
      ['Lapar', obs.hunger ?? 0],
      ['Mood', obs.mood ?? 0],
      ['Stres', obs.academic_stress ?? 0],
      ['Jenuh', obs.cafe_craving ?? 0],
    ];
    metersEl.innerHTML = meterDefs.map(([label, val]) => `
      <div class="meter-row">
        <span>${label}</span>
        <div class="meter-track"><div class="meter-fill" style="width: ${Math.max(0, Math.min(100, val))}%"></div></div>
        <span class="num" style="text-align: right;">${val.toFixed(1)}</span>
      </div>
    `).join('');

    const sp = gs.spending_by_category_idr || sess.info?.spending_by_category_idr || {};
    spendingStrip.innerHTML = `
      <div class="kv-panel-head">Pengeluaran Episode Berjalan</div>
      <div class="kv-row"><span class="kv-key">Transportasi</span><span class="kv-val">${formatIdr(sp.transport)}</span></div>
      <div class="kv-row"><span class="kv-key">Belanja Dapur</span><span class="kv-val">${formatIdr(sp.grocery)}</span></div>
      <div class="kv-row"><span class="kv-key">Makan Luar</span><span class="kv-val">${formatIdr(sp.external_food)}</span></div>
      <div class="kv-row"><span class="kv-key">Kafe &amp; Impulsif</span><span class="kv-val">${formatIdr((sp.cafe || 0) + (sp.minimarket_impulse || 0))}</span></div>
    `;

    const pantry = gs.pantry || {};
    const ingsCfg = appState.worldConfig?.pantry_and_cooking?.ingredients || {};
    const pantryItems = Object.entries(pantry)
      .filter(([k]) => k !== 'meal_prep' && k !== 'packed_lunch')
      .map(([k, qty]) => `${escapeHtml(ingsCfg[k]?.label || k)} ${qty}`)
      .join(' · ');
    pantryStrip.innerHTML = `
      <div class="kv-row"><span class="kv-key">Bahan Dapur</span><span class="kv-val">${pantryItems || '-'}</span></div>
      <div class="kv-row"><span class="kv-key">Siap Santap</span><span class="kv-val">Meal Prep ${pantry.meal_prep ?? 0} · Bekal ${pantry.packed_lunch ?? 0}</span></div>
    `;

    const headMeta = [
      ['activity', 'Aktivitas'],
      ['route', 'Preferensi Rute'],
      ['meal_mode', 'Mode Makan'],
      ['cook_combo', 'Kombinasi Masak'],
      ['bring_bekal', 'Bawa Bekal'],
      ['grocery_bundle', 'Paket Belanja'],
    ];
    actionFormEl.innerHTML = headMeta.map(([h, label]) => {
      const labels = sess.action_labels?.[h] || [];
      const masks = sess.action_masks?.[h] || [];
      return `
        <div class="field-group" id="wrap-act-${h}">
          <label>${label}</label>
          <select id="sim-act-${h}" class="control-select" ${done ? 'disabled' : ''}>
            ${labels.map((lbl, idx) => `<option value="${idx}" ${masks[idx] ? '' : 'disabled'}>${escapeHtml(lbl)}${masks[idx] ? '' : ' (terkunci)'}</option>`).join('')}
          </select>
        </div>
      `;
    }).join('');
    headMeta.forEach(([h]) => {
      const sel = actionFormEl.querySelector(`#sim-act-${h}`);
      const masks = sess.action_masks?.[h] || [];
      const firstValid = masks.findIndex(Boolean);
      if (sel && firstValid >= 0) sel.value = String(firstValid);
      if (sel && (h === 'activity' || h === 'meal_mode')) {
        sel.addEventListener('change', () => updateContextualActionVisibility(sess));
      }
    });
    updateContextualActionVisibility(sess);

    const lastEdges = sess.info?.step_events?.traversed_edges || [];
    viewport.render(appState.worldConfig, {
      interactive: false,
      agentNodeId: gs.current_node_id,
      highlightedEdgeIds: lastEdges,
    });
    if (lastConfigRef !== appState.worldConfig) {
      lastConfigRef = appState.worldConfig;
      requestAnimationFrame(() => viewport.fitToNodes(appState.worldConfig));
    }

    logTbody.innerHTML = appState.simLogs.slice(-30).reverse().map((row) => `
      <tr>
        <td class="num">${row.step}</td>
        <td>${escapeHtml(row.time)}</td>
        <td>${escapeHtml(row.location)}</td>
        <td>${escapeHtml(row.activity)}</td>
        <td class="code-id">${escapeHtml(row.route)}</td>
        <td>${escapeHtml(row.meal)}</td>
        <td class="num">${formatIdr(row.balance)}</td>
        <td class="num">${row.reward.toFixed(2)}</td>
      </tr>
    `).join('');
  }

  function pushLog(snap) {
    const ev = snap.info?.step_events || {};
    const gs = snap.graph_state || {};
    const actSpec = (appState.worldConfig?.activities || []).find((a) => a.id === ev.executed_activity);
    appState.simLogs.push({
      step: gs.current_step ?? 0,
      time: `H${gs.current_day ?? 1} ${gs.time_slot_name || ''}`,
      location: gs.current_node_label || gs.current_node_id || '-',
      activity: actSpec?.label || ev.executed_activity || '-',
      route: ev.executed_route || 'stay',
      meal: ev.executed_meal_mode || 'skip',
      balance: snap.observation?.balance_idr ?? 0,
      reward: snap.reward ?? 0,
    });
  }

  async function rebuildSession() {
    stopPlay();
    if (appState.simSession?.session_id) {
      api.deleteSimSession(appState.simSession.session_id).catch(() => {});
    }
    const seed = Math.max(0, Math.round(Number(seedInput.value || 42)));
    const sess = await api.createSimSession({ config: appState.worldConfig, seed });
    appState.simSession = sess;
    appState.simStale = false;
    appState.simLogs = [];
    renderSession();
  }

  async function ensureSession() {
    if (!appState.simSession) {
      await rebuildSession();
    }
  }

  async function doAutoStep() {
    if (isStepping) return;
    isStepping = true;
    try {
      await ensureSession();
      if (appState.simSession.terminated || appState.simSession.truncated) {
        stopPlay();
        return;
      }
      const mode = policyModeSel.value;
      const payload = mode === 'model'
        ? { mode: 'model', run_id: runSel.value, checkpoint: ckptSel.value, deterministic: true }
        : { mode: 'heuristic' };
      const snap = await api.autoStepSimSession(appState.simSession.session_id, payload);
      appState.simSession = snap;
      pushLog(snap);
      renderSession();
      if (snap.terminated || snap.truncated) stopPlay();
    } finally {
      isStepping = false;
    }
  }

  function scheduleNextPlayStep() {
    if (!playRunning) return;
    playTimeout = setTimeout(async () => {
      if (!playRunning) return;
      try {
        await doAutoStep();
        if (playRunning) scheduleNextPlayStep();
      } catch (err) {
        stopPlay();
        showToast(`Play dihentikan: ${err.message}`, 4500);
      }
    }, 260);
  }

  policyModeSel.addEventListener('change', () => {
    const isModel = policyModeSel.value === 'model';
    runSel.hidden = !isModel;
    ckptSel.hidden = !isModel;
  });

  stepBtn.addEventListener('click', async () => {
    if (isStepping) return;
    isStepping = true;
    try {
      await ensureSession();
      const heads = ['activity', 'route', 'meal_mode', 'cook_combo', 'bring_bekal', 'grocery_bundle'];
      const action = heads.map((h) => Number(actionFormEl.querySelector(`#sim-act-${h}`).value));
      const snap = await api.stepSimSession(appState.simSession.session_id, action);
      appState.simSession = snap;
      pushLog(snap);
      renderSession();
    } catch (err) {
      showToast(`Step gagal: ${err.message}`, 4500);
    } finally {
      isStepping = false;
    }
  });

  autoStepBtn.addEventListener('click', () => {
    doAutoStep().catch((err) => showToast(`Auto-step gagal: ${err.message}`, 4500));
  });

  playBtn.addEventListener('click', () => {
    if (playRunning) {
      stopPlay();
    } else {
      playRunning = true;
      playBtn.textContent = 'Pause';
      scheduleNextPlayStep();
    }
  });

  root.querySelector('#btn-sim-reset').addEventListener('click', () => {
    rebuildSession().catch((err) => showToast(`Reset gagal: ${err.message}`));
  });
  root.querySelector('#btn-sim-apply-new').addEventListener('click', () => {
    rebuildSession().catch((err) => showToast(`Gagal menerapkan config: ${err.message}`));
  });

  return {
    async refresh() {
      const runs = (appState.runs || []).filter((r) => (
        (r.has_best_checkpoint || r.has_final_checkpoint)
        && (r.final_summary !== null || Number(r.eval_points_count || 0) > 1 || r.run_id === appState.replayRunId)
      ));
      const modelOption = policyModeSel.querySelector('option[value="model"]');
      if (modelOption) modelOption.disabled = runs.length === 0;
      runSel.innerHTML = runs.map((r) => `<option value="${escapeHtml(r.run_id)}">${escapeHtml(formatRunDisplayLabel(r))}</option>`).join('');

      if (appState.replayRunId && runs.some((r) => r.run_id === appState.replayRunId)) {
        policyModeSel.value = 'model';
        runSel.value = appState.replayRunId;
        runSel.hidden = false;
        ckptSel.hidden = false;
        appState.replayRunId = null;
        await rebuildSession();
        return;
      }

      if (runs.length === 0 && policyModeSel.value === 'model') {
        policyModeSel.value = 'heuristic';
        runSel.hidden = true;
        ckptSel.hidden = true;
      }
      try {
        await ensureSession();
        renderSession();
      } catch (err) {
        showToast(`Gagal inisialisasi sandbox: ${err.message}`, 4500);
      }
    },
    onLeave() {
      stopPlay();
    },
  };
}
