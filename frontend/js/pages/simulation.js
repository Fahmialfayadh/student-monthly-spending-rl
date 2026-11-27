import { api } from '../api.js';
import { renderSpatialGraph } from '../graph_canvas.js';
import { appState, showToast } from '../state.js';

export function initSimulationPage(root) {
  const svg = root.querySelector('#sim-graph-svg');
  const subtitle = root.querySelector('#sim-step-subtitle');
  const balanceVal = root.querySelector('#sim-balance-val');
  const attendVal = root.querySelector('#sim-attend-val');
  const pantryVal = root.querySelector('#sim-pantry-val');
  const metersEl = root.querySelector('#sim-meters');
  const actionFormEl = root.querySelector('#sim-action-form');
  const logTbody = root.querySelector('#sim-log-tbody');
  const policyModeSel = root.querySelector('#sim-policy-mode');
  const runSel = root.querySelector('#sim-run-select');
  const playBtn = root.querySelector('#btn-sim-play');

  let playTimer = null;

  function renderSession() {
    const sess = appState.simSession;
    if (!sess) return;
    const obs = sess.observation || {};
    const gs = sess.graph_state || {};

    subtitle.textContent = `Step ${gs.current_step ?? 0} · Location: ${gs.current_node_id ?? '-'}${sess.terminated ? ' · TERMINATED' : ''}`;
    balanceVal.textContent = `Rp ${Math.round(obs.balance_idr ?? 0).toLocaleString()}`;
    attendVal.textContent = `${((obs.attendance_rate_so_far ?? 1) * 100).toFixed(0)}%`;
    pantryVal.textContent = `Rp ${Math.round(obs.pantry_inventory_value_idr ?? 0).toLocaleString()}`;

    const meterDefs = [
      ['Energy', obs.energy ?? 0],
      ['Hunger', obs.hunger ?? 0],
      ['Mood', obs.mood ?? 0],
      ['Stress', obs.academic_stress ?? 0],
      ['Craving', obs.cafe_craving ?? 0],
    ];
    metersEl.innerHTML = meterDefs.map(([label, val]) => `
      <div class="meter-row">
        <span>${label}</span>
        <div class="meter-track"><div class="meter-fill" style="width: ${Math.max(0, Math.min(100, val))}%"></div></div>
        <span class="mono" style="text-align: right;">${val.toFixed(1)}</span>
      </div>
    `).join('');

    const heads = ['activity', 'route', 'meal_mode', 'cook_combo', 'bring_bekal', 'grocery_bundle'];
    actionFormEl.innerHTML = heads.map((h) => {
      const labels = sess.action_labels?.[h] || [];
      const masks = sess.action_masks?.[h] || [];
      return `
        <div class="field-group">
          <label class="mono">${h}</label>
          <select id="sim-act-${h}" class="control-select mono">
            ${labels.map((lbl, idx) => `<option value="${idx}" ${masks[idx] ? '' : 'disabled'}>${idx}: ${lbl}${masks[idx] ? '' : ' [masked]'}</option>`).join('')}
          </select>
        </div>
      `;
    }).join('');
    heads.forEach((h) => {
      const sel = actionFormEl.querySelector(`#sim-act-${h}`);
      const masks = sess.action_masks?.[h] || [];
      const firstValid = masks.findIndex(Boolean);
      if (sel && firstValid >= 0) sel.value = String(firstValid);
    });

    const lastEdges = sess.info?.step_events?.traversed_edges || [];
    renderSpatialGraph(svg, appState.worldConfig, {
      interactive: false,
      agentNodeId: gs.current_node_id,
      highlightedEdgeIds: lastEdges,
    });

    logTbody.innerHTML = appState.simLogs.slice(-30).reverse().map((row) => `
      <tr>
        <td class="mono">${row.step}</td>
        <td class="mono">${row.location}</td>
        <td class="mono">${row.activity}</td>
        <td class="mono">${row.route}</td>
        <td class="mono">${row.meal}</td>
        <td class="mono">Rp ${Math.round(row.balance).toLocaleString()}</td>
        <td class="mono">${row.reward.toFixed(2)}</td>
      </tr>
    `).join('');
  }

  function pushLog(snap) {
    const ev = snap.info?.step_events || {};
    appState.simLogs.push({
      step: snap.graph_state?.current_step ?? 0,
      location: snap.graph_state?.current_node_id ?? '-',
      activity: ev.executed_activity || '-',
      route: ev.executed_route || 'stay',
      meal: ev.executed_meal_mode || 'skip',
      balance: snap.observation?.balance_idr ?? 0,
      reward: snap.reward ?? 0,
    });
  }

  async function ensureSession() {
    if (appState.simSession) return;
    const sess = await api.createSimSession({ config: appState.worldConfig, seed: 42 });
    appState.simSession = sess;
    appState.simLogs = [];
    renderSession();
  }

  async function doAutoStep() {
    await ensureSession();
    if (appState.simSession.terminated || appState.simSession.truncated) {
      stopPlay();
      return;
    }
    const mode = policyModeSel.value;
    const payload = mode === 'model'
      ? { mode: 'model', run_id: runSel.value, checkpoint: 'best', deterministic: true }
      : { mode: 'heuristic' };
    const snap = await api.autoStepSimSession(appState.simSession.session_id, payload);
    appState.simSession = snap;
    pushLog(snap);
    renderSession();
    if (snap.terminated || snap.truncated) stopPlay();
  }

  function stopPlay() {
    if (playTimer) {
      clearInterval(playTimer);
      playTimer = null;
      playBtn.textContent = 'Play';
    }
  }

  policyModeSel.addEventListener('change', () => {
    runSel.hidden = policyModeSel.value !== 'model';
  });

  root.querySelector('#btn-sim-step').addEventListener('click', async () => {
    try {
      await ensureSession();
      const heads = ['activity', 'route', 'meal_mode', 'cook_combo', 'bring_bekal', 'grocery_bundle'];
      const action = heads.map((h) => Number(actionFormEl.querySelector(`#sim-act-${h}`).value));
      const snap = await api.stepSimSession(appState.simSession.session_id, action);
      appState.simSession = snap;
      pushLog(snap);
      renderSession();
    } catch (err) {
      showToast(`Step failed: ${err.message}`);
    }
  });

  root.querySelector('#btn-sim-autostep').addEventListener('click', () => {
    doAutoStep().catch((err) => showToast(`Auto-step failed: ${err.message}`));
  });

  playBtn.addEventListener('click', () => {
    if (playTimer) {
      stopPlay();
    } else {
      playBtn.textContent = 'Pause';
      playTimer = setInterval(() => {
        doAutoStep().catch((err) => {
          stopPlay();
          showToast(`Play stopped: ${err.message}`);
        });
      }, 280);
    }
  });

  root.querySelector('#btn-sim-reset').addEventListener('click', async () => {
    stopPlay();
    try {
      appState.simSession = await api.createSimSession({ config: appState.worldConfig, seed: 42 });
      appState.simLogs = [];
      renderSession();
    } catch (err) {
      showToast(`Reset failed: ${err.message}`);
    }
  });

  return {
    async refresh() {
      const runs = appState.runs || [];
      runSel.innerHTML = runs.map((r) => `<option value="${r.run_id}">${r.run_id}</option>`).join('');
      try {
        await ensureSession();
        renderSession();
      } catch (err) {
        showToast(`Sandbox init failed: ${err.message}`);
      }
    },
  };
}
