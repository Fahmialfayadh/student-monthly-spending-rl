import { api } from '../api.js';
import { appState, showToast, subscribe } from '../state.js';
import { escapeHtml, formatIdr, formatRunDisplayLabel } from '../utils.js';

const TRAINING_PRESETS = [
  { id: 'fast', title: 'Cepat · 50k', sub: 'Uji cepat (~1 mnt)', total: 50000, evalInt: 10000, surv: 25000 },
  { id: 'medium', title: 'Menengah · 300k', sub: 'Seimbang', total: 300000, evalInt: 25000, surv: 150000 },
  { id: 'full', title: 'Penuh · 3M', sub: 'Kualitas terbaik', total: 3000000, evalInt: 100000, surv: 500000 },
];

const RESUME_INCREMENTS = [
  { id: 'inc_50k', title: '+50k Step', sub: 'Lanjut cepat', add: 50000 },
  { id: 'inc_250k', title: '+250k Step', sub: 'Fine-tune', add: 250000 },
  { id: 'inc_1m', title: '+1M Step', sub: 'Ekspansi jauh', add: 1000000 },
];

function svgNode(tag, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function formatStepShort(steps) {
  const s = Number(steps || 0);
  if (s >= 1_000_000) return `${(s / 1_000_000).toFixed(s % 1_000_000 === 0 ? 0 : 1)}M`;
  if (s >= 1_000) return `${Math.round(s / 1_000)}k`;
  return String(s);
}

function pickBestEvalRow(evalHistory) {
  if (!evalHistory || evalHistory.length === 0) return null;
  let best = evalHistory[0];
  for (const row of evalHistory) {
    const rComp = Number(row.completion_rate || 0);
    const bComp = Number(best.completion_rate || 0);
    if (rComp > bComp) {
      best = row;
    } else if (rComp === bComp) {
      const rSec = rComp >= 1.0 ? Number(row.mean_episode_reward || -1e9) : Number(row.mean_days_survived || 0);
      const bSec = bComp >= 1.0 ? Number(best.mean_episode_reward || -1e9) : Number(best.mean_days_survived || 0);
      if (rSec > bSec) best = row;
    }
  }
  return best;
}

export function initTrainingPage(root) {
  const formEl = root.querySelector('#training-form-container');
  const startBtn = root.querySelector('#btn-train-start');
  const gammaWarnEl = root.querySelector('#train-gamma-warning');
  const jobsEl = root.querySelector('#training-jobs-list');
  const activeRunTitleEl = root.querySelector('#active-run-title');
  const runsPanelSubEl = root.querySelector('#runs-panel-sub');
  const runEmptyEl = root.querySelector('#run-empty-state');
  const runChartBox = root.querySelector('#run-chart-container');
  const chartScrollEl = root.querySelector('#eval-chart-scroll');
  const chartReadoutEl = root.querySelector('#eval-chart-readout');
  const delRunBtn = root.querySelector('#btn-delete-run');
  const runSummaryEl = root.querySelector('#run-summary-metrics');
  const runAllocEl = root.querySelector('#run-allocation-panel');
  const replayBtn = root.querySelector('#btn-replay-run');
  const evalSvg = root.querySelector('#run-eval-svg');

  const openHistBtn = root.querySelector('#btn-open-run-history');
  const histDialog = root.querySelector('#run-history-dialog');
  const histListEl = root.querySelector('#dlg-runs-list');
  const histCountEl = root.querySelector('#dlg-runs-count');
  const cleanEmptyBtn = root.querySelector('#btn-clean-empty-runs');

  let pollTimer = null;
  let advancedOpen = false;
  let userEditedExpName = false;
  let selectedModelValue = '__new__'; // '__new__' | existing run_id
  let isAnyRunning = false;
  let currentDisplayedRunId = null;
  let activeCanonicalSha12 = '';
  let chartZoomPxPerPoint = 48;
  let lastEvalHistory = [];
  let chartManualScrolled = false;
  let refreshSeq = 0;

  subscribe((_st, reason) => {
    if (reason === 'config_loaded') {
      userEditedExpName = false;
      selectedModelValue = '__new__';
      activeCanonicalSha12 = String(_st.activeMeta?.canonicalSha256 || '').slice(0, 12);
    }
  });

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function splitResumableRuns() {
    const allRuns = appState.runs || [];
    const shaCandidates = new Set(
      [
        activeCanonicalSha12,
        String(appState.activeMeta?.canonicalSha256 || '').slice(0, 12),
        String(appState.activeMeta?.sha256 || '').slice(0, 12),
      ].filter(Boolean),
    );
    const valid = allRuns.filter((r) => {
      const hasCkpt = Boolean(r.has_best_checkpoint || r.has_final_checkpoint);
      const hasProgress = r.final_summary !== null || Number(r.eval_points_count || 0) > 1;
      return hasCkpt && hasProgress;
    });
    const compatRuns = [];
    const otherRuns = [];
    for (const r of valid) {
      const sha12 = r.env_config_sha256 ? String(r.env_config_sha256).slice(0, 12) : '';
      if (shaCandidates.size > 0 && sha12 && shaCandidates.has(sha12)) {
        compatRuns.push(r);
      } else {
        otherRuns.push(r);
      }
    }
    return { compatRuns, otherRuns, allResumable: valid };
  }

  function formatPointReadout(row, isBest = false) {
    if (!row) return 'Arahkan kursor ke titik evaluasi untuk detail metrik';
    const step = Number(row.timesteps || 0).toLocaleString('id-ID');
    const rew = Number(row.mean_episode_reward || 0).toFixed(1);
    const std = Number(row.std_episode_reward || 0).toFixed(1);
    const comp = `${(Number(row.completion_rate || 0) * 100).toFixed(0)}%`;
    const hari = Number(row.mean_days_survived || 0).toFixed(1);
    const bal = formatIdr(row.mean_final_balance_idr || 0);
    return `Step ${step} · Reward ${rew} ± ${std} · Selesai ${comp} (${hari} hr) · Sisa ${bal}${isBest ? ' · [BEST]' : ''}`;
  }

  function renderEvalChart(svg, evalHistory, fitMode = false) {
    lastEvalHistory = Array.isArray(evalHistory) ? evalHistory : [];
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const viewWidth = Math.max(460, chartScrollEl?.clientWidth || 540);
    const H = 220;
    if (lastEvalHistory.length === 0) {
      svg.setAttribute('viewBox', `0 0 ${viewWidth} ${H}`);
      svg.style.width = `${viewWidth}px`;
      const txt = svgNode('text', { x: viewWidth / 2, y: H / 2, 'text-anchor': 'middle', fill: 'var(--ink-muted)', 'font-size': 12 });
      txt.textContent = 'Belum ada riwayat evaluasi.';
      svg.appendChild(txt);
      chartReadoutEl.textContent = 'Belum ada titik evaluasi.';
      return;
    }

    const padL = 56;
    const padR = 24;
    const padT = 22;
    const padB = 38;
    const naturalW = padL + padR + Math.max(0, lastEvalHistory.length - 1) * chartZoomPxPerPoint;
    const totalW = fitMode ? viewWidth : Math.max(viewWidth, naturalW);
    const w = totalW - padL - padR;
    const h = H - padT - padB;

    svg.setAttribute('viewBox', `0 0 ${totalW} ${H}`);
    svg.style.width = `${totalW}px`;

    const steps = lastEvalHistory.map((r) => Number(r.timesteps ?? 0));
    const rewards = lastEvalHistory.map((r) => Number(r.mean_episode_reward ?? 0));
    const stds = lastEvalHistory.map((r) => Number(r.std_episode_reward ?? 0));

    let minR = Math.min(...rewards.map((r, i) => r - stds[i] * 0.5), ...rewards);
    let maxR = Math.max(...rewards.map((r, i) => r + stds[i] * 0.5), ...rewards);
    if (Math.abs(maxR - minR) < 1e-3) {
      minR -= 20;
      maxR += 20;
    } else {
      const padVal = (maxR - minR) * 0.1;
      minR -= padVal;
      maxR += padVal;
    }
    const spanR = Math.max(1e-6, maxR - minR);
    const minS = Math.min(...steps);
    const maxS = Math.max(...steps);
    const spanS = Math.max(1, maxS - minS);

    const yForVal = (val) => padT + (1 - (val - minR) / spanR) * h;

    [0, 0.25, 0.5, 0.75, 1].forEach((frac) => {
      const y = padT + (1 - frac) * h;
      const val = minR + frac * spanR;
      svg.appendChild(svgNode('line', { x1: padL, y1: y, x2: padL + w, y2: y, stroke: 'var(--border-subtle)', 'stroke-width': 1 }));
      const lbl = svgNode('text', { x: padL - 7, y: y + 3, 'text-anchor': 'end', fill: 'var(--ink-muted)', 'font-size': 10, 'font-family': 'var(--font-num)' });
      lbl.textContent = val.toFixed(0);
      svg.appendChild(lbl);
    });

    if (minR < 0 && maxR > 0) {
      const y0 = yForVal(0);
      svg.appendChild(svgNode('line', {
        x1: padL,
        y1: y0,
        x2: padL + w,
        y2: y0,
        stroke: 'var(--border-medium)',
        'stroke-width': 1,
        'stroke-dasharray': '4 3',
      }));
    }

    svg.appendChild(svgNode('line', { x1: padL, y1: padT, x2: padL, y2: padT + h, stroke: 'var(--border-strong)', 'stroke-width': 1 }));
    svg.appendChild(svgNode('line', { x1: padL, y1: padT + h, x2: padL + w, y2: padT + h, stroke: 'var(--border-strong)', 'stroke-width': 1 }));

    const coords = lastEvalHistory.map((row, i) => ({
      x: padL + (lastEvalHistory.length === 1 ? w / 2 : ((steps[i] - minS) / spanS) * w),
      y: yForVal(rewards[i]),
      yLow: yForVal(rewards[i] - stds[i] * 0.5),
      yHigh: yForVal(rewards[i] + stds[i] * 0.5),
      row,
    }));

    if (coords.length > 1) {
      const upper = coords.map((p) => `${p.x.toFixed(1)},${p.yHigh.toFixed(1)}`);
      const lower = coords.slice().reverse().map((p) => `${p.x.toFixed(1)},${p.yLow.toFixed(1)}`);
      svg.appendChild(svgNode('polygon', {
        points: [...upper, ...lower].join(' '),
        fill: 'rgba(17, 17, 17, 0.08)',
      }));

      svg.appendChild(svgNode('polyline', {
        fill: 'none',
        stroke: 'var(--ink-primary)',
        'stroke-width': 2.2,
        points: coords.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '),
      }));
    }

    const bestRow = pickBestEvalRow(lastEvalHistory);
    const hoverLine = svgNode('line', {
      x1: 0,
      y1: padT,
      x2: 0,
      y2: padT + h,
      stroke: 'var(--border-strong)',
      'stroke-width': 1,
      'stroke-dasharray': '3 3',
      visibility: 'hidden',
    });
    svg.appendChild(hoverLine);

    let lastTickX = -999;
    coords.forEach((p, idx) => {
      const isLast = idx === coords.length - 1;
      if (p.x - lastTickX >= 52 || (isLast && p.x - lastTickX >= 36)) {
        lastTickX = p.x;
        svg.appendChild(svgNode('line', { x1: p.x, y1: padT + h, x2: p.x, y2: padT + h + 4, stroke: 'var(--border-strong)', 'stroke-width': 1 }));
        const xt = svgNode('text', { x: p.x, y: padT + h + 15, 'text-anchor': 'middle', fill: 'var(--ink-muted)', 'font-size': 10, 'font-family': 'var(--font-num)' });
        xt.textContent = formatStepShort(p.row.timesteps);
        svg.appendChild(xt);
      }

      const isBest = bestRow && Number(p.row.timesteps) === Number(bestRow.timesteps);
      if (isBest) {
        svg.appendChild(svgNode('circle', {
          cx: p.x.toFixed(1),
          cy: p.y.toFixed(1),
          r: 6.5,
          fill: 'none',
          stroke: 'var(--status-ok)',
          'stroke-width': 1.8,
        }));
      }

      const dot = svgNode('circle', {
        cx: p.x.toFixed(1),
        cy: p.y.toFixed(1),
        r: isBest ? 3.8 : 3.2,
        fill: isBest ? 'var(--status-ok)' : 'var(--ink-primary)',
      });
      const hit = svgNode('circle', {
        cx: p.x.toFixed(1),
        cy: p.y.toFixed(1),
        r: 13,
        fill: 'transparent',
        style: 'cursor: crosshair;',
      });
      hit.addEventListener('mouseenter', () => {
        hoverLine.setAttribute('x1', p.x.toFixed(1));
        hoverLine.setAttribute('x2', p.x.toFixed(1));
        hoverLine.setAttribute('visibility', 'visible');
        chartReadoutEl.textContent = formatPointReadout(p.row, isBest);
      });
      hit.addEventListener('mouseleave', () => {
        hoverLine.setAttribute('visibility', 'hidden');
        chartReadoutEl.textContent = formatPointReadout(lastEvalHistory[lastEvalHistory.length - 1], bestRow === lastEvalHistory[lastEvalHistory.length - 1]);
      });

      svg.appendChild(dot);
      svg.appendChild(hit);
    });

    const yTitle = svgNode('text', { x: padL, y: 12, fill: 'var(--ink-secondary)', 'font-size': 10.5, 'font-weight': 600 });
    yTitle.textContent = 'Mean Episode Reward (±0.5 std)';
    svg.appendChild(yTitle);

    const xTitle = svgNode('text', { x: padL + w / 2, y: H - 6, 'text-anchor': 'middle', fill: 'var(--ink-secondary)', 'font-size': 10.5, 'font-weight': 600 });
    xTitle.textContent = `Timesteps (${formatStepShort(minS)} – ${formatStepShort(maxS)})`;
    svg.appendChild(xTitle);

    const latestRow = lastEvalHistory[lastEvalHistory.length - 1];
    chartReadoutEl.textContent = formatPointReadout(latestRow, bestRow === latestRow);

    if (!chartManualScrolled && chartScrollEl) {
      requestAnimationFrame(() => {
        chartScrollEl.scrollLeft = chartScrollEl.scrollWidth;
      });
    }
  }

  chartScrollEl.addEventListener('scroll', () => {
    const atRight = chartScrollEl.scrollWidth - chartScrollEl.scrollLeft - chartScrollEl.clientWidth < 16;
    chartManualScrolled = !atRight;
  });

  root.querySelector('#btn-chart-zoom-in').addEventListener('click', () => {
    chartZoomPxPerPoint = Math.min(140, Math.round(chartZoomPxPerPoint * 1.35));
    renderEvalChart(evalSvg, lastEvalHistory, false);
  });
  root.querySelector('#btn-chart-zoom-out').addEventListener('click', () => {
    chartZoomPxPerPoint = Math.max(18, Math.round(chartZoomPxPerPoint * 0.75));
    renderEvalChart(evalSvg, lastEvalHistory, false);
  });
  root.querySelector('#btn-chart-fit').addEventListener('click', () => {
    renderEvalChart(evalSvg, lastEvalHistory, true);
  });

  function checkGammaSync() {
    const tp = appState.trainingProfile;
    const shaping = appState.worldConfig?.rewards?.budget_potential_shaping;
    if (!tp || !shaping || !shaping.enabled) {
      gammaWarnEl.hidden = true;
      return;
    }
    const diff = Math.abs(Number(tp.ppo.discount_factor_gamma) - Number(shaping.gamma));
    if (diff > 1e-6) {
      gammaWarnEl.hidden = false;
      gammaWarnEl.innerHTML = `
        <span>Discount Gamma PPO (${tp.ppo.discount_factor_gamma}) berbeda dengan gamma shaping env (${shaping.gamma}).</span>
        <button id="btn-sync-gamma" class="btn btn-sm btn-primary" type="button">Samakan (${shaping.gamma})</button>
      `;
      gammaWarnEl.querySelector('#btn-sync-gamma').addEventListener('click', () => {
        tp.ppo.discount_factor_gamma = Number(shaping.gamma);
        renderForm();
        checkGammaSync();
      });
    } else {
      gammaWarnEl.hidden = true;
    }
  }

  function updateStartLockState() {
    const isResumeMode = selectedModelValue !== '__new__';
    if (startBtn) {
      startBtn.disabled = isAnyRunning;
      startBtn.textContent = isAnyRunning
        ? 'Sedang Training...'
        : isResumeMode
          ? 'Lanjutkan Training'
          : 'Mulai Training Baru';
    }
    const modelSel = formEl.querySelector('#tr-model-select');
    if (modelSel) modelSel.disabled = isAnyRunning;
    formEl.querySelectorAll('.preset-pill-btn, .resume-inc-btn').forEach((b) => {
      b.disabled = isAnyRunning;
    });
  }

  function renderForm() {
    const tp = appState.trainingProfile;
    if (!tp) return;
    const { compatRuns, otherRuns, allResumable } = splitResumableRuns();
    if (selectedModelValue !== '__new__' && !allResumable.some((r) => r.run_id === selectedModelValue)) {
      selectedModelValue = '__new__';
    }
    const resumeRun = selectedModelValue === '__new__'
      ? null
      : allResumable.find((r) => r.run_id === selectedModelValue) || null;
    const isResumeCompat = resumeRun ? compatRuns.some((r) => r.run_id === resumeRun.run_id) : false;
    const isResumeMode = Boolean(resumeRun);
    const resumePrevSteps = Number(resumeRun?.latest_eval_step || 0);
    if (isResumeMode && Number(tp.schedule.total_timesteps || 0) <= resumePrevSteps) {
      tp.schedule.total_timesteps = resumePrevSteps + 100000;
    }
    const curSteps = Number(tp.schedule.total_timesteps || 0);

    const renderOpt = (r) => `
      <option value="${escapeHtml(r.run_id)}" ${selectedModelValue === r.run_id ? 'selected' : ''}>
        ${escapeHtml(formatRunDisplayLabel(r))} · ${formatStepShort(r.latest_eval_step)}/${formatStepShort(r.total_timesteps)}
      </option>
    `;

    const modelHint = compatRuns.length > 0
      ? `${compatRuns.length} model cocok dengan config aktif${otherRuns.length > 0 ? ` (${otherRuns.length} model tersimpan lainnya juga tersedia)` : ''}.`
      : otherRuns.length > 0
        ? `Belum ada model khusus config ini; tersedia ${otherRuns.length} model tersimpan lainnya untuk di-resume.`
        : 'Belum ada model tersimpan dengan checkpoint yang dapat di-resume.';

    formEl.innerHTML = `
      <div class="field-group">
        <label for="tr-model-select">Model / Eksperimen</label>
        <select id="tr-model-select" class="control-select">
          <option value="__new__" ${!isResumeMode ? 'selected' : ''}>+ Buat Model Baru</option>
          ${compatRuns.length > 0 ? `
            <optgroup label="Cocok dengan Config Aktif">
              ${compatRuns.map(renderOpt).join('')}
            </optgroup>
          ` : ''}
          ${otherRuns.length > 0 ? `
            <optgroup label="Model Tersimpan Lainnya (Pakai Snapshot Config Model)">
              ${otherRuns.map(renderOpt).join('')}
            </optgroup>
          ` : ''}
        </select>
        <span class="hint-inline">${modelHint}</span>
      </div>

      ${!isResumeMode ? `
        <div class="field-group">
          <label>Preset Durasi Latihan</label>
          <div class="preset-pills">
            ${TRAINING_PRESETS.map((p) => `
              <button type="button" class="preset-pill-btn ${curSteps === p.total ? 'is-active' : ''}" data-preset="${p.id}">
                <span class="preset-pill-title">${p.title}</span>
                <span class="preset-pill-sub">${p.sub}</span>
              </button>
            `).join('')}
          </div>
        </div>

        <div class="field-row-2">
          <div class="field-group">
            <label for="tr-exp-name" class="tooltip-trigger" data-tooltip="Nama bebas untuk model baru yang akan dilatih (tidak harus sama dengan nama config).">Nama Model Baru</label>
            <input id="tr-exp-name" class="control-input code-id" value="${escapeHtml(tp.experiment_name)}" placeholder="mis. agen_hemat_v1">
          </div>
          <div class="field-group">
            <label for="tr-timesteps">Total Timesteps</label>
            <input id="tr-timesteps" class="control-input num" type="number" min="64" step="64" value="${tp.schedule.total_timesteps}">
          </div>
        </div>

        <div style="border-top: 1px solid var(--border-subtle); padding-top: 10px;" class="stack-col">
          <label class="checkbox-row">
            <input id="tr-bc-enabled" type="checkbox" ${tp.bc.enabled ? 'checked' : ''}>
            <span class="tooltip-trigger" data-tooltip="Warm-start policy dari demonstrasi heuristik sebelum fine-tuning PPO agar agen lebih cepat stabil.">Aktifkan Behavior Cloning Warm-Start</span>
          </label>
          ${tp.bc.enabled ? `
            <div class="field-row-3">
              <div class="field-group">
                <label>Episode Demo BC</label>
                <input id="tr-bc-ep" class="control-input num" type="number" min="1" step="1" value="${tp.bc.num_episodes}">
              </div>
              <div class="field-group">
                <label>Epoch BC</label>
                <input id="tr-bc-epochs" class="control-input num" type="number" min="1" step="1" value="${tp.bc.epochs}">
              </div>
              <div class="field-group">
                <label>Learning Rate BC</label>
                <input id="tr-bc-lr" class="control-input num" type="number" step="0.0001" min="0.00001" value="${tp.bc.learning_rate}">
              </div>
            </div>
          ` : ''}
        </div>
      ` : `
        <div class="kv-panel">
          <div class="kv-panel-head">Melanjutkan Model: ${escapeHtml(resumeRun.experiment_name || resumeRun.run_id)}</div>
          <div class="kv-row"><span class="kv-key">Kesesuaian Config</span><span class="kv-val">${isResumeCompat ? 'Cocok dengan config aktif' : 'Memakai snapshot config asli model'}</span></div>
          <div class="kv-row"><span class="kv-key">Checkpoint Awal</span><span class="kv-val">${resumeRun.has_best_checkpoint ? 'best_model' : 'final_model'}</span></div>
          <div class="kv-row"><span class="kv-key">Step Terlatih Sebelumnya</span><span class="kv-val num">${resumePrevSteps.toLocaleString('id-ID')} steps</span></div>
          <div class="kv-row"><span class="kv-key">Target Tambahan Sesi Ini</span><span class="kv-val num">+${Math.max(0, curSteps - resumePrevSteps).toLocaleString('id-ID')} steps</span></div>
        </div>

        <div class="field-group">
          <label>Tambah Durasi Resume</label>
          <div class="preset-pills">
            ${RESUME_INCREMENTS.map((inc) => `
              <button type="button" class="preset-pill-btn resume-inc-btn ${(curSteps - resumePrevSteps) === inc.add ? 'is-active' : ''}" data-add="${inc.add}">
                <span class="preset-pill-title">${inc.title}</span>
                <span class="preset-pill-sub">${inc.sub}</span>
              </button>
            `).join('')}
          </div>
        </div>

        <div class="field-group">
          <label for="tr-timesteps">Target Total Timesteps Kumulatif</label>
          <input id="tr-timesteps" class="control-input num" type="number" min="${resumePrevSteps + 64}" step="1000" value="${curSteps}">
        </div>
      `}

      <details id="tr-adv-details" class="section-disclosure" ${advancedOpen ? 'open' : ''}>
        <summary>Hyperparameter Lanjutan PPO &amp; Evaluasi</summary>
        <div class="disclosure-body">
          <div class="field-row-3">
            <div class="field-group">
              <label class="tooltip-trigger" data-tooltip="Harus sama dengan rewards.budget_potential_shaping.gamma saat shaping aktif.">Discount Gamma</label>
              <input id="tr-gamma" class="control-input num" type="number" step="0.001" min="0.5" max="1" value="${tp.ppo.discount_factor_gamma}">
            </div>
            <div class="field-group">
              <label class="tooltip-trigger" data-tooltip="Faktor bias-variance pada Generalized Advantage Estimation.">GAE Lambda</label>
              <input id="tr-gae" class="control-input num" type="number" step="0.01" min="0" max="1" value="${tp.ppo.gae_lambda}">
            </div>
            <div class="field-group">
              <label>Learning Rate</label>
              <input id="tr-lr" class="control-input num" type="number" step="0.00005" min="0.000001" value="${tp.ppo.learning_rate}">
            </div>
          </div>

          <div class="field-row-3">
            <div class="field-group">
              <label>Env Paralel</label>
              <input id="tr-envs" class="control-input num" type="number" min="1" max="32" step="1" value="${tp.ppo.num_environments}">
            </div>
            <div class="field-group">
              <label>Rollout Steps / Env</label>
              <input id="tr-rollout" class="control-input num" type="number" min="16" step="16" value="${tp.ppo.rollout_steps_per_env}">
            </div>
            <div class="field-group">
              <label>Batch Size</label>
              <input id="tr-batch" class="control-input num" type="number" min="8" step="8" value="${tp.ppo.batch_size}">
            </div>
          </div>

          <div class="field-row-3">
            <div class="field-group">
              <label class="tooltip-trigger" data-tooltip="Batas KL divergence untuk menghentikan update bila perubahan policy terlalu besar.">Target KL</label>
              <input id="tr-kl" class="control-input num" type="number" step="0.005" min="0.001" placeholder="nonaktif" value="${tp.ppo.target_kl ?? ''}">
            </div>
            <div class="field-group">
              <label class="tooltip-trigger" data-tooltip="Bobot bonus entropi untuk menjaga eksplorasi aksi.">Entropy Coef</label>
              <input id="tr-ent" class="control-input num" type="number" step="0.005" min="0" value="${tp.ppo.entropy_coefficient}">
            </div>
            <div class="field-group">
              <label>Interval Evaluasi (Step)</label>
              <input id="tr-eval-int" class="control-input num" type="number" min="64" step="64" value="${tp.schedule.eval_interval_steps}">
            </div>
          </div>
        </div>
      </details>
    `;

    formEl.querySelector('#tr-adv-details').addEventListener('toggle', (e) => {
      advancedOpen = e.target.open;
    });

    formEl.querySelector('#tr-model-select').addEventListener('change', async (e) => {
      selectedModelValue = e.target.value;
      chartManualScrolled = false;
      renderForm();
      await syncRightPanelOnly();
    });

    formEl.querySelectorAll('.preset-pill-btn[data-preset]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const preset = TRAINING_PRESETS.find((p) => p.id === btn.dataset.preset);
        if (!preset) return;
        tp.schedule.total_timesteps = preset.total;
        tp.schedule.eval_interval_steps = preset.evalInt;
        tp.schedule.survival_phase_timesteps = preset.surv;
        renderForm();
      });
    });

    formEl.querySelectorAll('.resume-inc-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const addSteps = Number(btn.dataset.add || 50000);
        tp.schedule.total_timesteps = resumePrevSteps + addSteps;
        renderForm();
      });
    });

    const bind = (id, fn, extra = null) => {
      const el = formEl.querySelector(id);
      if (!el) return;
      el.addEventListener('change', (e) => {
        fn(e.target);
        if (extra) extra();
      });
    };
    bind('#tr-exp-name', (el) => {
      userEditedExpName = true;
      tp.experiment_name = el.value.trim() || 'student_agent';
    });
    bind('#tr-timesteps', (el) => {
      const minTs = isResumeMode ? resumePrevSteps + 64 : 64;
      tp.schedule.total_timesteps = Math.max(minTs, Math.round(Number(el.value || minTs)));
    }, renderForm);
    bind('#tr-gamma', (el) => { tp.ppo.discount_factor_gamma = Number(el.value); }, checkGammaSync);
    bind('#tr-gae', (el) => { tp.ppo.gae_lambda = Number(el.value); });
    bind('#tr-lr', (el) => { tp.ppo.learning_rate = Number(el.value); });
    bind('#tr-envs', (el) => { tp.ppo.num_environments = Math.max(1, Math.round(Number(el.value || 1))); });
    bind('#tr-rollout', (el) => { tp.ppo.rollout_steps_per_env = Math.max(16, Math.round(Number(el.value || 16))); });
    bind('#tr-batch', (el) => { tp.ppo.batch_size = Math.max(8, Math.round(Number(el.value || 8))); });
    bind('#tr-kl', (el) => {
      const trimmed = el.value.trim();
      tp.ppo.target_kl = trimmed === '' || Number(trimmed) <= 0 ? null : Number(trimmed);
      if (tp.ppo.target_kl === null) el.value = '';
    });
    bind('#tr-ent', (el) => { tp.ppo.entropy_coefficient = Math.max(0, Number(el.value || 0)); });
    bind('#tr-eval-int', (el) => { tp.schedule.eval_interval_steps = Math.max(64, Math.round(Number(el.value || 64))); });
    bind('#tr-bc-enabled', (el) => { tp.bc.enabled = el.checked; }, renderForm);
    bind('#tr-bc-ep', (el) => { tp.bc.num_episodes = Math.max(1, Math.round(Number(el.value || 1))); });
    bind('#tr-bc-epochs', (el) => { tp.bc.epochs = Math.max(1, Math.round(Number(el.value || 1))); });
    bind('#tr-bc-lr', (el) => { tp.bc.learning_rate = Number(el.value); });
    updateStartLockState();
  }

  function renderRunHistoryDialog() {
    const runs = appState.runs || [];
    histCountEl.textContent = `${runs.length} folder run tersimpan di disk`;
    const emptyRuns = runs.filter((r) => !r.is_running && !r.final_summary && Number(r.eval_points_count || 0) <= 1);
    cleanEmptyBtn.disabled = emptyRuns.length === 0;
    cleanEmptyBtn.textContent = `Hapus Run Kosong (${emptyRuns.length})`;

    if (runs.length === 0) {
      histListEl.innerHTML = '<div style="padding: 12px;" class="panel-sub">Belum ada folder run di direktori runs/.</div>';
      return;
    }

    histListEl.innerHTML = runs.map((r) => {
      const statusTxt = r.is_running
        ? 'Sedang berjalan'
        : r.final_summary
          ? `Selesai (${(Number(r.final_summary.completion_rate || 0) * 100).toFixed(0)}%)`
          : Number(r.eval_points_count || 0) > 1
            ? `Terhenti di ${formatStepShort(r.latest_eval_step)}`
            : 'Kosong (Step 0)';
      return `
        <div class="run-history-row">
          <div style="min-width: 0;">
            <div style="font-weight: 600; color: var(--ink-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
              ${escapeHtml(formatRunDisplayLabel(r))}
            </div>
            <div class="hint-inline code-id">${escapeHtml(r.run_id)} · ${escapeHtml(statusTxt)}</div>
          </div>
          <button class="btn btn-sm btn-danger btn-hist-del" data-run="${escapeHtml(r.run_id)}" type="button" ${r.is_running ? 'disabled' : ''}>Hapus</button>
        </div>
      `;
    }).join('');

    histListEl.querySelectorAll('.btn-hist-del').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const rid = btn.dataset.run;
        try {
          await api.deleteRun(rid);
          if (selectedModelValue === rid) selectedModelValue = '__new__';
          showToast(`Run '${rid}' dihapus.`);
          await refreshJobsAndRuns();
          renderRunHistoryDialog();
        } catch (err) {
          showToast(`Gagal hapus: ${err.message}`, 4500);
        }
      });
    });
  }

  function pickTargetRun(jobs, allRuns) {
    const runningJob = jobs.find((j) => j.status === 'running');
    if (runningJob && runningJob.run_dir) {
      const runningRid = String(runningJob.run_dir).split('/').pop();
      const matched = allRuns.find((r) => r.run_id === runningRid);
      if (matched) return { run: matched, mode: 'running' };
      return {
        run: {
          run_id: runningRid,
          experiment_name: runningJob.experiment_name,
          is_running: true,
          has_best_checkpoint: false,
          has_final_checkpoint: false,
        },
        mode: 'running',
      };
    }

    if (selectedModelValue !== '__new__') {
      const chosen = allRuns.find((r) => r.run_id === selectedModelValue);
      if (chosen) {
        return { run: chosen, mode: chosen.final_summary ? 'completed' : 'partial' };
      }
    }

    const { compatRuns } = splitResumableRuns();
    if (compatRuns.length > 0) {
      return { run: compatRuns[0], mode: compatRuns[0].final_summary ? 'completed' : 'partial' };
    }
    return { run: null, mode: 'none' };
  }

  async function renderActiveRunPanel(target, hasRunning) {
    const { run, mode } = target;
    currentDisplayedRunId = run?.run_id || null;

    if (!run) {
      activeRunTitleEl.textContent = 'Hasil Model Konfigurasi Aktif';
      runsPanelSubEl.textContent = 'Belum ada model kompatibel untuk konfigurasi ini';
      runEmptyEl.hidden = false;
      runChartBox.hidden = true;
      runSummaryEl.innerHTML = '';
      runAllocEl.innerHTML = '';
      replayBtn.hidden = true;
      delRunBtn.hidden = true;
      return;
    }

    runEmptyEl.hidden = true;
    let detail = null;
    try {
      detail = await api.getRunDetail(run.run_id);
    } catch {
      detail = { eval_history: [], final_summary: null };
    }

    const hist = detail.eval_history || [];
    const latestHist = hist.length > 0 ? hist[hist.length - 1] : null;
    const s = detail.final_summary || latestHist;
    const isFinal = Boolean(detail.final_summary);
    const epDays = appState.worldConfig?.environment?.episode_days || 30;
    const hasCkpt = Boolean(run.has_best_checkpoint || run.has_final_checkpoint);

    if (mode === 'running') {
      activeRunTitleEl.textContent = `Evaluasi Langsung · ${run.experiment_name || run.run_id}`;
      runsPanelSubEl.textContent = latestHist
        ? `Checkpoint evaluasi terbaru pada step ${Number(latestHist.timesteps || 0).toLocaleString('id-ID')} (sedang berjalan)`
        : 'Menunggu evaluasi step pertama selesai...';
    } else if (isFinal) {
      activeRunTitleEl.textContent = `Hasil Model · ${run.experiment_name || run.run_id}`;
      runsPanelSubEl.textContent = `Selesai · ${formatRunDisplayLabel(run)}`;
    } else {
      activeRunTitleEl.textContent = `Progres Model · ${run.experiment_name || run.run_id}`;
      runsPanelSubEl.textContent = `Terhenti di step ${Number(latestHist?.timesteps || 0).toLocaleString('id-ID')} · pilih di Model/Eksperimen untuk Resume`;
    }

    replayBtn.hidden = !hasCkpt;
    delRunBtn.hidden = hasRunning || !currentDisplayedRunId;

    if (s) {
      const homeMeals = (s.mean_cook_fresh_count || 0) + (s.mean_meal_prep_count || 0) + (s.mean_reheat_prep_count || 0) + (s.mean_eat_bekal_count || 0);
      runSummaryEl.innerHTML = `
        <div class="metric-box"><div class="metric-box-label">Completion</div><div class="metric-box-value num">${(Number(s.completion_rate || 0) * 100).toFixed(0)}%</div></div>
        <div class="metric-box"><div class="metric-box-label">Rata-rata Hari</div><div class="metric-box-value num">${Number(s.mean_days_survived || 0).toFixed(1)} / ${epDays}</div></div>
        <div class="metric-box"><div class="metric-box-label">Sisa Saldo</div><div class="metric-box-value num">${formatIdr(s.mean_final_balance_idr || 0)}</div></div>
        <div class="metric-box"><div class="metric-box-label">Kehadiran</div><div class="metric-box-value num">${(Number(s.mean_class_attendance_rate || 0) * 100).toFixed(0)}%</div></div>
      `;
      runAllocEl.innerHTML = `
        <div class="kv-panel-head">${isFinal ? 'Alokasi Biaya & Kebiasaan Agen per Bulan (Evaluasi Akhir)' : `Snapshot Evaluasi Terbaru (Step ${Number(s.timesteps || 0).toLocaleString('id-ID')})`}</div>
        <div class="kv-row"><span class="kv-key">Transportasi</span><span class="kv-val">${formatIdr(s.mean_spent_transport_idr || 0)}</span></div>
        <div class="kv-row"><span class="kv-key">Belanja Bahan Dapur</span><span class="kv-val">${formatIdr(s.mean_spent_grocery_idr || 0)}</span></div>
        <div class="kv-row"><span class="kv-key">Makan Luar / Warung</span><span class="kv-val">${formatIdr(s.mean_spent_external_food_idr || 0)}</span></div>
        <div class="kv-row"><span class="kv-key">Kafe &amp; Jajan Impulsif</span><span class="kv-val">${formatIdr((s.mean_spent_cafe_idr || 0) + (s.mean_spent_minimarket_impulse_idr || 0))}</span></div>
        <div class="kv-row"><span class="kv-key">Masak / Bekal vs Hub / Kafe</span><span class="kv-val">${homeMeals.toFixed(1)}x masak · ${Number(s.mean_hub_or_leisure_count || 0).toFixed(1)}x hub/kafe</span></div>
      `;
    } else {
      runSummaryEl.innerHTML = '';
      runAllocEl.innerHTML = '';
    }

    runChartBox.hidden = hist.length === 0;
    if (hist.length > 0) {
      renderEvalChart(evalSvg, hist, false);
    }
  }

  let cachedJobs = [];

  async function syncRightPanelOnly() {
    const target = pickTargetRun(cachedJobs, appState.runs || []);
    await renderActiveRunPanel(target, isAnyRunning);
  }

  async function refreshJobsAndRuns() {
    const seq = ++refreshSeq;
    try {
      const [jobsRes, runsRes] = await Promise.all([api.listTrainingJobs(), api.listRuns()]);
      if (seq !== refreshSeq) return;
      const jobs = jobsRes.jobs || [];
      cachedJobs = jobs;
      const hasRunning = jobs.some((j) => j.status === 'running');
      isAnyRunning = hasRunning;

      const runningJob = jobs.find((j) => j.status === 'running');
      const displayJobs = runningJob ? [runningJob] : (jobs.length > 0 ? [jobs[jobs.length - 1]] : []);

      const prevLogScroll = new Map();
      jobsEl.querySelectorAll('.live-log-box').forEach((box) => {
        const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
        prevLogScroll.set(box.dataset.job, { top: box.scrollTop, atBottom });
      });

      if (displayJobs.length === 0) {
        jobsEl.innerHTML = '<p class="panel-sub">Siap melatih. Pilih "+ Buat Model Baru" atau model lama untuk di-resume di sebelah kiri.</p>';
      } else {
        jobsEl.innerHTML = `<div class="stack-col" style="gap: 10px;">${displayJobs.map((j) => {
          const pct = Number(j.progress_pct || 0);
          const stepTxt = `${Number(j.current_step || 0).toLocaleString('id-ID')} / ${Number(j.total_timesteps || 0).toLocaleString('id-ID')} (${pct}%)`;
          const logs = Array.isArray(j.logs) ? j.logs : [];
          const logText = logs.length > 0 ? logs.join('\n') : 'Menunggu output log dari proses training...';
          return `
            <div class="kv-panel">
              <div class="kv-panel-head" style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                <span>${escapeHtml(j.experiment_name)} · ${escapeHtml(j.status)} (${escapeHtml(j.phase || '-')})</span>
                ${j.status === 'running'
                  ? `<button class="btn btn-sm btn-danger btn-stop-job" data-job="${escapeHtml(j.job_id)}" type="button" ${j.stop_requested ? 'disabled' : ''}>${j.stop_requested ? 'Menghentikan...' : 'Hentikan Training'}</button>`
                  : `<span class="hint-inline">${escapeHtml(j.finished_at || j.started_at)}</span>`
                }
              </div>
              <div style="padding: 9px 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 11px;">
                  <span>Progres Step</span>
                  <span class="num">${stepTxt}</span>
                </div>
                <div class="progress-track"><div class="progress-fill" style="width: ${pct}%;"></div></div>
                ${j.error ? `<div style="color: var(--status-err); margin-top: 6px; font-size: 11px;">${escapeHtml(j.error)}</div>` : ''}
                <div class="live-log-head">
                  <span>Live Log (${logs.length} baris)</span>
                  <button class="inline-link-btn btn-copy-log" data-job="${escapeHtml(j.job_id)}" type="button">Salin Log</button>
                </div>
                <pre class="live-log-box" data-job="${escapeHtml(j.job_id)}">${escapeHtml(logText)}</pre>
              </div>
            </div>
          `;
        }).join('')}</div>`;

        jobsEl.querySelectorAll('.live-log-box').forEach((box) => {
          const prev = prevLogScroll.get(box.dataset.job);
          if (!prev || prev.atBottom) {
            box.scrollTop = box.scrollHeight;
          } else {
            box.scrollTop = prev.top;
          }
        });

        jobsEl.querySelectorAll('.btn-copy-log').forEach((btn) => {
          btn.addEventListener('click', async () => {
            const box = jobsEl.querySelector(`.live-log-box[data-job="${CSS.escape(btn.dataset.job)}"]`);
            if (!box) return;
            try {
              await navigator.clipboard.writeText(box.textContent || '');
              showToast('Log training disalin ke clipboard.');
            } catch {
              showToast('Tidak dapat menyalin otomatis; pilih teks log secara manual.');
            }
          });
        });

        jobsEl.querySelectorAll('.btn-stop-job').forEach((btn) => {
          btn.addEventListener('click', async () => {
            try {
              await api.stopTrainingJob(btn.dataset.job);
              showToast('Sinyal berhenti dikirim. Menunggu checkpoint aman...');
              refreshJobsAndRuns();
            } catch (err) {
              showToast(`Gagal stop: ${err.message}`);
            }
          });
        });
      }

      const allRuns = runsRes.runs || [];
      appState.runs = allRuns;
      openHistBtn.textContent = `Kelola Riwayat Run (${allRuns.length})`;

      if (runningJob && runningJob.run_dir) {
        const runningRid = String(runningJob.run_dir).split('/').pop();
        if (allRuns.some((r) => r.run_id === runningRid)) {
          selectedModelValue = runningRid;
        }
      }

      renderForm();
      await syncRightPanelOnly();
      if (histDialog.open) renderRunHistoryDialog();

      if (hasRunning && !pollTimer) {
        pollTimer = setInterval(refreshJobsAndRuns, 2000);
      } else if (!hasRunning && pollTimer) {
        stopPolling();
      }
    } catch (err) {
      showToast(`Gagal memuat runs: ${err.message}`);
    }
  }

  openHistBtn.addEventListener('click', () => {
    renderRunHistoryDialog();
    histDialog.showModal();
  });
  root.querySelector('#dlg-runs-close').addEventListener('click', () => histDialog.close());
  histDialog.addEventListener('click', (ev) => {
    if (ev.target === histDialog) histDialog.close();
  });

  cleanEmptyBtn.addEventListener('click', async () => {
    const emptyRuns = (appState.runs || []).filter((r) => !r.is_running && !r.final_summary && Number(r.eval_points_count || 0) <= 1);
    if (emptyRuns.length === 0) return;
    if (!window.confirm(`Hapus ${emptyRuns.length} folder run kosong (Step 0) dari disk?`)) return;
    let deleted = 0;
    for (const r of emptyRuns) {
      try {
        await api.deleteRun(r.run_id);
        deleted += 1;
      } catch {
        // skip
      }
    }
    showToast(`${deleted} folder run kosong dibersihkan.`);
    await refreshJobsAndRuns();
    renderRunHistoryDialog();
  });

  delRunBtn.addEventListener('click', async () => {
    if (isAnyRunning || !currentDisplayedRunId) return;
    if (!window.confirm(`Hapus model '${currentDisplayedRunId}' beserta checkpoint-nya secara permanen?`)) return;
    try {
      await api.deleteRun(currentDisplayedRunId);
      if (selectedModelValue === currentDisplayedRunId) selectedModelValue = '__new__';
      showToast('Model berhasil dihapus.');
      await refreshJobsAndRuns();
    } catch (err) {
      showToast(`Gagal hapus model: ${err.message}`, 4500);
    }
  });

  root.querySelector('#btn-refresh-runs').addEventListener('click', refreshJobsAndRuns);

  replayBtn.addEventListener('click', () => {
    if (!currentDisplayedRunId) return;
    appState.replayRunId = currentDisplayedRunId;
    location.hash = '#simulation';
  });

  root.querySelector('#btn-train-validate').addEventListener('click', async () => {
    try {
      const res = await api.validateTraining({
        training_profile: appState.trainingProfile,
        config: appState.worldConfig,
      });
      showToast(res.warnings?.length ? `Valid dengan catatan: ${res.warnings[0]}` : 'Konfigurasi training & gamma valid.');
    } catch (err) {
      showToast(`Validasi gagal: ${err.message}`, 5000);
    }
  });

  startBtn.addEventListener('click', async () => {
    if (isAnyRunning) return;
    const isResume = selectedModelValue !== '__new__';
    const resumeMeta = isResume ? (appState.runs || []).find((r) => r.run_id === selectedModelValue) : null;
    const ckpt = resumeMeta?.has_best_checkpoint ? 'best' : 'final';

    try {
      isAnyRunning = true;
      updateStartLockState();
      const payload = {
        training_profile: appState.trainingProfile,
        config: appState.worldConfig,
      };
      if (isResume && resumeMeta) {
        payload.resume_from_run_id = resumeMeta.run_id;
        payload.resume_checkpoint = ckpt;
      }
      const job = await api.startTraining(payload);
      chartManualScrolled = false;
      showToast(
        isResume && resumeMeta
          ? `Melanjutkan training '${resumeMeta.experiment_name || resumeMeta.run_id}' (${ckpt}).`
          : `Training model baru '${appState.trainingProfile.experiment_name}' dimulai (job ${job.job_id}).`,
      );
      await refreshJobsAndRuns();
    } catch (err) {
      isAnyRunning = false;
      updateStartLockState();
      showToast(`Gagal memulai training: ${err.message}`, 5000);
    }
  });

  return {
    async refresh() {
      if (appState.worldConfig) {
        if (!appState.activeMeta?.dirty && appState.activeMeta?.canonicalSha256) {
          activeCanonicalSha12 = String(appState.activeMeta.canonicalSha256).slice(0, 12);
        } else {
          try {
            const valRes = await api.validateConfig(appState.worldConfig);
            activeCanonicalSha12 = String(valRes.canonical_sha256 || '').slice(0, 12);
          } catch {
            activeCanonicalSha12 = String(appState.activeMeta?.canonicalSha256 || activeCanonicalSha12).slice(0, 12);
          }
        }
      }
      if (!userEditedExpName && appState.trainingProfile && selectedModelValue === '__new__') {
        const base = (appState.activeMeta?.name || 'agent').replace(/[^a-zA-Z0-9_-]/g, '_');
        appState.trainingProfile.experiment_name = `${base}_v1`;
      }
      await refreshJobsAndRuns();
      checkGammaSync();
    },
    onLeave() {
      stopPolling();
    },
  };
}
