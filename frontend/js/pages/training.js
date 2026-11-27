import { api } from '../api.js';
import { appState, showToast } from '../state.js';
import { escapeHtml, formatIdr, formatRunDisplayLabel } from '../utils.js';

const TRAINING_PRESETS = [
  { id: 'fast', title: 'Cepat · 50k', sub: 'Uji cepat (~1 mnt)', total: 50000, evalInt: 10000, surv: 25000 },
  { id: 'medium', title: 'Menengah · 300k', sub: 'Seimbang', total: 300000, evalInt: 25000, surv: 150000 },
  { id: 'full', title: 'Penuh · 3M', sub: 'Kualitas terbaik', total: 3000000, evalInt: 100000, surv: 500000 },
];

function svgNode(tag, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function renderEvalChart(svg, evalHistory) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!evalHistory || evalHistory.length === 0) {
    const txt = svgNode('text', { x: 270, y: 102, 'text-anchor': 'middle', fill: 'var(--ink-muted)', 'font-size': 12 });
    txt.textContent = 'Belum ada riwayat evaluasi untuk run ini.';
    svg.appendChild(txt);
    return;
  }

  const steps = evalHistory.map((r) => Number(r.timesteps ?? 0));
  const rewards = evalHistory.map((r) => Number(r.mean_episode_reward ?? 0));
  const minR = Math.min(...rewards);
  const maxR = Math.max(...rewards);
  const spanR = Math.max(1e-6, maxR - minR);
  const minS = Math.min(...steps);
  const maxS = Math.max(...steps);
  const spanS = Math.max(1, maxS - minS);

  const padL = 56;
  const padR = 18;
  const padT = 22;
  const padB = 34;
  const w = 540 - padL - padR;
  const h = 200 - padT - padB;

  [0, 0.5, 1].forEach((frac) => {
    const y = padT + (1 - frac) * h;
    const val = minR + frac * spanR;
    svg.appendChild(svgNode('line', { x1: padL, y1: y, x2: padL + w, y2: y, stroke: 'var(--border-subtle)', 'stroke-width': 1 }));
    const lbl = svgNode('text', { x: padL - 6, y: y + 3, 'text-anchor': 'end', fill: 'var(--ink-muted)', 'font-size': 10 });
    lbl.textContent = val.toFixed(0);
    svg.appendChild(lbl);
  });

  svg.appendChild(svgNode('line', { x1: padL, y1: padT, x2: padL, y2: padT + h, stroke: 'var(--border-strong)', 'stroke-width': 1 }));
  svg.appendChild(svgNode('line', { x1: padL, y1: padT + h, x2: padL + w, y2: padT + h, stroke: 'var(--border-strong)', 'stroke-width': 1 }));

  const coords = rewards.map((r, i) => ({
    x: padL + (rewards.length === 1 ? w / 2 : ((steps[i] - minS) / spanS) * w),
    y: padT + (1 - (r - minR) / spanR) * h,
  }));

  if (coords.length > 1) {
    svg.appendChild(svgNode('polyline', {
      fill: 'none',
      stroke: 'var(--ink-primary)',
      'stroke-width': 2,
      points: coords.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '),
    }));
  }
  coords.forEach((p) => {
    svg.appendChild(svgNode('circle', { cx: p.x.toFixed(1), cy: p.y.toFixed(1), r: 3.5, fill: 'var(--ink-primary)' }));
  });

  const yTitle = svgNode('text', { x: padL, y: 12, fill: 'var(--ink-secondary)', 'font-size': 10.5, 'font-weight': 600 });
  yTitle.textContent = 'Mean Episode Reward';
  svg.appendChild(yTitle);

  const xTitle = svgNode('text', { x: padL + w / 2, y: 194, 'text-anchor': 'middle', fill: 'var(--ink-secondary)', 'font-size': 10.5, 'font-weight': 600 });
  xTitle.textContent = `Timesteps (${minS.toLocaleString('id-ID')} – ${maxS.toLocaleString('id-ID')})`;
  svg.appendChild(xTitle);
}

export function initTrainingPage(root) {
  const formEl = root.querySelector('#training-form-container');
  const gammaWarnEl = root.querySelector('#train-gamma-warning');
  const jobsEl = root.querySelector('#training-jobs-list');
  const runsSel = root.querySelector('#runs-select');
  const delRunBtn = root.querySelector('#btn-delete-run');
  const runSummaryEl = root.querySelector('#run-summary-metrics');
  const runAllocEl = root.querySelector('#run-allocation-panel');
  const replayBtn = root.querySelector('#btn-replay-run');
  const evalSvg = root.querySelector('#run-eval-svg');

  let pollTimer = null;
  let advancedOpen = false;
  let userEditedExpName = false;

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

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

  function renderForm() {
    const tp = appState.trainingProfile;
    if (!tp) return;
    const curSteps = Number(tp.schedule.total_timesteps || 0);

    formEl.innerHTML = `
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
          <label>Nama Eksperimen</label>
          <input id="tr-exp-name" class="control-input code-id" value="${escapeHtml(tp.experiment_name)}">
        </div>
        <div class="field-group">
          <label>Total Timesteps</label>
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

    formEl.querySelectorAll('.preset-pill-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const preset = TRAINING_PRESETS.find((p) => p.id === btn.dataset.preset);
        if (!preset) return;
        tp.schedule.total_timesteps = preset.total;
        tp.schedule.eval_interval_steps = preset.evalInt;
        tp.schedule.survival_phase_timesteps = preset.surv;
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
    bind('#tr-timesteps', (el) => { tp.schedule.total_timesteps = Math.max(64, Math.round(Number(el.value || 64))); }, renderForm);
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
  }

  async function refreshJobsAndRuns() {
    try {
      const [jobsRes, runsRes] = await Promise.all([api.listTrainingJobs(), api.listRuns()]);
      const jobs = jobsRes.jobs || [];
      const hasRunning = jobs.some((j) => j.status === 'running');

      // Preserve scroll state per job log box before replacing DOM
      const prevLogScroll = new Map();
      jobsEl.querySelectorAll('.live-log-box').forEach((box) => {
        const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
        prevLogScroll.set(box.dataset.job, { top: box.scrollTop, atBottom });
      });

      if (jobs.length === 0) {
        jobsEl.innerHTML = '<p class="panel-sub">Belum ada sesi training aktif pada server ini.</p>';
      } else {
        jobsEl.innerHTML = `<div class="stack-col" style="gap: 10px;">${jobs.slice().reverse().map((j) => {
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

      const prevSelected = runsSel.value;
      appState.runs = runsRes.runs || [];
      runsSel.innerHTML = appState.runs.length === 0
        ? '<option value="">Belum ada run tersimpan</option>'
        : appState.runs.map((r) => `<option value="${escapeHtml(r.run_id)}">${escapeHtml(formatRunDisplayLabel(r))}</option>`).join('');

      if (appState.runs.some((r) => r.run_id === prevSelected)) {
        runsSel.value = prevSelected;
      }
      delRunBtn.hidden = !runsSel.value;
      if (runsSel.value) await loadRunDetail(runsSel.value);
      else {
        runSummaryEl.innerHTML = '';
        runAllocEl.innerHTML = '';
        replayBtn.hidden = true;
        renderEvalChart(evalSvg, []);
      }

      if (hasRunning && !pollTimer) {
        pollTimer = setInterval(refreshJobsAndRuns, 2000);
      } else if (!hasRunning && pollTimer) {
        stopPolling();
      }
    } catch (err) {
      showToast(`Gagal memuat runs: ${err.message}`);
    }
  }

  async function loadRunDetail(runId) {
    if (!runId) return;
    const detail = await api.getRunDetail(runId);
    const s = detail.final_summary;
    const epDays = appState.worldConfig?.environment?.episode_days || 30;
    replayBtn.hidden = !s;

    if (s) {
      const homeMeals = (s.mean_cook_fresh_count || 0) + (s.mean_meal_prep_count || 0) + (s.mean_reheat_prep_count || 0) + (s.mean_eat_bekal_count || 0);
      runSummaryEl.innerHTML = `
        <div class="metric-box"><div class="metric-box-label">Completion</div><div class="metric-box-value num">${(s.completion_rate * 100).toFixed(0)}%</div></div>
        <div class="metric-box"><div class="metric-box-label">Rata-rata Hari</div><div class="metric-box-value num">${s.mean_days_survived.toFixed(1)} / ${epDays}</div></div>
        <div class="metric-box"><div class="metric-box-label">Sisa Saldo</div><div class="metric-box-value num">${formatIdr(s.mean_final_balance_idr)}</div></div>
        <div class="metric-box"><div class="metric-box-label">Kehadiran</div><div class="metric-box-value num">${(s.mean_class_attendance_rate * 100).toFixed(0)}%</div></div>
      `;
      runAllocEl.innerHTML = `
        <div class="kv-panel-head">Alokasi Biaya &amp; Kebiasaan Agen per Bulan</div>
        <div class="kv-row"><span class="kv-key">Transportasi</span><span class="kv-val">${formatIdr(s.mean_spent_transport_idr)}</span></div>
        <div class="kv-row"><span class="kv-key">Belanja Bahan Dapur</span><span class="kv-val">${formatIdr(s.mean_spent_grocery_idr)}</span></div>
        <div class="kv-row"><span class="kv-key">Makan Luar / Warung</span><span class="kv-val">${formatIdr(s.mean_spent_external_food_idr)}</span></div>
        <div class="kv-row"><span class="kv-key">Kafe &amp; Jajan Impulsif</span><span class="kv-val">${formatIdr((s.mean_spent_cafe_idr || 0) + (s.mean_spent_minimarket_impulse_idr || 0))}</span></div>
        <div class="kv-row"><span class="kv-key">Masak / Bekal vs Hub / Kafe</span><span class="kv-val">${homeMeals.toFixed(1)}x masak · ${(s.mean_hub_or_leisure_count || 0).toFixed(1)}x hub/kafe</span></div>
      `;
    } else {
      runSummaryEl.innerHTML = '<p class="panel-sub">Run ini belum memiliki ringkasan evaluasi akhir.</p>';
      runAllocEl.innerHTML = '';
    }
    renderEvalChart(evalSvg, detail.eval_history || []);
  }

  runsSel.addEventListener('change', () => {
    delRunBtn.hidden = !runsSel.value;
    loadRunDetail(runsSel.value);
  });
  delRunBtn.addEventListener('click', async () => {
    const rid = runsSel.value;
    if (!rid) return;
    if (!window.confirm(`Hapus run '${rid}' beserta checkpoint-nya secara permanen?`)) return;
    try {
      await api.deleteRun(rid);
      showToast('Run berhasil dihapus.');
      await refreshJobsAndRuns();
    } catch (err) {
      showToast(`Gagal hapus run: ${err.message}`, 4500);
    }
  });
  root.querySelector('#btn-refresh-runs').addEventListener('click', refreshJobsAndRuns);

  replayBtn.addEventListener('click', () => {
    if (!runsSel.value) return;
    appState.replayRunId = runsSel.value;
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

  root.querySelector('#btn-train-start').addEventListener('click', async () => {
    try {
      const job = await api.startTraining({
        training_profile: appState.trainingProfile,
        config: appState.worldConfig,
      });
      showToast(`Training dimulai (job ${job.job_id}).`);
      await refreshJobsAndRuns();
    } catch (err) {
      showToast(`Gagal memulai training: ${err.message}`, 5000);
    }
  });

  return {
    refresh() {
      if (!userEditedExpName && appState.trainingProfile && appState.activeMeta?.name) {
        appState.trainingProfile.experiment_name = appState.activeMeta.name.replace(/[^a-zA-Z0-9_-]/g, '_');
      }
      renderForm();
      checkGammaSync();
      refreshJobsAndRuns();
    },
    onLeave() {
      stopPolling();
    },
  };
}
