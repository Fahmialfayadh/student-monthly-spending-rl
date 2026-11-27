import { api } from '../api.js';
import { appState, showToast } from '../state.js';

function renderEvalChart(svg, evalHistory) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!evalHistory || evalHistory.length === 0) {
    const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    txt.setAttribute('x', '260');
    txt.setAttribute('y', '88');
    txt.setAttribute('text-anchor', 'middle');
    txt.setAttribute('fill', 'var(--ink-muted)');
    txt.setAttribute('font-size', '11');
    txt.textContent = 'No evaluation history available for this run.';
    svg.appendChild(txt);
    return;
  }

  const rewards = evalHistory.map((r) => Number(r.mean_episode_reward ?? 0));
  const minR = Math.min(...rewards);
  const maxR = Math.max(...rewards);
  const span = Math.max(1e-6, maxR - minR);
  const padX = 44;
  const padY = 18;
  const w = 520 - padX * 2;
  const h = 170 - padY * 2;

  const pts = rewards.map((r, i) => {
    const x = padX + (rewards.length === 1 ? w / 2 : (i / (rewards.length - 1)) * w);
    const y = padY + (1 - (r - minR) / span) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  line.setAttribute('fill', 'none');
  line.setAttribute('stroke', 'var(--ink-primary)');
  line.setAttribute('stroke-width', '2');
  line.setAttribute('points', pts.join(' '));
  svg.appendChild(line);

  const axisLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  axisLabel.setAttribute('x', String(padX));
  axisLabel.setAttribute('y', '13');
  axisLabel.setAttribute('fill', 'var(--ink-secondary)');
  axisLabel.setAttribute('font-size', '10');
  axisLabel.setAttribute('font-family', 'var(--font-mono)');
  axisLabel.textContent = `Mean Episode Reward (${minR.toFixed(0)} .. ${maxR.toFixed(0)})`;
  svg.appendChild(axisLabel);
}

export function initTrainingPage(root) {
  const formEl = root.querySelector('#training-form-container');
  const jobsEl = root.querySelector('#training-jobs-list');
  const runsSel = root.querySelector('#runs-select');
  const runSummaryEl = root.querySelector('#run-summary-metrics');
  const evalSvg = root.querySelector('#run-eval-svg');

  function renderForm() {
    const tp = appState.trainingProfile;
    if (!tp) return;
    formEl.innerHTML = `
      <div class="field-row-2">
        <div class="field-group">
          <label>Experiment Name</label>
          <input id="tr-exp-name" class="control-input mono" value="${tp.experiment_name}">
        </div>
        <div class="field-group">
          <label>Total Timesteps</label>
          <input id="tr-timesteps" class="control-input mono" type="number" min="64" step="64" value="${tp.schedule.total_timesteps}">
        </div>
      </div>

      <div class="field-row-3">
        <div class="field-group" data-tooltip="Harus sama dengan rewards.budget_potential_shaping.gamma jika shaping aktif." data-tooltip-pos="top">
          <label class="tooltip-trigger">Discount Gamma</label>
          <input id="tr-gamma" class="control-input mono" type="number" step="0.001" min="0.5" max="1" value="${tp.ppo.discount_factor_gamma}">
        </div>
        <div class="field-group" data-tooltip="Faktor bias-variance pada Generalized Advantage Estimation." data-tooltip-pos="top">
          <label class="tooltip-trigger">GAE Lambda</label>
          <input id="tr-gae" class="control-input mono" type="number" step="0.01" min="0" max="1" value="${tp.ppo.gae_lambda}">
        </div>
        <div class="field-group">
          <label>Learning Rate</label>
          <input id="tr-lr" class="control-input mono" type="number" step="0.00005" value="${tp.ppo.learning_rate}">
        </div>
      </div>

      <div class="field-row-3">
        <div class="field-group">
          <label>Num Envs</label>
          <input id="tr-envs" class="control-input mono" type="number" min="1" max="32" value="${tp.ppo.num_environments}">
        </div>
        <div class="field-group">
          <label>Rollout Steps / Env</label>
          <input id="tr-rollout" class="control-input mono" type="number" min="16" value="${tp.ppo.rollout_steps_per_env}">
        </div>
        <div class="field-group">
          <label>Batch Size</label>
          <input id="tr-batch" class="control-input mono" type="number" min="8" value="${tp.ppo.batch_size}">
        </div>
      </div>

      <div class="field-row-3">
        <div class="field-group" data-tooltip="Batas KL divergence untuk early-stop update agar policy tidak kolaps." data-tooltip-pos="top">
          <label class="tooltip-trigger">Target KL</label>
          <input id="tr-kl" class="control-input mono" type="number" step="0.005" value="${tp.ppo.target_kl ?? ''}">
        </div>
        <div class="field-group" data-tooltip="Bobot bonus entropi untuk menjaga eksplorasi aksi." data-tooltip-pos="top">
          <label class="tooltip-trigger">Entropy Coef</label>
          <input id="tr-ent" class="control-input mono" type="number" step="0.005" value="${tp.ppo.entropy_coefficient}">
        </div>
        <div class="field-group">
          <label>Eval Interval Steps</label>
          <input id="tr-eval-int" class="control-input mono" type="number" min="64" value="${tp.schedule.eval_interval_steps}">
        </div>
      </div>

      <div style="border-top: 1px solid var(--border-subtle); padding-top: 10px;" class="stack-col">
        <label class="checkbox-row" data-tooltip="Warm-start policy dari demonstrasi heuristik sebelum fine-tuning PPO." data-tooltip-pos="top">
          <input id="tr-bc-enabled" type="checkbox" ${tp.bc.enabled ? 'checked' : ''}>
          <span class="tooltip-trigger">Enable Behavior Cloning Warm-Start</span>
        </label>
        <div class="field-row-3">
          <div class="field-group">
            <label>BC Episodes</label>
            <input id="tr-bc-ep" class="control-input mono" type="number" min="1" value="${tp.bc.num_episodes}">
          </div>
          <div class="field-group">
            <label>BC Epochs</label>
            <input id="tr-bc-epochs" class="control-input mono" type="number" min="1" value="${tp.bc.epochs}">
          </div>
          <div class="field-group">
            <label>BC Learning Rate</label>
            <input id="tr-bc-lr" class="control-input mono" type="number" step="0.0001" value="${tp.bc.learning_rate}">
          </div>
        </div>
      </div>
    `;

    const bind = (id, fn) => formEl.querySelector(id).addEventListener('change', (e) => fn(e.target));
    bind('#tr-exp-name', (el) => { tp.experiment_name = el.value; });
    bind('#tr-timesteps', (el) => { tp.schedule.total_timesteps = Number(el.value); });
    bind('#tr-gamma', (el) => { tp.ppo.discount_factor_gamma = Number(el.value); });
    bind('#tr-gae', (el) => { tp.ppo.gae_lambda = Number(el.value); });
    bind('#tr-lr', (el) => { tp.ppo.learning_rate = Number(el.value); });
    bind('#tr-envs', (el) => { tp.ppo.num_environments = Number(el.value); });
    bind('#tr-rollout', (el) => { tp.ppo.rollout_steps_per_env = Number(el.value); });
    bind('#tr-batch', (el) => { tp.ppo.batch_size = Number(el.value); });
    bind('#tr-kl', (el) => { tp.ppo.target_kl = el.value ? Number(el.value) : null; });
    bind('#tr-ent', (el) => { tp.ppo.entropy_coefficient = Number(el.value); });
    bind('#tr-eval-int', (el) => { tp.schedule.eval_interval_steps = Number(el.value); });
    bind('#tr-bc-enabled', (el) => { tp.bc.enabled = el.checked; });
    bind('#tr-bc-ep', (el) => { tp.bc.num_episodes = Number(el.value); });
    bind('#tr-bc-epochs', (el) => { tp.bc.epochs = Number(el.value); });
    bind('#tr-bc-lr', (el) => { tp.bc.learning_rate = Number(el.value); });
  }

  async function refreshJobsAndRuns() {
    try {
      const [jobsRes, runsRes] = await Promise.all([api.listTrainingJobs(), api.listRuns()]);
      const jobs = jobsRes.jobs || [];
      jobsEl.innerHTML = jobs.length === 0
        ? '<p class="panel-sub">No training jobs started in this server session.</p>'
        : jobs.map((j) => `
            <div style="padding: 8px 0; border-bottom: 1px solid var(--border-subtle);" class="mono">
              <strong>${j.experiment_name}</strong> · ${j.status} (${j.started_at})
              ${j.error ? `<div style="color: var(--status-err);">${j.error}</div>` : ''}
            </div>
          `).join('');

      appState.runs = runsRes.runs || [];
      runsSel.innerHTML = appState.runs.length === 0
        ? '<option value="">No saved runs</option>'
        : appState.runs.map((r) => `<option value="${r.run_id}">${r.run_id}</option>`).join('');
      if (appState.runs.length > 0) await loadRunDetail(runsSel.value);
    } catch (err) {
      showToast(`Refresh failed: ${err.message}`);
    }
  }

  async function loadRunDetail(runId) {
    if (!runId) return;
    const detail = await api.getRunDetail(runId);
    const s = detail.final_summary;
    runSummaryEl.innerHTML = s
      ? `
        <div class="metric-box"><div class="metric-box-label">Completion</div><div class="metric-box-value mono">${(s.completion_rate * 100).toFixed(0)}%</div></div>
        <div class="metric-box"><div class="metric-box-label">Days</div><div class="metric-box-value mono">${s.mean_days_survived.toFixed(1)}/30</div></div>
        <div class="metric-box"><div class="metric-box-label">Balance</div><div class="metric-box-value mono">Rp ${Math.round(s.mean_final_balance_idr).toLocaleString()}</div></div>
        <div class="metric-box"><div class="metric-box-label">Attendance</div><div class="metric-box-value mono">${(s.mean_class_attendance_rate * 100).toFixed(0)}%</div></div>
      `
      : '<p class="panel-sub">Run has no final_eval_summary.json yet.</p>';
    renderEvalChart(evalSvg, detail.eval_history || []);
  }

  runsSel.addEventListener('change', () => loadRunDetail(runsSel.value));
  root.querySelector('#btn-refresh-runs').addEventListener('click', refreshJobsAndRuns);

  root.querySelector('#btn-train-validate').addEventListener('click', async () => {
    try {
      const res = await api.validateTraining({
        training_profile: appState.trainingProfile,
        config: appState.worldConfig,
      });
      showToast(res.warnings?.length ? `Valid with warning: ${res.warnings[0]}` : 'Training profile & gamma valid.');
    } catch (err) {
      showToast(`Validation error: ${err.message}`, 4500);
    }
  });

  root.querySelector('#btn-train-start').addEventListener('click', async () => {
    try {
      const job = await api.startTraining({
        training_profile: appState.trainingProfile,
        config: appState.worldConfig,
      });
      showToast(`Started training job ${job.job_id}`);
      await refreshJobsAndRuns();
    } catch (err) {
      showToast(`Start failed: ${err.message}`, 4500);
    }
  });

  return {
    refresh() {
      renderForm();
      refreshJobsAndRuns();
    },
  };
}
