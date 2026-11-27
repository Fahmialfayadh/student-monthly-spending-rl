# Student Life RL (30 days, 5 steps/day)

`StudentLifeEnv` + `MaskablePPO` agent managing one monthly budget (Rp 1,465,000 operational),
biometrics, cooking/meal-prep, transport, and class attendance over 150 steps.

## How to run

Use the existing environment (do not install new packages):

```bash
/cache/ml/.venv/bin/python -m jupyter nbconvert --to notebook --execute main.ipynb --inplace
```

Full training writes artifacts to `runs/` (git-ignored) and appends to `runs/experiments_summary.csv`.
Temporary check scripts live in `scripts/tmp/` (git-ignored) and must be deleted after use.

## Reward design (2026-11-26)

The terminal objective is unchanged: `+100` for surviving 30 days plus `+10` per Rp 100,000 left.
Guidance is added without dictating behavior:

- `min_hunger_to_eat: 20.0` blocks direct meals while full in action masks and execution;
  `meal_prep` and `pack_bekal` remain available.
- Three goal-progress observation features: `days_remaining`, `burn_rate_per_day_idr`,
  `budget_runway_days` (observation dim 35 -> 38).
- Potential-based shaping `gamma * Phi(s') - Phi(s)` on balance coverage, `enabled: true`
  under `rewards.budget_potential_shaping`. `gamma` must match `TRAIN_CONFIG.discount_factor_gamma`
  in `main.ipynb` (both `0.995`). Terminal potential is zero, so the shaping does not change the
  optimal policy; it only improves per-step credit assignment.
- No per-rupiah spending tax: the agent is free to find its own allocation of the budget.

## Meal-prep energy gates (2026-11-26)

`meal_prep_energy_cost` 38 -> 30, `meal_prep_min_energy` 55 -> 48,
`morning_meal_prep_extra_energy_cost` 22 -> 16 (morning total 60 -> 46), and
`morning_meal_prep_min_energy` 72 -> 62. Batch size, mood gate, cook_fresh costs, rewards,
and task goals are unchanged. Rationale: mid-day energy after class/commute is 50-70, so the old
gate made prep unreachable in exactly the window where it was needed. Prep still costs more than a
single fresh cook (30 vs 22) and more in the morning (46 vs 40).

## Training schedule (3M steps)

`TRAINING_SCHEDULE` in `main.ipynb`: 3,000,000 timesteps, evaluation every 100,000 steps,
`survival_phase_timesteps=500_000`. Checkpoint selection is lexicographic: maximize
`completion_rate` first; while incomplete, maximize `mean_days_survived`; once 100% complete,
maximize `mean_final_balance_idr`, then attendance, then mean reward. This makes `best_model.zip`
keep the surviving model and only replace it with a richer one. If completion is still below 100%
at 500k, training continues trying; 500k is only a milestone, not a phase switch. The phase label
stays `survive` until an evaluation actually reaches 100% completion, and the checkpoint rule keeps
survival dominant. The learning rate stays at 3e-4 until the first evaluation with 100% completion,
then decays to 3e-5 over the remaining steps (SB3's own progress resets at each `learn()` chunk).
`eval_history.csv` carries `training_phase`; `train_config.json` and `final_eval_summary.json`
record the 500k target and best checkpoint score.

A 3M run with immediate LR decay (run `20261126_115417`) diverged into a cafe-heavy local optimum
(~15 days, Rp 450k/spending on cafe) while identical-config runs with constant LR reached ~25 days
by 100k steps. The LR is therefore held constant until survival is achieved.

## Resume and offline evaluation

`StudentPPOAgent.train_with_logging(..., resume_from_run=RUN_DIR, resume_checkpoint="best")`
continues an interrupted run in the same directory. It restores model/optimizer weights,
VecNormalize statistics and training flags, `eval_history.csv`, best checkpoint score/timestep,
survival/LR-decay state, and rebuilds the custom LR schedule. The config SHA must match the
original run; use `RUN_DIR / "env_config_snapshot.yaml"` as `yaml_path` when resuming. The master
log row is replaced instead of duplicated.

Notebook cell `cell-recover-eval` is standalone: it does not depend on variables from the training
cell. It finds the latest run (or uses `RUN_DIR`), evaluates `best_model.zip` with seed 11000,
writes `manual_eval_summary.json` + CSVs, and defines `run_dir`, `eval_history_df`, `final_summary`,
`final_episodes_df`, and `best_episode_trace_df` for the cells below. Commented lines in that cell
show the resume call.

## Logged metrics

`days_survived = ceil(episode_steps / steps_per_day)` is the day reached when an episode ends
(30 for completed episodes). It is written per episode in `final_eval_episodes.csv`, aggregated
as `mean_days_survived` in `eval_history.csv` / `final_eval_summary.json`, and printed as `Hari`
in the training log. CPU training is not bit-reproducible across runs, so probe numbers vary.

## Quick-check log

Date: 2026-11-26 | Python: `/cache/ml/.venv/bin/python` | seed 42/7 as noted

| Check | Result |
| --- | --- |
| Action gates | eat blocked at hunger=0, allowed at hunger=50; prep/pack gated by stock |
| Observation | 38 features, three new finite goal-progress values |
| Potential shaping | finite per-step component (e.g. -0.05 at step 0) |
| Heuristic baseline (3 seeds) | 100% completion, 30.0/30 days, Rp 193,226 left, ~5.7 home-cooked meals/episode |
| MaskablePPO smoke (1,280 steps, seed 7) | runs without error; 20.5/30 days, finite reward/return |
| RL probe (25k steps, seed 42, 5 eval episodes) | 24.8/30 days (122.2/150 steps), 0% completion, Rp 30,991 left, 0 cooking, 41.8 warung meals |

The 25k-step probe is **not** conclusive: it shows better survival than the previous 83.6-step
run (day ~17) but no cooking yet. Run the full 3M-step schedule in `main.ipynb` before judging behavior.
