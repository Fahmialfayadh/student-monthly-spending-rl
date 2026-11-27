from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from src.env import StudentLifeEnv
from src.env.pantry import PantryManager

__all__ = [
    "BehaviorCloningPretrainer",
    "default_balanced_expert_policy",
]

_HEAD_ORDER = (
    "activity",
    "route",
    "meal_mode",
    "cook_combo",
    "bring_bekal",
    "grocery_bundle",
)

_PREFERRED_COMBOS = (
    PantryManager.COOK_COMBOS.index((1, 1, 0, 1)),  # 1 rice + 1 egg + 1 tempe
    PantryManager.COOK_COMBOS.index((1, 0, 1, 0)),  # 1 rice + 1 chicken
    PantryManager.COOK_COMBOS.index((1, 1, 1, 0)),  # 1 rice + 1 egg + 1 chicken
    PantryManager.COOK_COMBOS.index((1, 0, 1, 1)),  # 1 rice + 1 chicken + 1 tempe
    PantryManager.COOK_COMBOS.index((1, 2, 0, 0)),  # 1 rice + 2 egg
    PantryManager.COOK_COMBOS.index((1, 0, 0, 2)),  # 1 rice + 2 tempe
    PantryManager.COOK_COMBOS.index((1, 1, 0, 0)),  # 1 rice + 1 egg
    PantryManager.COOK_COMBOS.index((1, 0, 0, 1)),  # 1 rice + 1 tempe
)


def _compute_mask_offsets(env: StudentLifeEnv) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for head_name, dim_len in zip(_HEAD_ORDER, env.action_space.nvec):
        offsets[head_name] = cursor
        cursor += int(dim_len)
    return offsets


def _is_allowed(
    mask: np.ndarray, offsets: dict[str, int], head: str, index: int | None
) -> bool:
    if index is None or index < 0:
        return False
    return bool(mask[offsets[head] + index])


def _first_valid(mask: np.ndarray, offsets: dict[str, int], head: str, size: int) -> int:
    start = offsets[head]
    for idx in range(size):
        if bool(mask[start + idx]):
            return idx
    raise RuntimeError(f"All-False action mask pada head '{head}'.")


def _find_activity_index(env: StudentLifeEnv, handler: str) -> int | None:
    for idx, aid in enumerate(env.activity_names):
        if env.activity_specs[aid].handler == handler:
            return idx
    return None


def default_balanced_expert_policy(env: StudentLifeEnv) -> np.ndarray:
    """
    Demonstrator heuristik seimbang berbasis WorldGraph & handler aktivitas untuk
    warm-start Behavior Cloning pada topologi graf mana pun.
    """
    mask = env.action_masks()
    offsets = _compute_mask_offsets(env)
    _, _, step_in_day, day_name = env.academics.get_time_context(env.current_step)
    is_night = step_in_day == (env.steps_per_day - 1)
    class_now = env.academics.is_class_scheduled(day_name, step_in_day)
    at_home = env.current_node_id == env.home_node_id
    at_outside = not at_home
    days_left = (env.total_steps - env.current_step) / env.steps_per_day

    act_rest = _find_activity_index(env, "rest") or 0
    act_attend = _find_activity_index(env, "attend_schedule")
    act_grocery = _find_activity_index(env, "grocery_shop")
    act_leisure = _find_activity_index(env, "leisure_reset")
    act_hub = _find_activity_index(env, "hub_stay")
    act_study = _find_activity_index(env, "study")
    act_wait = _find_activity_index(env, "wait")

    rice = env.pantry.get_ingredient_total_units("rice")
    egg = env.pantry.get_ingredient_total_units("egg")
    chicken = env.pantry.get_ingredient_total_units("chicken")
    tempe = env.pantry.get_ingredient_total_units("tempe")
    prep = env.pantry.get_meal_prep_total_portions()

    # 1. Cek kebutuhan belanja dapur yang pas (hindari bahan basi)
    need_rice = 1 if (rice <= 2 and days_left >= 5.0) else 0
    need_egg = 1 if (egg <= 2 and days_left >= 6.0) else 0
    need_chicken = 1 if (chicken == 0 and tempe <= 2 and days_left >= 7.0) else 0
    need_tempe = 1 if (tempe == 0 and chicken <= 2 and days_left >= 5.0) else 0
    desired_bundle_tuple = (need_rice, need_egg, need_chicken, need_tempe)

    grocery_bundle = _first_valid(
        mask, offsets, "grocery_bundle", len(env.pantry.grocery_bundles)
    )
    wants_grocery = False
    if any(desired_bundle_tuple):
        candidate_idx = PantryManager.GROCERY_BUNDLES.index(desired_bundle_tuple)
        if _is_allowed(mask, offsets, "grocery_bundle", candidate_idx):
            grocery_bundle = candidate_idx
            wants_grocery = True
        else:
            rice_only_idx = PantryManager.GROCERY_BUNDLES.index((1, 0, 0, 0))
            if need_rice and _is_allowed(mask, offsets, "grocery_bundle", rice_only_idx):
                grocery_bundle = rice_only_idx
                wants_grocery = True

    # 2. Pilih aktivitas utama berdasarkan handler & kemampuan graf
    activity = act_rest
    if class_now and _is_allowed(mask, offsets, "activity", act_attend):
        activity = int(act_attend)
    elif (
        at_home
        and (not is_night)
        and wants_grocery
        and _is_allowed(mask, offsets, "activity", act_grocery)
    ):
        activity = int(act_grocery)
    elif (
        env.cafe_craving > 46.0
        and env.balance_idr > 180_000.0
        and _is_allowed(mask, offsets, "activity", act_leisure)
    ):
        activity = int(act_leisure)
    elif (
        (not is_night)
        and env.biometrics.hub_visits_today == 0
        and (
            (at_outside and (env.academic_stress > 12.0 or env.cafe_craving > 34.0))
            or (at_home and (env.academic_stress > 28.0 or env.cafe_craving > 38.0))
        )
        and _is_allowed(mask, offsets, "activity", act_hub)
    ):
        activity = int(act_hub)
    elif (
        at_outside
        and (not is_night)
        and env.academics.has_later_class_today(day_name, step_in_day)
    ):
        if _is_allowed(mask, offsets, "activity", act_hub):
            activity = int(act_hub)
        elif _is_allowed(mask, offsets, "activity", act_study):
            activity = int(act_study)
        elif _is_allowed(mask, offsets, "activity", act_wait):
            activity = int(act_wait)
    elif at_home and (not is_night) and env.academic_stress > 18.0 and _is_allowed(mask, offsets, "activity", act_study):
        activity = int(act_study)

    if not _is_allowed(mask, offsets, "activity", activity):
        activity = _first_valid(mask, offsets, "activity", len(env.activity_names))

    # 3. Pilih preferensi rute: hemat (cheapest) kecuali hujan atau panas terik saat lelah
    fastest_idx = env.route_names.index("fastest") if "fastest" in env.route_names else 0
    cheapest_idx = env.route_names.index("cheapest") if "cheapest" in env.route_names else 0
    prefer_fastest = env.current_is_raining or (step_in_day == 2 and env.energy < 45.0)
    if prefer_fastest and _is_allowed(mask, offsets, "route", fastest_idx):
        route = fastest_idx
    elif _is_allowed(mask, offsets, "route", cheapest_idx):
        route = cheapest_idx
    else:
        route = _first_valid(mask, offsets, "route", len(env.route_names))

    # 4. Bawa bekal ringan jika beraktivitas ke luar rumah dan ada stok meal_prep
    chosen_handler = env.activity_specs[env.activity_names[activity]].handler
    leaving_for_campus_or_hub = chosen_handler in ("attend_schedule", "hub_stay")
    bring_bekal = 1 if (_is_allowed(mask, offsets, "bring_bekal", 1) and leaving_for_campus_or_hub) else 0

    # 5. Pilih komposisi masak (prioritaskan variasi anti-boredom & bahan cepat basi)
    cook_combo = _first_valid(mask, offsets, "cook_combo", len(env.pantry.cook_combos))
    valid_combos = [
        idx for idx in _PREFERRED_COMBOS if _is_allowed(mask, offsets, "cook_combo", idx)
    ]
    if valid_combos:
        tempe_life = env.pantry.get_min_ingredient_shelf_life("tempe") or 999
        chicken_life = env.pantry.get_min_ingredient_shelf_life("chicken") or 999

        def _combo_priority(combo_idx: int) -> tuple[int, int, float]:
            qty = env.pantry.decode_cook_combo_dict(combo_idx)
            _, _, _, sig = env.pantry.calculate_custom_meal_profile(qty)
            repeats = env.pantry.get_recent_meal_count(sig)
            urgency_bonus = 0
            if qty.get("tempe", 0) > 0 and tempe_life <= 10:
                urgency_bonus -= 2
            if qty.get("chicken", 0) > 0 and chicken_life <= 12:
                urgency_bonus -= 2
            cost = env.pantry.compute_combo_ingredient_value_idr(qty)
            return (repeats, urgency_bonus, cost)

        valid_combos.sort(key=_combo_priority)
        cook_combo = valid_combos[0]

    # 6. Pilih meal_mode (meal_prep saat stok prep kosong, masak segar / bekal saat lapar)
    current_has_kitchen = env.world_graph.get_node(env.current_node_id).has_kitchen
    meal_mode = 0
    if (
        (at_home or current_has_kitchen or is_night)
        and prep == 0
        and days_left >= 3.0
        and env.energy >= 52.0
        and _is_allowed(mask, offsets, "meal_mode", 2)
    ):
        prep_candidates = [
            idx
            for idx in valid_combos
            if env.pantry.max_portions_possible(env.pantry.decode_cook_combo_dict(idx)) >= 3
        ]
        if prep_candidates:
            cook_combo = prep_candidates[0]
            meal_mode = 2
    elif env.hunger >= 40.0:
        if _is_allowed(mask, offsets, "meal_mode", 3) and (
            at_outside or env.pantry.has_non_repeating_meal_prep()
        ):
            meal_mode = 3
        elif (at_home or current_has_kitchen or is_night) and _is_allowed(mask, offsets, "meal_mode", 1):
            meal_mode = 1
        elif _is_allowed(mask, offsets, "meal_mode", 3):
            meal_mode = 3
        elif _is_allowed(mask, offsets, "meal_mode", 4):
            meal_mode = 4
        elif _is_allowed(mask, offsets, "meal_mode", 5):
            meal_mode = 5

    return np.asarray(
        [activity, route, meal_mode, cook_combo, bring_bekal, grocery_bundle],
        dtype=np.int64,
    )


class BehaviorCloningPretrainer:
    """
    Pelatih warm-start Behavior Cloning (Supervised Learning) langsung pada
    policy (Actor + Critic) dan statistik VecNormalize milik StudentPPOAgent,
    sebelum dilepas ke fine-tuning MaskablePPO murni.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def collect_demonstrations(
        self,
        expert_fn: Callable[[StudentLifeEnv], np.ndarray] = default_balanced_expert_policy,
        num_episodes: int = 60,
        seed: int = 5000,
    ) -> dict[str, Any]:
        """Mengumpulkan transisi (obs, action, mask, discounted return) dari env jujur (kulkas 0)."""
        env = StudentLifeEnv(yaml_path=self.agent.yaml_path, exploring_starts_pantry=False)
        gamma = float(self.agent.discount_factor_gamma)

        all_obs: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        all_returns: list[float] = []
        all_running_rets: list[float] = []
        episode_rewards: list[float] = []
        episode_balances: list[float] = []
        episode_completions: list[int] = []

        for ep_idx in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep_idx)
            done = False
            ep_obs: list[np.ndarray] = []
            ep_actions: list[np.ndarray] = []
            ep_masks: list[np.ndarray] = []
            ep_rewards: list[float] = []
            running_ret = 0.0
            ep_running_rets: list[float] = []
            info: dict[str, Any] = {}

            while not done:
                mask = env.action_masks()
                action = np.asarray(expert_fn(env), dtype=np.int64)
                next_obs, reward, done, _, info = env.step(action)

                ep_obs.append(obs)
                ep_actions.append(action)
                ep_masks.append(mask)
                ep_rewards.append(float(reward))

                running_ret = running_ret * gamma + float(reward)
                ep_running_rets.append(running_ret)
                obs = next_obs

            # Hitung discounted return G_t mundur dari belakang untuk target Critic
            discounted = np.zeros(len(ep_rewards), dtype=np.float32)
            g = 0.0
            for t in range(len(ep_rewards) - 1, -1, -1):
                g = ep_rewards[t] + gamma * g
                discounted[t] = g

            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)
            all_masks.extend(ep_masks)
            all_returns.extend(discounted.tolist())
            all_running_rets.extend(ep_running_rets)

            episode_rewards.append(float(sum(ep_rewards)))
            episode_balances.append(float(env.balance_idr))
            term_reason = info.get("step_events", {}).get("termination_reason", "")
            episode_completions.append(int(term_reason == "completed_30_days"))

        return {
            "observations": np.asarray(all_obs, dtype=np.float32),
            "actions": np.asarray(all_actions, dtype=np.int64),
            "action_masks": np.asarray(all_masks, dtype=bool),
            "returns": np.asarray(all_returns, dtype=np.float32),
            "running_returns": np.asarray(all_running_rets, dtype=np.float32),
            "demo_mean_reward": float(np.mean(episode_rewards)),
            "demo_completion_rate": float(np.mean(episode_completions)),
            "demo_mean_final_balance_idr": float(np.mean(episode_balances)),
        }

    def warmup_vec_normalize(
        self, observations: np.ndarray, running_returns: np.ndarray
    ) -> None:
        """Memanaskan statistik obs_rms & ret_rms pada VecNormalize milik StudentPPOAgent."""
        vec_env = self.agent.training_env
        vec_env.obs_rms.update(observations)
        vec_env.ret_rms.update(running_returns)

    def pretrain(
        self,
        expert_fn: Callable[[StudentLifeEnv], np.ndarray] = default_balanced_expert_policy,
        num_episodes: int = 60,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        entropy_coef: float = 0.05,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        demo_seed: int = 5000,
        verbose: bool = True,
        log_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Melatih Actor (masked Cross-Entropy + entropy bonus) dan Critic (MSE return)
        pada agent.model.policy menggunakan data demonstrasi, lalu menyinkronkan VecNormalize.
        """
        if log_callback is not None:
            log_callback(
                f"[BC Pretrain] Mengumpulkan {int(num_episodes)} episode demonstrasi heuristik..."
            )
        demo_data = self.collect_demonstrations(
            expert_fn=expert_fn,
            num_episodes=num_episodes,
            seed=demo_seed,
        )
        self.warmup_vec_normalize(demo_data["observations"], demo_data["running_returns"])
        if log_callback is not None:
            log_callback(
                f"[BC Pretrain] Demo siap: {int(demo_data['observations'].shape[0]):,} transisi | "
                f"mean_reward={demo_data['demo_mean_reward']:.2f} | "
                f"complete={demo_data['demo_completion_rate'] * 100:.0f}% | "
                f"sisa=Rp {demo_data['demo_mean_final_balance_idr']:,.0f}"
            )

        vec_env = self.agent.training_env
        norm_obs = vec_env.normalize_obs(demo_data["observations"]).astype(np.float32)
        ret_std = float(np.sqrt(vec_env.ret_rms.var + vec_env.epsilon))
        norm_returns = (demo_data["returns"] / ret_std).astype(np.float32)

        policy = self.agent.model.policy
        device = policy.device
        policy.set_training_mode(True)

        obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=device)
        act_tensor = torch.as_tensor(demo_data["actions"], dtype=torch.long, device=device)
        mask_tensor = torch.as_tensor(demo_data["action_masks"], dtype=torch.bool, device=device)
        ret_tensor = torch.as_tensor(norm_returns, dtype=torch.float32, device=device)

        dataset_size = obs_tensor.shape[0]
        optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
        rng = np.random.default_rng(self.agent.seed)

        history: list[dict[str, float]] = []
        for epoch_idx in range(1, int(epochs) + 1):
            indices = rng.permutation(dataset_size)
            epoch_actor_losses: list[float] = []
            epoch_critic_losses: list[float] = []
            epoch_entropies: list[float] = []

            for start in range(0, dataset_size, int(batch_size)):
                batch_idx = indices[start : start + int(batch_size)]
                obs_b = obs_tensor[batch_idx]
                act_b = act_tensor[batch_idx]
                mask_b = mask_tensor[batch_idx]
                ret_b = ret_tensor[batch_idx]

                if self.agent.use_action_masking:
                    values, log_prob, entropy = policy.evaluate_actions(
                        obs_b, act_b, action_masks=mask_b
                    )
                else:
                    values, log_prob, entropy = policy.evaluate_actions(obs_b, act_b)

                actor_loss = -log_prob.mean() - float(entropy_coef) * entropy.mean()
                critic_loss = nn.functional.mse_loss(values.flatten(), ret_b)
                total_loss = actor_loss + float(vf_coef) * critic_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), float(max_grad_norm))
                optimizer.step()

                epoch_actor_losses.append(float(actor_loss.item()))
                epoch_critic_losses.append(float(critic_loss.item()))
                epoch_entropies.append(float(entropy.mean().item()))

            row = {
                "epoch": float(epoch_idx),
                "actor_loss": float(np.mean(epoch_actor_losses)),
                "critic_loss": float(np.mean(epoch_critic_losses)),
                "entropy": float(np.mean(epoch_entropies)),
            }
            history.append(row)
            msg = (
                f"[BC Pretrain] Epoch {epoch_idx:>2d}/{epochs} | "
                f"actor_loss={row['actor_loss']:.4f} | "
                f"critic_loss={row['critic_loss']:.4f} | "
                f"entropy={row['entropy']:.3f}"
            )
            if log_callback is not None:
                log_callback(msg)
            if verbose and (epoch_idx == 1 or epoch_idx % 5 == 0 or epoch_idx == int(epochs)):
                print(msg)

        policy.set_training_mode(False)
        return {
            "num_transitions": int(dataset_size),
            "demo_mean_reward": demo_data["demo_mean_reward"],
            "demo_completion_rate": demo_data["demo_completion_rate"],
            "demo_mean_final_balance_idr": demo_data["demo_mean_final_balance_idr"],
            "history": history,
        }
