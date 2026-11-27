class PantryManager:
    """
    Mengelola inventaris dapur berbasis Smart-FIFO (bahan mentah, meal prep kulkas,
    serta kotak bekal), valuasi stok dapur (IDR), racikan lauk mandiri,
    kejenuhan menu berbasis window, dan penuaan stok.
    """

    INGREDIENT_NAMES = ("rice", "egg", "chicken", "tempe")

    COOK_COMBOS: tuple[tuple[int, int, int, int], ...] = (
        (1, 0, 0, 0),  # plain_rice
        (1, 1, 0, 0),  # rice + 1 egg
        (1, 2, 0, 0),  # rice + 2 egg
        (1, 0, 1, 0),  # rice + 1 chicken
        (1, 0, 2, 0),  # rice + 2 chicken
        (1, 0, 0, 1),  # rice + 1 tempe
        (1, 0, 0, 2),  # rice + 2 tempe
        (1, 1, 1, 0),  # rice + 1 egg + 1 chicken
        (1, 1, 0, 1),  # rice + 1 egg + 1 tempe
        (1, 0, 1, 1),  # rice + 1 chicken + 1 tempe
        (0, 1, 0, 0),  # 1 egg
        (0, 2, 0, 0),  # 2 egg
        (0, 0, 1, 0),  # 1 chicken
        (0, 0, 2, 0),  # 2 chicken
        (0, 0, 0, 1),  # 1 tempe
        (0, 1, 0, 1),  # 1 egg + 1 tempe
        (0, 1, 1, 0),  # 1 egg + 1 chicken
        (0, 0, 1, 1),  # 1 chicken + 1 tempe
    )

    GROCERY_BUNDLES: tuple[tuple[int, int, int, int], ...] = tuple(
        (
            (mask >> 0) & 1,
            (mask >> 1) & 1,
            (mask >> 2) & 1,
            (mask >> 3) & 1,
        )
        for mask in range(1, 16)
    )

    def __init__(self, pantry_config: dict, external_food_ids: tuple[str, ...] = ()):
        self.pantry_config = pantry_config
        self.ingredients_config = pantry_config["ingredients"]
        self.composition_rules = pantry_config["meal_composition_rules"]
        self.cooking_gates = pantry_config["cooking_costs_and_gates"]
        self.boredom_config = pantry_config["boredom_mechanics"]
        self.external_food_ids = tuple(external_food_ids)

        self.unit_prices_idr: dict[str, float] = {
            name: float(self.ingredients_config[name]["pack_price_idr"])
            / float(self.ingredients_config[name]["units_per_pack"])
            for name in self.INGREDIENT_NAMES
        }

        self.ingredient_batches: dict[str, list[dict[str, int]]] = {}
        self.meal_prep_batches: list[dict[str, float | int | str]] = []
        self.packed_lunch_portions: list[dict[str, float | str]] = []
        self.last_meal_signature: str | None = None
        self.recent_meal_signatures: list[str] = []
        self.consecutive_same_meal_count = 0
        self.total_spoiled_units = 0

        self.reset()

    def reset(self) -> None:
        self.ingredient_batches = {name: [] for name in self.INGREDIENT_NAMES}
        self.meal_prep_batches = []
        self.packed_lunch_portions = []
        self.last_meal_signature = None
        self.recent_meal_signatures = []
        self.consecutive_same_meal_count = 0
        self.total_spoiled_units = 0

        initial_pantry = self.pantry_config["initial_pantry"]
        for ingredient_name in self.INGREDIENT_NAMES:
            initial_quantity = int(initial_pantry.get(ingredient_name, 0))
            if initial_quantity > 0:
                shelf_life_steps = int(self.ingredients_config[ingredient_name]["shelf_life_steps"])
                self.ingredient_batches[ingredient_name].append({
                    "quantity": initial_quantity,
                    "remaining_shelf_life_steps": shelf_life_steps,
                })

        initial_meal_prep_portions = int(initial_pantry.get("meal_prep", 0))
        if initial_meal_prep_portions > 0:
            default_cost = (
                self.unit_prices_idr["rice"]
                + self.unit_prices_idr["egg"]
                + self.unit_prices_idr["tempe"]
            )
            self.meal_prep_batches.append({
                "portions": initial_meal_prep_portions,
                "remaining_shelf_life_steps": int(self.cooking_gates["meal_prep_shelf_life_steps"]),
                "hunger_delta": -50.0,
                "energy_delta": 19.0,
                "mood_delta": 6.0,
                "meal_signature": "egg+tempe",
                "portion_value_idr": default_cost,
            })

    @classmethod
    def decode_cook_combo(cls, combo_index: int) -> dict[str, int]:
        rice, egg, chicken, tempe = cls.COOK_COMBOS[int(combo_index)]
        return {"rice": rice, "egg": egg, "chicken": chicken, "tempe": tempe}

    @classmethod
    def decode_grocery_bundle(cls, bundle_index: int) -> dict[str, bool]:
        rice, egg, chicken, tempe = cls.GROCERY_BUNDLES[int(bundle_index)]
        return {
            "rice": bool(rice),
            "egg": bool(egg),
            "chicken": bool(chicken),
            "tempe": bool(tempe),
        }

    def get_grocery_bundle_cost_idr(self, bundle_index: int) -> float:
        flags = self.decode_grocery_bundle(bundle_index)
        return sum(
            float(self.ingredients_config[name]["pack_price_idr"])
            for name in self.INGREDIENT_NAMES
            if flags[name]
        )

    def compute_combo_ingredient_value_idr(self, quantities: dict[str, int]) -> float:
        return sum(
            self.unit_prices_idr[name] * int(quantities[name]) for name in self.INGREDIENT_NAMES
        )

    def get_inventory_value_idr(self) -> float:
        raw_value = sum(
            self.get_ingredient_total_units(name) * self.unit_prices_idr[name]
            for name in self.INGREDIENT_NAMES
        )
        prep_value = sum(
            int(batch["portions"]) * float(batch.get("portion_value_idr", 0.0))
            for batch in self.meal_prep_batches
        )
        bekal_value = sum(
            float(portion.get("portion_value_idr", 0.0)) for portion in self.packed_lunch_portions
        )
        return float(raw_value + prep_value + bekal_value)

    def get_ingredient_total_units(self, ingredient_name: str) -> int:
        return sum(batch["quantity"] for batch in self.ingredient_batches[ingredient_name])

    def get_meal_prep_total_portions(self) -> int:
        return sum(int(batch["portions"]) for batch in self.meal_prep_batches)

    def get_packed_lunch_total_portions(self) -> int:
        return len(self.packed_lunch_portions)

    def get_total_raw_units(self) -> int:
        return sum(self.get_ingredient_total_units(name) for name in self.INGREDIENT_NAMES)

    def get_min_ingredient_shelf_life(self, ingredient_name: str) -> int:
        batches = self.ingredient_batches[ingredient_name]
        if not batches:
            return 0
        return min(batch["remaining_shelf_life_steps"] for batch in batches)

    def get_min_meal_prep_shelf_life(self) -> int:
        if not self.meal_prep_batches:
            return 0
        return min(int(batch["remaining_shelf_life_steps"]) for batch in self.meal_prep_batches)

    def get_stock_summary(self) -> dict[str, int]:
        return {
            "rice": self.get_ingredient_total_units("rice"),
            "egg": self.get_ingredient_total_units("egg"),
            "chicken": self.get_ingredient_total_units("chicken"),
            "tempe": self.get_ingredient_total_units("tempe"),
            "meal_prep": self.get_meal_prep_total_portions(),
            "packed_lunch": self.get_packed_lunch_total_portions(),
        }

    def get_recent_meal_count(self, meal_signature: str) -> int:
        return sum(1 for sig in self.recent_meal_signatures if sig == meal_signature)

    def has_non_repeating_meal_prep(self) -> bool:
        if not self.meal_prep_batches:
            return False
        return any(
            self.get_recent_meal_count(str(batch["meal_signature"])) == 0
            for batch in self.meal_prep_batches
        )

    def has_non_repeating_packed_lunch(self) -> bool:
        if not self.packed_lunch_portions:
            return False
        return any(
            self.get_recent_meal_count(str(portion["meal_signature"])) == 0
            for portion in self.packed_lunch_portions
        )

    def get_last_meal_flag_names(self) -> list[str]:
        return [
            "last_meal_had_egg",
            "last_meal_had_chicken",
            "last_meal_had_tempe",
            *[f"last_meal_was_{fid}" for fid in self.external_food_ids],
            "last_meal_was_plain_rice",
        ]

    def get_last_meal_one_hot_flags(self) -> list[float]:
        """Mengembalikan flag biner identitas menu terakhir secara dinamis sesuai config."""
        signature = self.last_meal_signature or ""
        parts = set(signature.split("+")) if signature else set()
        flags = [
            1.0 if "egg" in parts else 0.0,
            1.0 if "chicken" in parts else 0.0,
            1.0 if "tempe" in parts else 0.0,
        ]
        for food_id in self.external_food_ids:
            flags.append(1.0 if signature == food_id else 0.0)
        flags.append(1.0 if signature == "plain_rice" else 0.0)
        return flags

    def max_portions_possible(self, requested_quantities: dict[str, int]) -> int:
        limits: list[int] = []
        for name in self.INGREDIENT_NAMES:
            qty = int(requested_quantities[name])
            if qty > 0:
                limits.append(self.get_ingredient_total_units(name) // qty)
        return min(limits) if limits else 0

    def has_enough_ingredients(
        self, requested_quantities: dict[str, int], batch_multiplier: int = 1
    ) -> bool:
        return self.max_portions_possible(requested_quantities) >= batch_multiplier

    def add_grocery_pack(self, ingredient_name: str) -> None:
        ingredient_profile = self.ingredients_config[ingredient_name]
        self.ingredient_batches[ingredient_name].append({
            "quantity": int(ingredient_profile["units_per_pack"]),
            "remaining_shelf_life_steps": int(ingredient_profile["shelf_life_steps"]),
        })

    def consume_ingredients_fifo(
        self, requested_quantities: dict[str, int], batch_multiplier: int = 1
    ) -> None:
        for ingredient_name, quantity in requested_quantities.items():
            units_to_consume = int(quantity) * int(batch_multiplier)
            if units_to_consume <= 0:
                continue

            remaining_to_consume = units_to_consume
            updated_batches: list[dict[str, int]] = []
            for batch in self.ingredient_batches[ingredient_name]:
                if remaining_to_consume <= 0:
                    updated_batches.append(batch)
                    continue
                if batch["quantity"] <= remaining_to_consume:
                    remaining_to_consume -= batch["quantity"]
                else:
                    batch["quantity"] -= remaining_to_consume
                    remaining_to_consume = 0
                    updated_batches.append(batch)
            self.ingredient_batches[ingredient_name] = updated_batches

    def store_meal_prep_batch(
        self,
        portions: int,
        hunger_delta: float,
        energy_delta: float,
        mood_delta: float,
        meal_signature: str,
        portion_value_idr: float,
    ) -> None:
        if portions <= 0:
            return
        self.meal_prep_batches.append({
            "portions": int(portions),
            "remaining_shelf_life_steps": int(self.cooking_gates["meal_prep_shelf_life_steps"]),
            "hunger_delta": float(hunger_delta),
            "energy_delta": float(energy_delta),
            "mood_delta": float(mood_delta),
            "meal_signature": str(meal_signature),
            "portion_value_idr": float(portion_value_idr),
        })

    def pop_one_meal_prep_portion_smart_fifo(
        self, avoid_signature: str | None = None
    ) -> dict[str, float | str] | None:
        if not self.meal_prep_batches:
            return None

        selected_idx = 0
        if avoid_signature is not None:
            for idx, batch in enumerate(self.meal_prep_batches):
                if str(batch["meal_signature"]) != avoid_signature:
                    selected_idx = idx
                    break

        selected_batch = self.meal_prep_batches[selected_idx]
        portion_profile: dict[str, float | str] = {
            "hunger_delta": float(selected_batch["hunger_delta"]),
            "energy_delta": float(selected_batch["energy_delta"]),
            "mood_delta": float(selected_batch["mood_delta"]),
            "meal_signature": str(selected_batch["meal_signature"]),
            "portion_value_idr": float(selected_batch.get("portion_value_idr", 0.0)),
        }
        selected_batch["portions"] = int(selected_batch["portions"]) - 1
        if int(selected_batch["portions"]) <= 0:
            self.meal_prep_batches.pop(selected_idx)
        return portion_profile

    def pack_one_bekal_from_prep(self) -> dict[str, float | str] | None:
        max_portions = int(self.cooking_gates["max_packed_lunch_portions"])
        if len(self.packed_lunch_portions) >= max_portions:
            return None
        avoid_sig = (
            str(self.packed_lunch_portions[-1]["meal_signature"])
            if self.packed_lunch_portions
            else self.last_meal_signature
        )
        portion_profile = self.pop_one_meal_prep_portion_smart_fifo(avoid_signature=avoid_sig)
        if portion_profile is None:
            return None
        self.packed_lunch_portions.append(portion_profile)
        return portion_profile

    def pop_one_packed_lunch_portion(
        self, avoid_signature: str | None = None
    ) -> dict[str, float | str] | None:
        if not self.packed_lunch_portions:
            return None
        selected_idx = 0
        if avoid_signature is not None:
            for idx, portion in enumerate(self.packed_lunch_portions):
                if str(portion["meal_signature"]) != avoid_signature:
                    selected_idx = idx
                    break
        return self.packed_lunch_portions.pop(selected_idx)

    def spoil_uneaten_packed_lunch_end_of_day(self) -> int:
        spoiled_bekal_count = len(self.packed_lunch_portions)
        if spoiled_bekal_count > 0:
            self.packed_lunch_portions = []
            self.total_spoiled_units += spoiled_bekal_count
        return spoiled_bekal_count

    def age_and_spoil_batches(self) -> int:
        spoiled_units_this_step = 0

        for ingredient_name in self.INGREDIENT_NAMES:
            surviving_batches: list[dict[str, int]] = []
            for batch in self.ingredient_batches[ingredient_name]:
                batch["remaining_shelf_life_steps"] -= 1
                if batch["remaining_shelf_life_steps"] <= 0:
                    spoiled_units_this_step += batch["quantity"]
                else:
                    surviving_batches.append(batch)
            self.ingredient_batches[ingredient_name] = surviving_batches

        surviving_prep_batches: list[dict[str, float | int | str]] = []
        for prep_batch in self.meal_prep_batches:
            prep_batch["remaining_shelf_life_steps"] = (
                int(prep_batch["remaining_shelf_life_steps"]) - 1
            )
            if int(prep_batch["remaining_shelf_life_steps"]) <= 0:
                spoiled_units_this_step += int(prep_batch["portions"])
            else:
                surviving_prep_batches.append(prep_batch)
        self.meal_prep_batches = surviving_prep_batches

        self.total_spoiled_units += spoiled_units_this_step
        return spoiled_units_this_step

    def calculate_custom_meal_profile(
        self, ingredient_quantities: dict[str, int]
    ) -> tuple[float, float, float, str]:
        total_hunger_delta = 0.0
        total_energy_gain = 0.0
        total_mood_delta = 0.0
        active_side_dishes: list[str] = []

        for ingredient_name in self.INGREDIENT_NAMES:
            quantity = int(ingredient_quantities[ingredient_name])
            if quantity <= 0:
                continue
            nutrition = self.ingredients_config[ingredient_name]["nutrition_per_unit"]
            total_hunger_delta += float(nutrition["hunger"]) * quantity
            total_energy_gain += float(nutrition["energy"]) * quantity
            total_mood_delta += float(nutrition["mood"]) * quantity
            if ingredient_name != "rice":
                active_side_dishes.append(ingredient_name)

        if len(active_side_dishes) == 0 and int(ingredient_quantities["rice"]) > 0:
            total_mood_delta += float(self.composition_rules["plain_rice_only_mood_penalty"])
            meal_signature = "plain_rice"
        else:
            if len(active_side_dishes) >= 2:
                total_mood_delta += float(self.composition_rules["multi_lauk_mood_bonus"])
            meal_signature = "+".join(sorted(active_side_dishes))

        max_hunger_reduction = float(self.composition_rules["max_meal_hunger_reduction"])
        max_energy_gain = float(self.composition_rules["max_meal_energy_gain"])

        total_hunger_delta = max(-max_hunger_reduction, total_hunger_delta)
        total_energy_gain = min(max_energy_gain, total_energy_gain)
        return total_hunger_delta, total_energy_gain, total_mood_delta, meal_signature

    def compute_boredom_mood_penalty(self, meal_signature: str) -> float:
        window_size = int(self.boredom_config.get("recent_meal_window", 4))
        penalty_schedule = self.boredom_config["recent_repeat_mood_penalty"]
        consecutive_extra = float(self.boredom_config.get("consecutive_extra_mood_penalty", 0.0))

        recent_repeats = self.get_recent_meal_count(meal_signature)
        is_consecutive = meal_signature == self.last_meal_signature
        if is_consecutive:
            self.consecutive_same_meal_count += 1
        else:
            self.last_meal_signature = meal_signature
            self.consecutive_same_meal_count = 0

        self.recent_meal_signatures.append(meal_signature)
        if len(self.recent_meal_signatures) > window_size:
            self.recent_meal_signatures = self.recent_meal_signatures[-window_size:]

        penalty_index = min(recent_repeats, len(penalty_schedule) - 1)
        boredom_mood_penalty = float(penalty_schedule[penalty_index])
        if is_consecutive and recent_repeats > 0:
            boredom_mood_penalty += consecutive_extra

        if bool(self.boredom_config.get("apply_sensitivity_to_penalty", True)):
            boredom_mood_penalty *= float(self.boredom_config["sensitivity"])

        return boredom_mood_penalty
