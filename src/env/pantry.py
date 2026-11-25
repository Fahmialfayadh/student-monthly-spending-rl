class PantryManager:
    """
    Mengelola inventaris dapur berbasis FIFO (bahan mentah, meal prep kulkas,
    serta kotak bekal yang dibawa ke kampus), perhitungan nutrisi racikan lauk
    mandiri, kejenuhan menu (boredom), dan penuaan/pembusukan stok.
    """

    INGREDIENT_NAMES = ("rice", "egg", "chicken", "tempe")

    def __init__(self, pantry_config: dict):
        self.pantry_config = pantry_config
        self.ingredients_config = pantry_config["ingredients"]
        self.composition_rules = pantry_config["meal_composition_rules"]
        self.cooking_gates = pantry_config["cooking_costs_and_gates"]
        self.boredom_config = pantry_config["boredom_mechanics"]

        self.ingredient_batches: dict[str, list[dict[str, int]]] = {}
        self.meal_prep_batches: list[dict[str, float | int | str]] = []
        self.packed_lunch_portions: list[dict[str, float | str]] = []
        self.last_meal_signature: str | None = None
        self.consecutive_same_meal_count = 0
        self.total_spoiled_units = 0

        self.reset()

    def reset(self) -> None:
        self.ingredient_batches = {name: [] for name in self.INGREDIENT_NAMES}
        self.meal_prep_batches = []
        self.packed_lunch_portions = []
        self.last_meal_signature = None
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
            self.meal_prep_batches.append({
                "portions": initial_meal_prep_portions,
                "remaining_shelf_life_steps": int(self.cooking_gates["meal_prep_shelf_life_steps"]),
                "hunger_delta": -52.0,
                "energy_delta": 20.0,
                "mood_delta": 4.5,
                "meal_signature": "egg+tempe",
            })

    # ------------------------------------------------------------------
    # Cek Stok & Daya Simpan
    # ------------------------------------------------------------------
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

    def get_last_meal_one_hot_flags(self) -> list[float]:
        """Mengembalikan 7 flag biner identitas menu terakhir agar observasi 100% Markovian."""
        signature = self.last_meal_signature or ""
        parts = set(signature.split("+")) if signature else set()
        return [
            1.0 if "egg" in parts else 0.0,
            1.0 if "chicken" in parts else 0.0,
            1.0 if "tempe" in parts else 0.0,
            1.0 if signature == "warung_hemat" else 0.0,
            1.0 if signature == "warung_variasi" else 0.0,
            1.0 if signature == "craving_snack" else 0.0,
            1.0 if signature == "plain_rice" else 0.0,
        ]

    def has_enough_ingredients(
        self, requested_quantities: dict[str, int], batch_multiplier: int = 1
    ) -> bool:
        return all(
            self.get_ingredient_total_units(name) >= requested_quantities[name] * batch_multiplier
            for name in self.INGREDIENT_NAMES
        )

    # ------------------------------------------------------------------
    # Tambah, Konsumsi FIFO, Bawa Bekal, & Penuaan Stok
    # ------------------------------------------------------------------
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
            units_to_consume = quantity * batch_multiplier
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
        hunger_delta: float,
        energy_delta: float,
        mood_delta: float,
        meal_signature: str,
    ) -> None:
        self.meal_prep_batches.append({
            "portions": int(self.cooking_gates["meal_prep_batch_size"]),
            "remaining_shelf_life_steps": int(self.cooking_gates["meal_prep_shelf_life_steps"]),
            "hunger_delta": hunger_delta,
            "energy_delta": energy_delta,
            "mood_delta": mood_delta,
            "meal_signature": meal_signature,
        })

    def pop_one_meal_prep_portion_fifo(self) -> dict[str, float | str] | None:
        if not self.meal_prep_batches:
            return None

        oldest_batch = self.meal_prep_batches[0]
        portion_profile: dict[str, float | str] = {
            "hunger_delta": float(oldest_batch["hunger_delta"]),
            "energy_delta": float(oldest_batch["energy_delta"]),
            "mood_delta": float(oldest_batch["mood_delta"]),
            "meal_signature": str(oldest_batch["meal_signature"]),
        }
        oldest_batch["portions"] = int(oldest_batch["portions"]) - 1
        if int(oldest_batch["portions"]) <= 0:
            self.meal_prep_batches.pop(0)
        return portion_profile

    def pack_one_bekal_from_prep(self) -> dict[str, float | str] | None:
        """Memindahkan 1 porsi meal_prep dari kulkas ke kotak bekal di tas."""
        max_portions = int(self.cooking_gates["max_packed_lunch_portions"])
        if len(self.packed_lunch_portions) >= max_portions:
            return None
        portion_profile = self.pop_one_meal_prep_portion_fifo()
        if portion_profile is None:
            return None
        self.packed_lunch_portions.append(portion_profile)
        return portion_profile

    def pop_one_packed_lunch_portion(self) -> dict[str, float | str] | None:
        if not self.packed_lunch_portions:
            return None
        return self.packed_lunch_portions.pop(0)

    def spoil_uneaten_packed_lunch_end_of_day(self) -> int:
        """Bekal yang dibawa seharian dan tidak dimakan hingga akhir hari menjadi basi."""
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

    # ------------------------------------------------------------------
    # Racikan Lauk Mandiri & Mekanik Kejenuhan (Boredom)
    # ------------------------------------------------------------------
    def calculate_custom_meal_profile(
        self, ingredient_quantities: dict[str, int]
    ) -> tuple[float, float, float, str]:
        total_hunger_delta = 0.0
        total_energy_gain = 0.0
        total_mood_delta = 0.0
        active_side_dishes: list[str] = []

        for ingredient_name in self.INGREDIENT_NAMES:
            quantity = ingredient_quantities[ingredient_name]
            if quantity <= 0:
                continue
            nutrition = self.ingredients_config[ingredient_name]["nutrition_per_unit"]
            total_hunger_delta += float(nutrition["hunger"]) * quantity
            total_energy_gain += float(nutrition["energy"]) * quantity
            total_mood_delta += float(nutrition["mood"]) * quantity
            if ingredient_name != "rice":
                active_side_dishes.append(ingredient_name)

        if len(active_side_dishes) == 0 and ingredient_quantities["rice"] > 0:
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
        penalty_schedule = self.boredom_config["consecutive_meal_mood_penalty"]

        if meal_signature == self.last_meal_signature:
            self.consecutive_same_meal_count += 1
        else:
            self.last_meal_signature = meal_signature
            self.consecutive_same_meal_count = 0

        penalty_index = min(self.consecutive_same_meal_count, len(penalty_schedule) - 1)
        boredom_mood_penalty = float(penalty_schedule[penalty_index])

        if bool(self.boredom_config.get("apply_sensitivity_to_penalty", False)):
            boredom_mood_penalty *= float(self.boredom_config["sensitivity"])

        return boredom_mood_penalty
