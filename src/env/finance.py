class FinanceManager:
    """
    Mengelola satu saldo operasional bulanan tunggal (balance_idr) dan mencatat
    rincian pengeluaran per kategori (transport, belanja dapur, makan luar,
    kafe, dan jajan impulsif) agar agen RL bebas menemukan alokasi terbaik.
    """

    def __init__(self, finance_config: dict):
        self.finance_config = finance_config
        self.monthly_budget_idr = float(finance_config["monthly_budget"])
        self.balance_idr = 0.0
        self.spent_by_category_idr: dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        self.balance_idr = float(self.finance_config["initial_operational_budget"])
        self.spent_by_category_idr = {
            "transport": 0.0,
            "grocery": 0.0,
            "external_food": 0.0,
            "cafe": 0.0,
            "minimarket_impulse": 0.0,
        }

    def is_balance_low(self) -> bool:
        low_balance_threshold = float(self.finance_config["alerts"]["low_balance_threshold"])
        return self.balance_idr < low_balance_threshold

    def can_afford(self, amount_idr: float) -> bool:
        return self.balance_idr >= amount_idr

    def pay(self, amount_idr: float, category: str) -> bool:
        """Membayar sejumlah biaya dari saldo operasional dan mencatatnya pada kategori terkait."""
        if amount_idr <= 0.0:
            return True
        if self.balance_idr < amount_idr:
            return False
        self.balance_idr -= amount_idr
        self.spent_by_category_idr[category] = self.spent_by_category_idr.get(category, 0.0) + amount_idr
        return True

    def drain_remaining_for_emergency(self, category: str = "transport") -> float:
        """Menguras sisa saldo saat kondisi darurat pulang malam dengan dana tidak cukup."""
        drained_amount = self.balance_idr
        if drained_amount > 0.0:
            self.spent_by_category_idr[category] = (
                self.spent_by_category_idr.get(category, 0.0) + drained_amount
            )
            self.balance_idr = 0.0
        return drained_amount

    def get_spending_summary(self) -> dict[str, float]:
        return dict(self.spent_by_category_idr)
