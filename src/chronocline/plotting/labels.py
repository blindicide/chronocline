"""Small English/Russian plotting translation layer."""

LABELS = {
    "en": {
        "step": "Quantizer step Δ / σ",
        "capacity": "Capacity (bits/symbol)",
        "matrix": "Output bin",
        "input": "Input delay",
    },
    "ru": {
        "step": "Шаг квантования Δ / σ",
        "capacity": "Пропускная способность (бит/символ)",
        "matrix": "Выходной интервал",
        "input": "Входная задержка",
    },
}


def label(key: str, locale: str = "en") -> str:
    """Translate generic plot labels without translating mathematical symbols."""
    return LABELS[locale][key]
