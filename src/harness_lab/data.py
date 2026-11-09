"""Fictional data for the Codex example: how a company uses an internal LLM platform.

One row per working day, department and model (January-June 2026). The numbers are made
up, but they hide a few stories a good analysis should find:

* ``modelo-frontera`` serves a small share of the requests but most of the cost.
* Ingeniería is the most expensive department: it uses the frontier model the most.
* Soporte doubles its usage from April (a support chatbot goes live).
* Marketing has an error spike in March (a broken integration).

    uv run python -m harness_lab.data workspace/datos/uso_llm.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEPARTMENTS = {  # name: (requests per working day, share of frontier-model requests)
    "Ingeniería": (900, 0.30),
    "Soporte": (600, 0.05),
    "Marketing": (350, 0.10),
    "Ventas": (300, 0.08),
    "Finanzas": (150, 0.12),
    "RR. HH.": (80, 0.05),
}
MODELS = {  # name: (EUR per 1M input tokens, EUR per 1M output tokens, mean latency in ms)
    "qwen-4b-local": (0.02, 0.02, 350),
    "qwen-27b": (0.30, 1.20, 900),
    "modelo-frontera": (3.00, 15.00, 2200),
}
TOKENS = {"qwen-4b-local": (600, 250), "qwen-27b": (1500, 500), "modelo-frontera": (4000, 1200)}


def make_llm_usage(path: Path | str | None = None, seed: int = 0) -> pd.DataFrame:
    """Build the dataset; if ``path`` is given, also write it (``.xlsx`` or ``.csv``)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-01-01", "2026-06-30")
    rows = []
    for day in days:
        for dept, (base, frontier_share) in DEPARTMENTS.items():
            volume = base * rng.uniform(0.8, 1.2)
            if dept == "Soporte" and day >= pd.Timestamp("2026-04-01"):
                volume *= 2.0  # support chatbot goes live
            shares = {
                "modelo-frontera": frontier_share,
                "qwen-27b": 0.35,
                "qwen-4b-local": 0.65 - frontier_share,
            }
            for model, (eur_in, eur_out, latency) in MODELS.items():
                requests = int(rng.poisson(volume * shares[model]))
                tok_in, tok_out = TOKENS[model]
                tokens_in = int(requests * tok_in * rng.uniform(0.7, 1.3))
                tokens_out = int(requests * tok_out * rng.uniform(0.7, 1.3))
                error_rate = 0.01
                if dept == "Marketing" and day.month == 3:
                    error_rate = 0.18  # broken integration
                rows.append(
                    {
                        "fecha": day.date(),
                        "departamento": dept,
                        "modelo": model,
                        "peticiones": requests,
                        "tokens_entrada": tokens_in,
                        "tokens_salida": tokens_out,
                        "coste_eur": round((tokens_in * eur_in + tokens_out * eur_out) / 1e6, 2),
                        "latencia_media_ms": int(latency * rng.uniform(0.85, 1.25)) if requests else 0,
                        "errores": int(rng.binomial(requests, error_rate)),
                    }
                )
    df = pd.DataFrame(rows)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False, sheet_name="uso")
    return df


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "workspace/datos/uso_llm.xlsx")
    data = make_llm_usage(out)
    print(f"{len(data)} rows -> {out}")
