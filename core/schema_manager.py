"""
Schema Manager
Gestisce lo schema del dataset: colonne di default + colonne custom aggiunte dall'utente.
Salva e carica la configurazione da config/schema.json.
"""

import json
import os
from typing import Any

CONFIG_PATH = "config/schema.json"

# ─────────────────────────────────────────────
# SCHEMA DI DEFAULT
# ─────────────────────────────────────────────

DEFAULT_COLUMNS = {
    # Identificatori (non usati dal modello, solo per display)
    "employee_id":                    {"type": "str",   "role": "id",      "desc": "ID univoco dipendente"},
    "nome":                           {"type": "str",   "role": "pii",     "desc": "Nome"},
    "cognome":                        {"type": "str",   "role": "pii",     "desc": "Cognome"},
    "data_assunzione":                {"type": "str",   "role": "pii",     "desc": "Data assunzione (YYYY-MM-DD)"},

    # Feature usate dal modello
    "eta":                            {"type": "int",   "role": "feature", "min": 18,   "max": 70,  "desc": "Età in anni"},
    "dipartimento":                   {"type": "str",   "role": "feature", "categorical": True,      "desc": "Dipartimento"},
    "ruolo":                          {"type": "str",   "role": "feature", "categorical": True,      "desc": "Ruolo aziendale"},
    "anni_in_azienda":                {"type": "float", "role": "feature", "min": 0,    "max": 50,  "desc": "Anni dall'assunzione"},
    "stipendio_annuo":                {"type": "int",   "role": "feature", "min": 15000,"max": 500000, "desc": "RAL in euro"},
    "num_promozioni":                 {"type": "int",   "role": "feature", "min": 0,    "max": 20,  "desc": "Promozioni ricevute"},
    "anni_senza_promozione":          {"type": "float", "role": "feature", "min": 0,    "max": 30,  "desc": "Anni dall'ultima promozione"},
    "crescita_stipendio_percentuale": {"type": "float", "role": "feature", "min": -20,  "max": 50,  "desc": "Crescita % stipendio annua"},
    "ore_settimanali_medie":          {"type": "float", "role": "feature", "min": 20,   "max": 80,  "desc": "Ore lavorate/settimana"},
    "giorni_ultimo_checkin_hr":       {"type": "int",   "role": "feature", "min": 0,    "max": 730, "desc": "Giorni fa dall'ultimo check-in HR"},
    "manager_score":                  {"type": "float", "role": "feature", "min": 1,    "max": 10,  "desc": "Score feedback manager (1-10)"},
    "enps_score":                     {"type": "float", "role": "feature", "min": 1,    "max": 10,  "desc": "Employee NPS (1-10)"},
    "giorni_assenza_anno":            {"type": "int",   "role": "feature", "min": 0,    "max": 365, "desc": "Giorni assenza nell'anno"},
    "num_progetti_anno":              {"type": "int",   "role": "feature", "min": 0,    "max": 50,  "desc": "Progetti seguiti nell'anno"},
    "ferie_residue":                  {"type": "int",   "role": "feature", "min": 0,    "max": 60,  "desc": "Giorni ferie residui"},

    # Target
    "churn_label":                    {"type": "int",   "role": "target",  "desc": "0 = resta, 1 = a rischio"},
}

COLUMN_TYPES   = ["int", "float", "str", "bool"]
COLUMN_ROLES   = ["feature", "id", "pii", "target", "ignore"]


# ─────────────────────────────────────────────
# LOAD / SAVE
# ─────────────────────────────────────────────

def load_schema() -> dict:
    """Carica lo schema da file. Se non esiste o e corrotto, restituisce il default."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            pass
    return {"columns": DEFAULT_COLUMNS.copy(), "custom_columns": {}}


def save_schema(schema: dict):
    """Salva lo schema su file."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


def reset_schema():
    """Ripristina lo schema di default."""
    save_schema({"columns": DEFAULT_COLUMNS.copy(), "custom_columns": {}})


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_feature_columns(schema: dict) -> list:
    """Restituisce solo le colonne con role='feature'."""
    all_cols = {**schema.get("columns", {}), **schema.get("custom_columns", {})}
    return [col for col, meta in all_cols.items() if meta.get("role") == "feature"]


def get_numeric_features(schema: dict) -> list:
    """Restituisce le feature numeriche (int/float)."""
    all_cols = {**schema.get("columns", {}), **schema.get("custom_columns", {})}
    return [
        col for col, meta in all_cols.items()
        if meta.get("role") == "feature" and meta.get("type") in ("int", "float")
        and not meta.get("categorical")
    ]


def get_categorical_features(schema: dict) -> list:
    """Restituisce le feature categoriche."""
    all_cols = {**schema.get("columns", {}), **schema.get("custom_columns", {})}
    return [
        col for col, meta in all_cols.items()
        if meta.get("role") == "feature" and meta.get("categorical")
    ]


def get_pii_columns(schema: dict) -> list:
    """Restituisce le colonne PII (da anonimizzare)."""
    all_cols = {**schema.get("columns", {}), **schema.get("custom_columns", {})}
    return [col for col, meta in all_cols.items() if meta.get("role") == "pii"]


def get_required_columns(schema: dict) -> list:
    """Restituisce tutte le colonne richieste (id + pii + feature + target)."""
    all_cols = {**schema.get("columns", {}), **schema.get("custom_columns", {})}
    return [col for col, meta in all_cols.items() if meta.get("role") != "ignore"]


def add_custom_column(schema: dict, name: str, col_type: str, role: str,
                      desc: str, min_val: Any = None, max_val: Any = None,
                      categorical: bool = False) -> dict:
    """Aggiunge una colonna custom allo schema."""
    schema = schema.copy()
    schema.setdefault("custom_columns", {})
    schema["custom_columns"][name] = {
        "type": col_type,
        "role": role,
        "desc": desc,
        "custom": True,
        **({"min": min_val, "max": max_val} if min_val is not None else {}),
        **({"categorical": True} if categorical else {}),
    }
    return schema


def remove_custom_column(schema: dict, name: str) -> dict:
    """Rimuove una colonna custom dallo schema."""
    schema = schema.copy()
    schema.setdefault("custom_columns", {})
    schema["custom_columns"].pop(name, None)
    return schema


def generate_template_csv(schema: dict) -> str:
    """Genera il contenuto CSV del template scaricabile."""
    import pandas as pd
    import io

    all_cols = {**schema.get("columns", {}), **schema.get("custom_columns", {})}
    required = [c for c, m in all_cols.items() if m.get("role") != "ignore"]

    # Riga di esempio
    example = {
        "employee_id": "EMP0001",
        "nome": "Mario", "cognome": "Rossi",
        "data_assunzione": "2021-06-15",
        "eta": 32, "dipartimento": "Engineering", "ruolo": "Senior Dev",
        "anni_in_azienda": 4.5, "stipendio_annuo": 52000,
        "num_promozioni": 1, "anni_senza_promozione": 2.5,
        "crescita_stipendio_percentuale": 3.5, "ore_settimanali_medie": 46.0,
        "giorni_ultimo_checkin_hr": 75, "manager_score": 7.5,
        "enps_score": 6.0, "giorni_assenza_anno": 3,
        "num_progetti_anno": 5, "ferie_residue": 8, "churn_label": 0,
    }

    # Aggiungi valori di esempio per colonne custom
    for col, meta in schema.get("custom_columns", {}).items():
        if meta.get("type") in ("int", "float"):
            min_v = meta.get("min", 0)
            max_v = meta.get("max", 100)
            example[col] = round((min_v + max_v) / 2, 1)
        elif meta.get("type") == "bool":
            example[col] = False
        else:
            example[col] = "esempio"

    row = {col: example.get(col, "") for col in required}
    df = pd.DataFrame([row])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
