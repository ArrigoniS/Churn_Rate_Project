"""
HR Dataset Validator
Valida un CSV custom caricato dall'utente prima di passarlo al modello.
"""

import pandas as pd
import numpy as np
from typing import Tuple

# ─────────────────────────────────────────────
# SCHEMA ATTESO
# ─────────────────────────────────────────────
REQUIRED_COLUMNS = {
    "employee_id":                   {"type": "str",   "desc": "ID univoco del dipendente (es. EMP0001)"},
    "nome":                          {"type": "str",   "desc": "Nome del dipendente"},
    "cognome":                       {"type": "str",   "desc": "Cognome del dipendente"},
    "eta":                           {"type": "int",   "min": 18, "max": 70, "desc": "Età in anni"},
    "dipartimento":                  {"type": "str",   "desc": "Dipartimento (es. Engineering, Sales...)"},
    "ruolo":                         {"type": "str",   "desc": "Ruolo aziendale"},
    "anni_in_azienda":               {"type": "float", "min": 0, "max": 50, "desc": "Anni dall'assunzione"},
    "stipendio_annuo":               {"type": "int",   "min": 15000, "max": 500000, "desc": "RAL in euro"},
    "num_promozioni":                {"type": "int",   "min": 0, "max": 20, "desc": "Numero di promozioni ricevute"},
    "anni_senza_promozione":         {"type": "float", "min": 0, "max": 30, "desc": "Anni dall'ultima promozione"},
    "crescita_stipendio_percentuale":{"type": "float", "min": -20, "max": 50, "desc": "Crescita media % stipendio annua"},
    "ore_settimanali_medie":         {"type": "float", "min": 20, "max": 80, "desc": "Ore lavorate a settimana (media)"},
    "giorni_ultimo_checkin_hr":      {"type": "int",   "min": 0, "max": 730, "desc": "Giorni fa dall'ultimo check-in HR"},
    "manager_score":                 {"type": "float", "min": 1, "max": 10, "desc": "Score feedback manager (1-10)"},
    "enps_score":                    {"type": "float", "min": 1, "max": 10, "desc": "Score engagement aziendale (1-10)"},
    "giorni_assenza_anno":           {"type": "int",   "min": 0, "max": 365, "desc": "Giorni di assenza nell'anno"},
    "num_progetti_anno":             {"type": "int",   "min": 0, "max": 50, "desc": "Numero progetti seguiti nell'anno"},
    "ferie_residue":                 {"type": "int",   "min": 0, "max": 60, "desc": "Giorni di ferie residui"},
}

OPTIONAL_COLUMNS = ["data_assunzione", "churn_label", "churn_probability"]


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, list, list]:
    """
    Valida un DataFrame caricato dall'utente.
    
    Returns:
        (is_valid, errors, warnings)
        - is_valid: True se il dataset può essere usato dal modello
        - errors: lista di errori bloccanti
        - warnings: lista di avvisi non bloccanti
    """
    errors = []
    warnings = []

    # 1. Colonne mancanti
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Colonne mancanti: {', '.join(missing_cols)}")

    if errors:
        return False, errors, warnings

    # 2. Duplicati su employee_id
    dupes = df["employee_id"].duplicated().sum()
    if dupes > 0:
        errors.append(f"Trovati {dupes} employee_id duplicati — devono essere univoci")

    # 3. Valori nulli
    null_counts = df[list(REQUIRED_COLUMNS.keys())].isnull().sum()
    for col, count in null_counts[null_counts > 0].items():
        pct = count / len(df) * 100
        if pct > 20:
            errors.append(f"'{col}': troppi valori nulli ({count} righe, {pct:.0f}%) — max 20%")
        else:
            warnings.append(f"'{col}': {count} valori nulli ({pct:.0f}%) — verranno imputati con la mediana")

    # 4. Range check per colonne numeriche
    for col, schema in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            continue
        if schema["type"] in ("int", "float") and "min" in schema:
            col_data = pd.to_numeric(df[col], errors="coerce").dropna()
            out_of_range = ((col_data < schema["min"]) | (col_data > schema["max"])).sum()
            if out_of_range > 0:
                pct = out_of_range / len(df) * 100
                if pct > 10:
                    errors.append(
                        f"'{col}': {out_of_range} valori fuori range [{schema['min']}, {schema['max']}] ({pct:.0f}%)"
                    )
                else:
                    warnings.append(
                        f"'{col}': {out_of_range} valori fuori range [{schema['min']}, {schema['max']}] — saranno clippati"
                    )

    # 5. Dimensione minima
    if len(df) < 10:
        errors.append(f"Dataset troppo piccolo: {len(df)} righe (minimo: 10)")
    elif len(df) < 50:
        warnings.append(f"Dataset piccolo ({len(df)} righe) — le previsioni potrebbero essere meno affidabili")

    # 6. Anomalie logiche
    if "anni_in_azienda" in df.columns and "anni_senza_promozione" in df.columns:
        incoherent = (
            pd.to_numeric(df["anni_senza_promozione"], errors="coerce") >
            pd.to_numeric(df["anni_in_azienda"], errors="coerce") + 0.1
        ).sum()
        if incoherent > 0:
            warnings.append(
                f"{incoherent} dipendenti con 'anni_senza_promozione' > 'anni_in_azienda' — valori sospetti"
            )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa i valori nulli nelle colonne numeriche con la mediana."""
    df = df.copy()
    for col, schema in REQUIRED_COLUMNS.items():
        if col in df.columns and schema["type"] in ("int", "float"):
            median_val = pd.to_numeric(df[col], errors="coerce").median()
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median_val)
            # Clip ai range ammessi
            if "min" in schema:
                df[col] = df[col].clip(lower=schema["min"], upper=schema["max"])
    return df


def generate_template() -> pd.DataFrame:
    """Genera un template CSV con 3 righe di esempio da scaricare."""
    examples = [
        {
            "employee_id": "EMP0001",
            "nome": "Mario",
            "cognome": "Rossi",
            "eta": 32,
            "dipartimento": "Engineering",
            "ruolo": "Senior Dev",
            "anni_in_azienda": 4.5,
            "stipendio_annuo": 52000,
            "num_promozioni": 1,
            "anni_senza_promozione": 2.5,
            "crescita_stipendio_percentuale": 3.5,
            "ore_settimanali_medie": 46.0,
            "giorni_ultimo_checkin_hr": 75,
            "manager_score": 7.5,
            "enps_score": 6.0,
            "giorni_assenza_anno": 3,
            "num_progetti_anno": 5,
            "ferie_residue": 8,
            "data_assunzione": "2020-06-15",
        },
        {
            "employee_id": "EMP0002",
            "nome": "Giulia",
            "cognome": "Bianchi",
            "eta": 28,
            "dipartimento": "Sales",
            "ruolo": "Account Executive",
            "anni_in_azienda": 1.2,
            "stipendio_annuo": 38000,
            "num_promozioni": 0,
            "anni_senza_promozione": 1.2,
            "crescita_stipendio_percentuale": 1.5,
            "ore_settimanali_medie": 53.0,
            "giorni_ultimo_checkin_hr": 210,
            "manager_score": 4.0,
            "enps_score": 3.5,
            "giorni_assenza_anno": 12,
            "num_progetti_anno": 2,
            "ferie_residue": 22,
            "data_assunzione": "2023-11-01",
        },
        {
            "employee_id": "EMP0003",
            "nome": "Luca",
            "cognome": "Verdi",
            "eta": 45,
            "dipartimento": "HR",
            "ruolo": "HR Manager",
            "anni_in_azienda": 9.0,
            "stipendio_annuo": 48000,
            "num_promozioni": 3,
            "anni_senza_promozione": 1.0,
            "crescita_stipendio_percentuale": 6.0,
            "ore_settimanali_medie": 40.0,
            "giorni_ultimo_checkin_hr": 14,
            "manager_score": 8.5,
            "enps_score": 8.0,
            "giorni_assenza_anno": 2,
            "num_progetti_anno": 6,
            "ferie_residue": 5,
            "data_assunzione": "2015-03-20",
        },
    ]
    return pd.DataFrame(examples)


def print_schema():
    """Stampa lo schema atteso in modo human-readable."""
    print("\n" + "=" * 60)
    print("SCHEMA CSV ATTESO")
    print("=" * 60)
    print(f"{'Colonna':<40} {'Tipo':<8} {'Range':<20} Descrizione")
    print("-" * 100)
    for col, schema in REQUIRED_COLUMNS.items():
        range_str = ""
        if "min" in schema:
            range_str = f"[{schema['min']}, {schema['max']}]"
        print(f"  {col:<38} {schema['type']:<8} {range_str:<20} {schema['desc']}")
    print("\nColonne opzionali:", ", ".join(OPTIONAL_COLUMNS))