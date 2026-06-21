"""
Dataset Builder
Genera dataset simulati configurabili per il training del modello.
Supporta colonne custom definite nello schema.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker("it_IT")


# ─────────────────────────────────────────────
# CONFIGURAZIONE DEFAULT SIMULAZIONE
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_employees": 600,
    "churn_rate_target": 0.30,       # % dipendenti a rischio
    "departments": {
        "Engineering":  {"weight": 20, "base_salary": 42000},
        "Sales":        {"weight": 15, "base_salary": 35000},
        "Marketing":    {"weight": 12, "base_salary": 32000},
        "HR":           {"weight": 8,  "base_salary": 30000},
        "Finance":      {"weight": 10, "base_salary": 38000},
        "Operations":   {"weight": 10, "base_salary": 28000},
        "Product":      {"weight": 15, "base_salary": 44000},
        "Support":      {"weight": 10, "base_salary": 26000},
    },
    "age_range":         [22, 62],
    "seniority_range":   [0, 12],     # anni in azienda
    "overwork_pct":      0.25,        # % dipendenti in overload
    "low_engagement_pct":0.20,        # % con basso eNPS
    "salary_growth_mean":4.5,
    "salary_growth_std": 3.0,
}

ROLES = {
    "Engineering": ["Junior Dev", "Mid Dev", "Senior Dev", "Tech Lead", "Engineering Manager"],
    "Sales":       ["Sales Rep", "Account Executive", "Senior AE", "Sales Manager", "VP Sales"],
    "Marketing":   ["Marketing Specialist", "Content Manager", "Growth Manager", "Marketing Director"],
    "HR":          ["HR Specialist", "HR Business Partner", "HR Manager", "Head of HR"],
    "Finance":     ["Analyst", "Senior Analyst", "Finance Manager", "CFO"],
    "Operations":  ["Ops Specialist", "Ops Manager", "Head of Ops"],
    "Product":     ["Product Manager", "Senior PM", "Head of Product", "CPO"],
    "Support":     ["Support Agent", "Senior Agent", "Support Lead", "Support Manager"],
}


# ─────────────────────────────────────────────
# CORE GENERATOR
# ─────────────────────────────────────────────

def generate_dataset(config: dict = None, schema: dict = None,
                     seed: int = 42) -> pd.DataFrame:
    """
    Genera un dataset simulato configurabile.

    Args:
        config: dizionario di configurazione (usa DEFAULT_CONFIG se None)
        schema: schema con colonne custom (da schema_manager)
        seed:   seed per riproducibilità

    Returns:
        DataFrame con tutti i dipendenti generati
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    np.random.seed(seed)
    random.seed(seed)

    n = cfg["n_employees"]
    departments = cfg["departments"]
    dept_names  = list(departments.keys())
    dept_weights = [departments[d]["weight"] for d in dept_names]

    employees = []
    for i in range(1, n + 1):
        emp = _generate_employee(i, cfg, dept_names, dept_weights, schema)
        employees.append(emp)

    df = pd.DataFrame(employees)

    # Aggiusta il churn rate per avvicinarsi al target
    df = _adjust_churn_rate(df, cfg["churn_rate_target"])

    print(f"Dataset generato: {len(df)} dipendenti")
    print(f"Churn rate effettivo: {df['churn_label'].mean():.1%}")
    return df


def _generate_employee(emp_id: int, cfg: dict, dept_names: list,
                        dept_weights: list, schema: dict = None) -> dict:
    department = random.choices(dept_names, weights=dept_weights)[0]
    dept_cfg   = cfg["departments"][department]
    roles_list = ROLES.get(department, ["Specialist", "Senior", "Manager"])
    role_idx   = random.choices(range(len(roles_list)),
                                weights=[30, 30, 20, 12, 8][:len(roles_list)])[0]
    role = roles_list[role_idx]

    # Date
    max_years = cfg["seniority_range"][1]
    hire_date = datetime.now() - timedelta(days=random.uniform(0, max_years * 365))
    anni_in_azienda = (datetime.now() - hire_date).days / 365.0

    age_min, age_max = cfg["age_range"]
    age = int(np.clip(np.random.normal((age_min + age_max) / 2, 8), age_min, age_max))

    salary = int(dept_cfg["base_salary"] * (1 + role_idx * 0.22) * np.random.uniform(0.88, 1.18))
    last_promo = random.uniform(0.1, min(anni_in_azienda + 0.1, 6.0))
    crescita = round(np.random.normal(cfg["salary_growth_mean"], cfg["salary_growth_std"]), 1)

    # Overload: % configurabile
    if random.random() < cfg["overwork_pct"]:
        ore = round(np.random.normal(53, 5), 1)
    else:
        ore = round(np.random.normal(40, 5), 1)
    ore = float(np.clip(ore, 25, 75))

    # Engagement
    if random.random() < cfg["low_engagement_pct"]:
        enps = round(np.random.normal(3.5, 1.5), 1)
        manager_score = round(np.random.normal(4.0, 1.8), 1)
    else:
        enps = round(np.random.normal(7.0, 1.5), 1)
        manager_score = round(np.random.normal(7.2, 1.5), 1)

    giorni_checkin = random.choices(
        [random.randint(7, 30), random.randint(31, 90), random.randint(91, 365)],
        weights=[40, 35, 25]
    )[0]

    emp = {
        "employee_id":                    f"EMP{emp_id:04d}",
        "nome":                           fake.first_name(),
        "cognome":                        fake.last_name(),
        "data_assunzione":                hire_date.strftime("%Y-%m-%d"),
        "eta":                            age,
        "dipartimento":                   department,
        "ruolo":                          role,
        "anni_in_azienda":                round(anni_in_azienda, 1),
        "stipendio_annuo":                salary,
        "num_promozioni":                 max(0, int(anni_in_azienda / 2.5) + random.randint(-1, 1)),
        "anni_senza_promozione":          round(last_promo, 1),
        "crescita_stipendio_percentuale": crescita,
        "ore_settimanali_medie":          ore,
        "giorni_ultimo_checkin_hr":       giorni_checkin,
        "manager_score":                  float(np.clip(manager_score, 1, 10)),
        "enps_score":                     float(np.clip(enps, 1, 10)),
        "giorni_assenza_anno":            max(0, int(np.random.normal(5, 4))),
        "num_progetti_anno":              max(0, int(np.random.normal(4, 2))),
        "ferie_residue":                  max(0, int(np.random.normal(8, 7))),
    }

    # Colonne custom dallo schema
    if schema:
        for col, meta in schema.get("custom_columns", {}).items():
            emp[col] = _generate_custom_value(meta)

    # Calcola churn probability
    emp["churn_probability"] = _compute_churn_probability(emp)
    emp["churn_label"] = 1 if emp["churn_probability"] > 0.45 else 0
    return emp


def _generate_custom_value(meta: dict):
    """Genera un valore casuale per una colonna custom."""
    col_type = meta.get("type", "float")
    min_v = meta.get("min", 0)
    max_v = meta.get("max", 100)

    if meta.get("categorical"):
        options = meta.get("options", ["A", "B", "C"])
        return random.choice(options)
    if col_type == "bool":
        return random.random() > 0.5
    if col_type == "int":
        return int(np.random.uniform(min_v, max_v))
    if col_type == "float":
        return round(np.random.uniform(min_v, max_v), 2)
    return ""


def _compute_churn_probability(row: dict) -> float:
    """Formula logistica per calcolare la probabilità di churn."""
    score = 0.0
    if row.get("ore_settimanali_medie", 0) > 50:          score += 0.25
    elif row.get("ore_settimanali_medie", 0) > 45:        score += 0.12
    if row.get("anni_senza_promozione", 0) > 3:           score += 0.20
    elif row.get("anni_senza_promozione", 0) > 2:         score += 0.10
    if row.get("crescita_stipendio_percentuale", 10) < 2: score += 0.18
    elif row.get("crescita_stipendio_percentuale", 10) < 5: score += 0.07
    if row.get("giorni_ultimo_checkin_hr", 0) > 180:      score += 0.15
    elif row.get("giorni_ultimo_checkin_hr", 0) > 90:     score += 0.06
    if row.get("manager_score", 10) < 3:                  score += 0.15
    elif row.get("manager_score", 10) < 5:                score += 0.06
    if row.get("enps_score", 10) < 3:                     score += 0.12
    elif row.get("enps_score", 10) < 5:                   score += 0.05
    if row.get("giorni_assenza_anno", 0) > 15:            score += 0.08
    if row.get("anni_in_azienda", 5) > 8:                 score += 0.05
    if row.get("anni_in_azienda", 5) < 1:                 score += 0.10
    if row.get("num_progetti_anno", 5) < 2:               score += 0.07
    if row.get("ferie_residue", 0) > 20:                  score += 0.06
    noise = np.random.normal(0, 0.05)
    return float(np.clip(min(score, 0.92) + noise, 0.02, 0.97))


def _adjust_churn_rate(df: pd.DataFrame, target_rate: float) -> pd.DataFrame:
    """
    Aggiusta i label per avvicinarsi al churn rate target
    modificando la soglia di classificazione.
    """
    # Trova la soglia che produce il churn rate desiderato
    probs = df["churn_probability"].values
    thresholds = np.linspace(0.1, 0.9, 100)
    best_thresh = 0.45
    best_diff = float("inf")
    for t in thresholds:
        rate = (probs > t).mean()
        diff = abs(rate - target_rate)
        if diff < best_diff:
            best_diff = diff
            best_thresh = t
    df["churn_label"] = (df["churn_probability"] > best_thresh).astype(int)
    return df
