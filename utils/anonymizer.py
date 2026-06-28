"""
Anonymizer — GDPR-compliant PII protection
Anonimizza i dati sensibili prima di inviarli al modello LLM.
De-anonimizza i risultati in locale prima di mostrarli in UI.

Campi anonimizzati: nome, cognome, employee_id, data_assunzione
Campi mantenuti: tutti i dati numerici e categorici usati dal modello
"""

import hashlib
import re
from typing import Tuple


# Campi PII da rimuovere prima della chiamata LLM
PII_FIELDS = ["nome", "cognome", "employee_id", "data_assunzione"]

# Campi safe da mandare al LLM (solo dati analitici)
SAFE_FIELDS = [
    "eta", "dipartimento", "ruolo", "anni_in_azienda",
    "stipendio_annuo", "num_promozioni", "anni_senza_promozione",
    "crescita_stipendio_percentuale", "ore_settimanali_medie",
    "giorni_ultimo_checkin_hr", "manager_score", "enps_score",
    "giorni_assenza_anno", "num_progetti_anno", "ferie_residue",
    "churn_score", "risk_level", "causa_principale",
]


def anonymize(employee: dict, index: int) -> Tuple[dict, dict]:
    """
    Rimuove i campi PII dal profilo dipendente prima della chiamata LLM.

    Args:
        employee: dict completo con tutti i dati del dipendente
        index:    numero progressivo usato come alias anonimo

    Returns:
        (anon_employee, pii_map)
        - anon_employee: dict sicuro da mandare al LLM
        - pii_map: dict con i dati originali per la de-anonimizzazione
    """
    # Salva i dati sensibili
    pii_map = {field: employee.get(field, "") for field in PII_FIELDS}

    # Costruisce il profilo anonimo con solo campi safe
    anon_employee = {field: employee[field] for field in SAFE_FIELDS if field in employee}

    # Alias anonimo leggibile (es. "Dipendente #0042")
    anon_employee["alias"] = f"Dipendente #{index:04d}"

    # Hash one-way dell'employee_id per tracciabilità interna senza esporre l'ID
    raw_id = str(employee.get("employee_id", f"emp_{index}"))
    anon_employee["id_hash"] = hashlib.sha256(raw_id.encode()).hexdigest()[:8]

    return anon_employee, pii_map


def deanonymize(analysis: dict, pii_map: dict) -> dict:
    """
    Reinserisce i dati reali nell'analisi LLM dopo averla ricevuta.
    Sostituisce anche eventuali riferimenti all'alias nel testo.

    Args:
        analysis: dict restituito dal LLM
        pii_map:  dict con i dati PII originali

    Returns:
        analysis arricchita con i dati reali
    """
    analysis = analysis.copy()

    nome     = pii_map.get("nome", "")
    cognome  = pii_map.get("cognome", "")
    nome_completo = f"{nome} {cognome}".strip()

    # Aggiunge i campi reali all'analisi
    analysis["nome"]        = nome
    analysis["cognome"]     = cognome
    analysis["employee_id"] = pii_map.get("employee_id", "")

    # Sostituisce "Dipendente #XXXX" con il nome reale nei testi
    for field in ["sintesi", "azione_immediata", "azione_medio_termine"]:
        if field in analysis and isinstance(analysis[field], str):
            analysis[field] = re.sub(
                r"Dipendente\s*#\d{4}",
                nome_completo if nome_completo else "il dipendente",
                analysis[field],
            )

    return analysis


def anonymize_batch(employees: list) -> Tuple[list, list]:
    """
    Anonimizza una lista di dipendenti.

    Returns:
        (anon_list, pii_maps)
        - anon_list:  lista di profili anonimi
        - pii_maps:   lista di dict PII nell'ordine corrispondente
    """
    anon_list = []
    pii_maps  = []
    for i, emp in enumerate(employees, start=1):
        anon_emp, pii_map = anonymize(emp, i)
        anon_list.append(anon_emp)
        pii_maps.append(pii_map)
    return anon_list, pii_maps


def deanonymize_results(results: list, pii_maps: list) -> list:
    """
    De-anonimizza una lista di risultati LLM.

    Args:
        results:   lista di dict {employee, analysis}
        pii_maps:  lista di dict PII nell'ordine corrispondente

    Returns:
        lista di risultati con dati reali reinseriti
    """
    restored = []
    for result, pii_map in zip(results, pii_maps):
        r = result.copy()
        # Reinserisce PII nell'employee
        r["employee"] = {**result["employee"], **pii_map}
        # De-anonimizza i testi dell'analisi
        r["analysis"] = deanonymize(result["analysis"], pii_map)
        restored.append(r)
    return restored
