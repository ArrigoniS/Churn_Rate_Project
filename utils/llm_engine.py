"""
LLM Engine — Universal (powered by LiteLLM)
Supporta qualsiasi LLM esistente con un'unica interfaccia.

Installazione:
    pip install litellm

Esempi di model string:
    Anthropic  → "anthropic/claude-haiku-4-5-20251001"
    OpenAI     → "openai/gpt-4o-mini"
    Gemini     → "gemini/gemini-1.5-flash"
    Mistral    → "mistral/mistral-small"
    Ollama     → "ollama/llama3"
    Groq       → "groq/llama3-8b-8192"
    ... e altri 100+

Docs completa modelli: https://docs.litellm.ai/docs/providers
"""

import os
import json

# Import anonymizer compatibile sia con esecuzione da root che come sottomodulo
try:
    from utils.anonymizer import anonymize, deanonymize, anonymize_batch, deanonymize_results
except ImportError:
    from anonymizer import anonymize, deanonymize, anonymize_batch, deanonymize_results


# ─────────────────────────────────────────────
# MODELLI SUGGERITI (mostrati nella UI)
# ─────────────────────────────────────────────

SUGGESTED_MODELS = {
    "⚡ Veloci & economici": [
        "anthropic/claude-haiku-4-5-20251001",
        "openai/gpt-4o-mini",
        "gemini/gemini-1.5-flash",
        "groq/llama3-8b-8192",
        "mistral/mistral-small",
    ],
    "🧠 Più potenti": [
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-6",
        "openai/gpt-4o",
        "gemini/gemini-1.5-pro",
        "mistral/mistral-large-latest",
        "cohere/command-r-plus",
    ],
    "🏠 Locali (Ollama)": [
        "ollama/llama3",
        "ollama/llama3:70b",
        "ollama/mistral",
        "ollama/mixtral",
        "ollama/gemma2",
        "ollama/phi3",
    ],
    "☁️ Altri cloud": [
        "groq/mixtral-8x7b-32768",
        "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        "azure/gpt-4o",
    ],
}

# Mappa provider → variabile/i d'ambiente per la API key
# Gemini richiede sia GEMINI_API_KEY che GOOGLE_API_KEY (LiteLLM accetta entrambe)
PROVIDER_ENV_KEYS = {
    "anthropic":   ["ANTHROPIC_API_KEY"],
    "openai":      ["OPENAI_API_KEY"],
    "gemini":      ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "mistral":     ["MISTRAL_API_KEY"],
    "cohere":      ["COHERE_API_KEY"],
    "groq":        ["GROQ_API_KEY"],
    "together_ai": ["TOGETHERAI_API_KEY"],
    "azure":       ["AZURE_API_KEY"],
    "bedrock":     ["AWS_ACCESS_KEY_ID"],
    "ollama":      [],  # nessuna key necessaria
}


# ─────────────────────────────────────────────
# SETUP API KEY
# ─────────────────────────────────────────────

def set_api_key(model: str, api_key: str):
    """Imposta la API key per tutte le variabili d'ambiente del provider."""
    if not api_key:
        return
    provider = get_provider_from_model(model)
    for env_var in PROVIDER_ENV_KEYS.get(provider, []):
        os.environ[env_var] = api_key


def get_provider_from_model(model: str) -> str:
    """Estrae il provider da una model string es. 'gemini/gemini-1.5-flash' → 'gemini'"""
    return model.split("/")[0].lower() if "/" in model else "unknown"


def needs_api_key(model: str) -> bool:
    """Ritorna True se il modello richiede una API key."""
    provider = get_provider_from_model(model)
    return len(PROVIDER_ENV_KEYS.get(provider, [])) > 0


def _colab_secret(key_name: str):
    """Legge un secret da Google Colab se disponibile."""
    try:
        from google.colab import userdata
        return userdata.get(key_name)
    except Exception:
        return None


def auto_load_key(model: str):
    """Carica la API key da variabili d'ambiente o Colab Secrets."""
    provider = get_provider_from_model(model)
    for env_var in PROVIDER_ENV_KEYS.get(provider, []):
        if os.environ.get(env_var):
            continue
        secret = _colab_secret(env_var)
        if secret:
            os.environ[env_var] = secret


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Sei un esperto HR analyst con 15 anni di esperienza nella retention dei talenti.
Ricevi il profilo di un dipendente con il suo churn score e devi produrre un'analisi concisa e azionabile.
Rispondi SOLO con un oggetto JSON valido, senza markdown, senza backtick, senza testo aggiuntivo.
Usa un tono professionale ma diretto. Le analisi devono essere specifiche al profilo, non generiche.
Sii estremamente conciso: ogni campo testuale massimo 120 caratteri. Non usare apostrofi escaped."""


def build_prompt(employee: dict) -> str:
    return f"""Analizza questo dipendente e genera un report HR strutturato.

PROFILO:
- Alias: {employee.get('alias', employee.get('nome', '') + ' ' + employee.get('cognome', ''))}
- Eta: {employee.get('eta', 'N/D')} anni | Dip: {employee.get('dipartimento', 'N/D')} - {employee.get('ruolo', 'N/D')}
- Anzianita: {employee.get('anni_in_azienda', 'N/D')} anni | Stipendio: EUR {int(employee.get('stipendio_annuo') or 0):,}
- Promozioni: {employee.get('num_promozioni', 'N/D')} (ultima: {employee.get('anni_senza_promozione', 'N/D')} anni fa)
- Crescita stipendio: {employee.get('crescita_stipendio_percentuale', 'N/D')}%/anno
- Ore/sett: {employee.get('ore_settimanali_medie', 'N/D')}h | Ultimo HR check-in: {employee.get('giorni_ultimo_checkin_hr', 'N/D')} giorni fa
- Manager score: {employee.get('manager_score', 'N/D')}/10 | eNPS: {employee.get('enps_score', 'N/D')}/10
- Assenze: {employee.get('giorni_assenza_anno', 'N/D')} gg | Progetti: {employee.get('num_progetti_anno', 'N/D')} | Ferie residue: {employee.get('ferie_residue', 'N/D')} gg

SCORE ML:
- Churn score: {employee.get('churn_score', 0)}/100 | Rischio: {employee.get('risk_level', 'N/D')}
- Causa principale: {employee.get('causa_principale', 'N/D')}

Rispondi SOLO con questo JSON:
{{
  "sintesi": "max 120 caratteri sul rischio principale",
  "causa_principale": "causa piu critica in max 6 parole",
  "cause_secondarie": ["causa 2", "causa 3"],
  "azione_immediata": "azione concreta entro 2 settimane",
  "azione_medio_termine": "azione concreta nei prossimi 3 mesi",
  "segnali_positivi": ["punto di forza da valorizzare"],
  "urgenza": "immediata | alta | media | bassa"
}}"""


# ─────────────────────────────────────────────
# CORE: ANALISI SINGOLO DIPENDENTE
# ─────────────────────────────────────────────

def analyze_employee(employee: dict, model: str, api_key: str = None,
                     use_anonymization: bool = True) -> dict:
    """
    Genera l'analisi LLM per un singolo dipendente.

    Args:
        employee:          dict con i dati del dipendente
        model:             model string es. "gemini/gemini-1.5-flash"
        api_key:           API key (opzionale se già in env o Colab Secrets)
        use_anonymization: se True, rimuove PII prima della chiamata LLM

    Returns:
        dict con l'analisi strutturata
    """
    try:
        from litellm import completion
    except ImportError:
        raise ImportError("Esegui: pip install litellm")

    if api_key:
        set_api_key(model, api_key)
    else:
        auto_load_key(model)

    # Anonimizzazione
    if use_anonymization:
        anon_employee, pii_map = anonymize(employee, index=1)
        payload = anon_employee
    else:
        payload = employee
        pii_map = {}

    try:
        import time as _time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_prompt(payload)},
                    ],
                    max_tokens=2000,
                    temperature=0.3,
                )
                break  # successo, esci dal loop
            except Exception as e:
                err_str = str(e)
                # Rate limit → aspetta e riprova
                if "429" in err_str or "RateLimitError" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < max_retries - 1:
                        wait = 15 * (attempt + 1)  # 15s, 30s, 45s
                        _time.sleep(wait)
                        continue
                raise  # altri errori → propaga subito

        raw = response.choices[0].message.content.strip()

        # Pulizia backtick (gestisce ```json, ```JSON, ``` con o senza newline)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # Se c'è testo prima/dopo il JSON, estrai solo il blocco { }
        if not raw.startswith("{"):
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

        analysis = json.loads(raw.strip())
        analysis["llm_ok"]     = True
        analysis["model_used"] = model
        analysis["anonymized"] = use_anonymization

        # De-anonimizzazione
        if use_anonymization and pii_map:
            analysis = deanonymize(analysis, pii_map)

        return analysis

    except json.JSONDecodeError:
        return _fallback_analysis(employee, error="JSON non valido dalla LLM", model=model)
    except Exception as e:
        return _fallback_analysis(employee, error=str(e), model=model)


# ─────────────────────────────────────────────
# CORE: ANALISI BATCH
# ─────────────────────────────────────────────

def analyze_batch(df_scored, model: str, top_n: int = 20,
                  api_key: str = None, use_anonymization: bool = True) -> list:
    """
    Analizza i top N dipendenti a più alto rischio.

    Returns:
        lista di dict {employee, analysis}
    """
    if api_key:
        set_api_key(model, api_key)
    else:
        auto_load_key(model)

    top_employees = [row.to_dict() for _, row in df_scored.head(top_n).iterrows()]
    total    = len(top_employees)
    provider = get_provider_from_model(model)

    print(f"Analisi LLM — provider: {provider} | modello: {model}")
    print(f"Anonimizzazione: {'ON' if use_anonymization else 'OFF'} | Dipendenti: {total}")
    print("-" * 55)

    if use_anonymization:
        anon_employees, pii_maps = anonymize_batch(top_employees)
    else:
        anon_employees = top_employees
        pii_maps       = [{} for _ in top_employees]

    results = []
    for i, (anon_emp, original_emp, pii_map) in enumerate(
        zip(anon_employees, top_employees, pii_maps), 1
    ):
        nome  = f"{original_emp.get('nome','')} {original_emp.get('cognome','')}".strip()
        score = original_emp.get("churn_score", 0)
        print(f"  [{i:02d}/{total}] {nome:<25} score: {score:.0f} ...", end=" ", flush=True)

        analysis = analyze_employee(
            anon_emp, model=model, api_key=api_key,
            use_anonymization=False,  # già anonimizzato sopra
        )

        if use_anonymization and pii_map:
            analysis = deanonymize(analysis, pii_map)

        status = "OK" if analysis.get("llm_ok") else "FALLBACK"
        print(f"{status}  {analysis.get('causa_principale', analysis.get('error', ''))}")

        results.append({"employee": original_emp, "analysis": analysis})

    successes = sum(1 for r in results if r["analysis"].get("llm_ok"))
    print(f"\nCompletato: {successes}/{total} analisi riuscite con {model}")
    return results


# ─────────────────────────────────────────────
# FALLBACK RULE-BASED
# ─────────────────────────────────────────────

def _fallback_analysis(employee: dict, error: str = "", model: str = "") -> dict:
    """Analisi rule-based di fallback se la LLM non risponde."""
    causa = employee.get("causa_principale", "Profilo da monitorare")
    score = employee.get("churn_score", 0)

    if score >= 75:
        urgenza = "immediata"
        azione  = "Pianificare colloquio di retention entro questa settimana"
    elif score >= 55:
        urgenza = "alta"
        azione  = "Pianificare check-in HR entro 2 settimane"
    else:
        urgenza = "media"
        azione  = "Includere nel prossimo ciclo di feedback HR"

    return {
        "sintesi":              f"Dipendente con churn score {score:.0f}/100. Causa principale: {causa}.",
        "causa_principale":     causa,
        "cause_secondarie":     [],
        "azione_immediata":     azione,
        "azione_medio_termine": "Rivalutare condizioni contrattuali e percorso di crescita",
        "segnali_positivi":     [],
        "urgenza":              urgenza,
        "llm_ok":               False,
        "model_used":           model,
        "error":                error,
    }


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    test_model = sys.argv[1] if len(sys.argv) > 1 else "anthropic/claude-haiku-4-5-20251001"

    test_employee = {
        "nome": "Giulia", "cognome": "Bianchi", "eta": 29,
        "dipartimento": "Engineering", "ruolo": "Mid Dev",
        "anni_in_azienda": 2.5, "stipendio_annuo": 40000,
        "num_promozioni": 0, "anni_senza_promozione": 2.5,
        "crescita_stipendio_percentuale": 1.2, "ore_settimanali_medie": 54,
        "giorni_ultimo_checkin_hr": 210, "manager_score": 4.2,
        "enps_score": 3.8, "giorni_assenza_anno": 11,
        "num_progetti_anno": 3, "ferie_residue": 18,
        "churn_score": 81.0, "risk_level": "Critico",
        "causa_principale": "Overload lavorativo",
    }

    print(f"Test con modello: {test_model}")
    result = analyze_employee(test_employee, model=test_model)
    print(json.dumps(result, indent=2, ensure_ascii=False))
