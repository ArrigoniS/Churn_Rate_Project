"""
Pagina 1 — Configurazione Dataset
Permette di:
- Generare un dataset simulato con parametri configurabili
- Caricare un CSV custom
- Aggiungere/rimuovere colonne dallo schema
- Scaricare il template CSV aggiornato
"""

import os
import io
import json
import pandas as pd
import streamlit as st
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schema_manager import (
    load_schema, save_schema, reset_schema, add_custom_column,
    remove_custom_column, generate_template_csv,
    COLUMN_TYPES, COLUMN_ROLES, DEFAULT_COLUMNS,
)
from core.dataset_builder import generate_dataset, DEFAULT_CONFIG
from utils.validator import validate_dataset, impute_missing

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Configurazione — HR Churn",
    page_icon="⚙️",
    layout="wide",
)

# CSS condiviso
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1300px !important; }
.stButton > button {
    background: #e94560 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] { background: #13151c !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD SCHEMA
# ─────────────────────────────────────────────
if "schema" not in st.session_state:
    st.session_state.schema = load_schema()

schema = st.session_state.schema

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# ⚙️ Configurazione Dataset")
st.markdown(
    "<p style='color:#6b7280;'>Definisci la sorgente dati, personalizza le colonne "
    "e scarica il template per il tuo CSV.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────
# TAB
# ─────────────────────────────────────────────
tab_sim, tab_csv, tab_schema = st.tabs([
    "🎲 Dataset simulato",
    "📤 Carica CSV",
    "🧩 Gestione colonne",
])


# ══════════════════════════════════════════════
# TAB 1 — DATASET SIMULATO
# ══════════════════════════════════════════════
with tab_sim:
    st.markdown("### Configura il dataset simulato")
    st.markdown(
        "<p style='color:#6b7280; font-size:0.88rem;'>Personalizza i parametri "
        "e genera un dataset realistico con dati fake.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Parametri generali")
        n_employees = st.slider("Numero dipendenti", 100, 2000, 600, 50)
        churn_target = st.slider(
            "Churn rate target (%)", 10, 60, 30, 5,
            help="Percentuale di dipendenti che il dataset deve classificare come a rischio"
        )
        seed = st.number_input("Seed (riproducibilità)", value=42, step=1)

        st.markdown("#### Profilo dipendenti")
        age_min, age_max = st.slider("Range età", 18, 70, (22, 62))
        sen_min, sen_max = st.slider("Anzianità massima (anni)", 1, 20, (0, 12))
        overwork_pct = st.slider(
            "% dipendenti in overload (>50h/sett)", 5, 60, 25, 5,
            help="Quanti dipendenti nel dataset lavorano oltre le 50 ore settimanali"
        )
        low_eng_pct = st.slider(
            "% dipendenti con basso engagement", 5, 50, 20, 5,
            help="Quanti dipendenti hanno eNPS e manager score bassi"
        )
        salary_growth_mean = st.slider("Crescita stipendio media (%)", 0.0, 15.0, 4.5, 0.5)

    with col2:
        st.markdown("#### Dipartimenti e stipendi base")
        st.caption("Modifica i pesi (influenza la distribuzione) e gli stipendi base")

        dept_config = {}
        default_depts = DEFAULT_CONFIG["departments"]
        for dept, vals in default_depts.items():
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown(f"<div style='padding-top:8px; font-size:0.88rem;'>{dept}</div>",
                            unsafe_allow_html=True)
            with c2:
                weight = st.number_input(f"Peso##{dept}", 1, 100,
                                         vals["weight"], 1, label_visibility="collapsed")
            with c3:
                salary = st.number_input(f"Stipendio##{dept}", 15000, 200000,
                                         vals["base_salary"], 1000, label_visibility="collapsed",
                                         format="%d")
            dept_config[dept] = {"weight": weight, "base_salary": salary}

    st.markdown("---")

    # Preview stimata
    total_w = sum(d["weight"] for d in dept_config.values())
    preview_cols = st.columns(5)
    preview_data = [
        ("Dipendenti", str(n_employees), "#e2e8f0"),
        ("Churn target", f"{churn_target}%", "#e94560"),
        ("Overload", f"{overwork_pct}%", "#f97316"),
        ("Low engagement", f"{low_eng_pct}%", "#eab308"),
        ("Seed", str(seed), "#06b6d4"),
    ]
    for col, (label, val, color) in zip(preview_cols, preview_data):
        with col:
            st.markdown(
                f"<div style='background:#13151c; border:1px solid #252836; border-radius:10px;"
                f"padding:0.8rem; text-align:center;'>"
                f"<div style='font-size:1.4rem; font-weight:800; color:{color}; "
                f"font-family:DM Mono,monospace;'>{val}</div>"
                f"<div style='font-size:0.7rem; color:#6b7280; text-transform:uppercase; "
                f"letter-spacing:0.08em;'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    gen_btn = st.button("🎲 Genera dataset simulato", width="content")

    if gen_btn:
        config = {
            "n_employees":         n_employees,
            "churn_rate_target":   churn_target / 100,
            "departments":         dept_config,
            "age_range":           [age_min, age_max],
            "seniority_range":     [sen_min, sen_max],
            "overwork_pct":        overwork_pct / 100,
            "low_engagement_pct":  low_eng_pct / 100,
            "salary_growth_mean":  salary_growth_mean,
            "salary_growth_std":   3.0,
        }

        with st.spinner("Generazione in corso..."):
            os.makedirs("data", exist_ok=True)
            df = generate_dataset(config=config, schema=schema, seed=int(seed))
            df.to_csv("data/hr_dataset.csv", index=False)
            st.session_state.df_train = df
            st.session_state.dataset_config = config

        st.success(f"✓ Dataset generato: **{len(df)} dipendenti** | "
                   f"Churn rate effettivo: **{df['churn_label'].mean():.1%}**")

        # Anteprima
        st.markdown("**Anteprima (prime 5 righe):**")
        st.dataframe(df.head(), width="stretch", hide_index=True)

        # Download
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Scarica dataset generato (CSV)",
            data=csv_bytes,
            file_name="hr_dataset_simulato.csv",
            mime="text/csv",
        )

        # Distribuzione dipartimenti
        st.markdown("**Distribuzione dipartimenti:**")
        dept_dist = df["dipartimento"].value_counts()
        st.bar_chart(dept_dist)


# ══════════════════════════════════════════════
# TAB 2 — CARICA CSV
# ══════════════════════════════════════════════
with tab_csv:
    st.markdown("### Carica il tuo dataset")

    col_up, col_tmpl = st.columns([2, 1])

    with col_tmpl:
        st.markdown("#### Template CSV")
        st.markdown(
            "<p style='color:#6b7280; font-size:0.85rem;'>Scarica il template con lo schema "
            "attuale (include le colonne custom che hai aggiunto).</p>",
            unsafe_allow_html=True,
        )
        template_csv = generate_template_csv(schema)
        st.download_button(
            "⬇️ Scarica template CSV",
            data=template_csv.encode("utf-8"),
            file_name="template_hr_churn.csv",
            mime="text/csv",
            width="stretch",
        )

        st.markdown("#### Schema attuale")
        all_cols = {**schema.get("columns", {}), **schema.get("custom_columns", {})}
        schema_df = pd.DataFrame([
            {"Colonna": col, "Tipo": meta["type"], "Ruolo": meta["role"], "Descrizione": meta["desc"]}
            for col, meta in all_cols.items()
        ])
        st.dataframe(schema_df, width="stretch", hide_index=True, height=350)

    with col_up:
        st.markdown("#### Upload CSV")
        uploaded = st.file_uploader(
            "Carica il tuo CSV",
            type=["csv"],
            help="Il file deve seguire lo schema mostrato a destra.",
        )

        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                st.markdown(f"**Righe:** {len(df_upload)} | **Colonne:** {df_upload.shape[1]}")

                # Validazione
                is_valid, errors, warnings = validate_dataset(df_upload)

                for w in warnings:
                    st.warning(f"⚠ {w}")
                for e in errors:
                    st.error(f"❌ {e}")

                if is_valid:
                    df_clean = impute_missing(df_upload)
                    os.makedirs("data", exist_ok=True)
                    df_clean.to_csv("data/hr_dataset.csv", index=False)
                    st.session_state.df_train = df_clean
                    st.success(f"✓ Dataset caricato e validato: **{len(df_clean)} dipendenti**")

                    st.markdown("**Anteprima:**")
                    st.dataframe(df_clean.head(), width="stretch", hide_index=True)

                    # Stats rapide
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Dipendenti", len(df_clean))
                    if "churn_label" in df_clean.columns:
                        sc2.metric("Churn rate", f"{df_clean['churn_label'].mean():.1%}")
                    sc3.metric("Colonne", df_clean.shape[1])

            except Exception as ex:
                st.error(f"Errore nel caricamento: {ex}")


# ══════════════════════════════════════════════
# TAB 3 — GESTIONE COLONNE
# ══════════════════════════════════════════════
with tab_schema:
    st.markdown("### Gestione colonne dello schema")

    col_existing, col_add = st.columns([1, 1])

    # ── Colonne esistenti ──
    with col_existing:
        st.markdown("#### Colonne di default")
        default_df = pd.DataFrame([
            {
                "Colonna": col,
                "Tipo":    meta["type"],
                "Ruolo":   meta["role"],
                "Range":   f"[{meta.get('min','')}, {meta.get('max','')}]"
                           if "min" in meta else "—",
                "Desc":    meta["desc"],
            }
            for col, meta in DEFAULT_COLUMNS.items()
        ])
        st.dataframe(default_df, width="stretch", hide_index=True, height=380)

        st.markdown("#### Colonne custom aggiunte")
        custom_cols = schema.get("custom_columns", {})
        if custom_cols:
            custom_df = pd.DataFrame([
                {"Colonna": col, "Tipo": meta["type"], "Ruolo": meta["role"],
                 "Desc": meta["desc"]}
                for col, meta in custom_cols.items()
            ])
            st.dataframe(custom_df, width="stretch", hide_index=True)

            # Rimozione
            to_remove = st.selectbox(
                "Rimuovi colonna custom",
                options=["— seleziona —"] + list(custom_cols.keys()),
            )
            if to_remove != "— seleziona —":
                if st.button(f"🗑 Rimuovi '{to_remove}'", type="secondary"):
                    schema = remove_custom_column(schema, to_remove)
                    save_schema(schema)
                    st.session_state.schema = schema
                    st.success(f"Colonna '{to_remove}' rimossa.")
                    st.rerun()
        else:
            st.info("Nessuna colonna custom aggiunta.")

        # Reset
        st.markdown("---")
        if st.button("↩️ Ripristina schema di default", type="secondary"):
            reset_schema()
            st.session_state.schema = load_schema()
            st.success("Schema ripristinato al default.")
            st.rerun()

    # ── Aggiungi colonna ──
    with col_add:
        st.markdown("#### Aggiungi colonna custom")
        st.markdown(
            "<p style='color:#6b7280; font-size:0.85rem;'>Le colonne custom vengono "
            "aggiunte al template CSV, al dataset simulato e usate dal modello se "
            "il ruolo è 'feature'.</p>",
            unsafe_allow_html=True,
        )

        with st.form("add_column_form"):
            col_name = st.text_input(
                "Nome colonna",
                placeholder="es. remote_work_days",
                help="Solo lettere minuscole, numeri e underscore",
            )
            col_type = st.selectbox("Tipo dato", COLUMN_TYPES)
            col_role = st.selectbox(
                "Ruolo",
                COLUMN_ROLES,
                help="'feature' → usata dal modello | 'pii' → anonimizzata | 'ignore' → ignorata",
            )
            col_desc = st.text_input("Descrizione", placeholder="es. Giorni di lavoro da remoto/settimana")
            is_categorical = st.checkbox("È categorica (stringa con valori fissi)?")

            c1, c2 = st.columns(2)
            with c1:
                col_min = st.number_input("Valore minimo", value=0.0) if not is_categorical else None
            with c2:
                col_max = st.number_input("Valore massimo", value=100.0) if not is_categorical else None

            cat_options = None
            if is_categorical:
                cat_opts_str = st.text_input(
                    "Valori possibili (separati da virgola)",
                    placeholder="es. remote,hybrid,office",
                )
                cat_options = [v.strip() for v in cat_opts_str.split(",") if v.strip()]

            submitted = st.form_submit_button("➕ Aggiungi colonna")

            if submitted:
                # Validazione nome
                import re
                if not col_name:
                    st.error("Il nome colonna è obbligatorio.")
                elif not re.match(r'^[a-z][a-z0-9_]*$', col_name):
                    st.error("Nome non valido — usa solo lettere minuscole, numeri e underscore.")
                elif col_name in {**schema.get("columns", {}), **schema.get("custom_columns", {})}:
                    st.error(f"La colonna '{col_name}' esiste già.")
                else:
                    meta = {
                        "type": col_type, "role": col_role, "desc": col_desc,
                        "custom": True,
                    }
                    if not is_categorical:
                        meta["min"] = col_min
                        meta["max"] = col_max
                    else:
                        meta["categorical"] = True
                        if cat_options:
                            meta["options"] = cat_options

                    schema = add_custom_column(
                        schema, col_name, col_type, col_role, col_desc,
                        min_val=col_min if not is_categorical else None,
                        max_val=col_max if not is_categorical else None,
                        categorical=is_categorical,
                    )
                    if is_categorical and cat_options:
                        schema["custom_columns"][col_name]["options"] = cat_options

                    save_schema(schema)
                    st.session_state.schema = schema
                    st.success(f"✓ Colonna '{col_name}' aggiunta!")
                    st.rerun()

        # Info ruoli
        st.markdown("---")
        st.markdown("**Guida ai ruoli:**")
        roles_info = {
            "feature": "Usata dal modello ML per la predizione",
            "pii":     "Dato sensibile — anonimizzato prima del LLM",
            "id":      "Identificatore — non usato dal modello",
            "target":  "Variabile da predire (churn_label)",
            "ignore":  "Presente nel CSV ma ignorata completamente",
        }
        for role, desc in roles_info.items():
            st.markdown(
                f"<div style='display:flex; gap:8px; margin-bottom:4px;'>"
                f"<code style='background:#1a1d27; padding:2px 8px; border-radius:4px; "
                f"font-size:0.8rem;'>{role}</code>"
                f"<span style='font-size:0.85rem; color:#94a3b8;'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

    # ── Stato e download schema ──
    st.markdown("---")
    schema_cols = st.columns([2, 1])
    with schema_cols[0]:
        n_default = len(schema.get("columns", {}))
        n_custom  = len(schema.get("custom_columns", {}))
        n_feature = len([c for c, m in {**schema.get("columns", {}),
                                         **schema.get("custom_columns", {})}.items()
                         if m.get("role") == "feature"])
        st.markdown(
            f"**Schema attuale:** {n_default} colonne di default + "
            f"**{n_custom} custom** | **{n_feature} feature** per il modello"
        )
    with schema_cols[1]:
        schema_json = json.dumps(schema, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "⬇️ Esporta schema (JSON)",
            data=schema_json,
            file_name="schema.json",
            mime="application/json",
            width="stretch",
        )

# ─────────────────────────────────────────────
# STATO SESSIONE — mostra cosa è pronto
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Stato configurazione")

s1, s2, s3 = st.columns(3)

with s1:
    if "df_train" in st.session_state and st.session_state.df_train is not None:
        df_t = st.session_state.df_train
        st.success(f"✓ Dataset pronto — {len(df_t)} dipendenti")
    else:
        # Prova a caricare da file
        if os.path.exists("data/hr_dataset.csv"):
            df_t = pd.read_csv("data/hr_dataset.csv")
            st.session_state.df_train = df_t
            st.success(f"✓ Dataset trovato — {len(df_t)} dipendenti")
        else:
            st.warning("⚠ Nessun dataset — genera o carica un CSV")

with s2:
    n_custom = len(schema.get("custom_columns", {}))
    if n_custom > 0:
        st.info(f"🧩 {n_custom} colonne custom nello schema")
    else:
        st.info("🧩 Schema di default (nessuna colonna custom)")

with s3:
    if os.path.exists("models/churn_model.pkl"):
        st.success("✓ Modello trainato trovato")
    else:
        st.warning("⚠ Nessun modello — vai alla pagina Training")
