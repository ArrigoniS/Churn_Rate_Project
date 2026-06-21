"""
Pagina 3 — Analisi & Scoring
Permette di:
- Generare un dataset di test con il modello appena trainato
- Caricare un CSV operativo per lo scoring
- Visualizzare la dashboard completa post-training
- Avviare l'analisi AI e scaricare i report
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schema_manager   import load_schema, get_numeric_features, get_categorical_features
from core.model_trainer    import load_training_artifacts
from core.dataset_builder  import generate_dataset
from utils.validator       import validate_dataset, impute_missing

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisi — HR Churn",
    page_icon="🔮",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }
.stButton > button {
    background: #e94560 !important; color: white !important;
    border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}
[data-testid="stSidebar"] { background: #13151c !important; }
.kpi-card {
    background: #13151c; border: 1px solid #252836; border-radius: 12px;
    padding: 1.2rem 1.4rem; text-align: center;
}
.kpi-number { font-family: 'DM Mono', monospace; font-size: 2.6rem;
              font-weight: 800; line-height: 1; margin-bottom: 4px; }
.kpi-label  { font-size: 0.72rem; color: #6b7280; text-transform: uppercase;
              letter-spacing: 0.08em; }
.emp-card   { background: #13151c; border: 1px solid #252836;
              border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 6px; }
.badge { display:inline-block; padding:3px 10px; border-radius:20px;
         font-size:0.72rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; }
.badge-critico { background:rgba(239,68,68,0.15);  color:#ef4444; border:1px solid rgba(239,68,68,0.3); }
.badge-alto    { background:rgba(249,115,22,0.15); color:#f97316; border:1px solid rgba(249,115,22,0.3); }
.badge-medio   { background:rgba(234,179,8,0.15);  color:#eab308; border:1px solid rgba(234,179,8,0.3); }
.badge-basso   { background:rgba(34,197,94,0.15);  color:#22c55e; border:1px solid rgba(34,197,94,0.3); }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)
RISK_COLORS = {"Critico":"#ef4444","Alto":"#f97316","Medio":"#eab308","Basso":"#22c55e"}


def badge_html(risk):
    return f'<span class="badge badge-{risk.lower()}">{risk}</span>'

def score_color(s):
    if s >= 75: return "#ef4444"
    if s >= 55: return "#f97316"
    if s >= 30: return "#eab308"
    return "#22c55e"


# ─────────────────────────────────────────────
# LOAD ARTIFACTS + SCHEMA
# ─────────────────────────────────────────────
schema    = load_schema()
artifacts = load_training_artifacts()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🔮 Analisi & Scoring")
st.markdown(
    "<p style='color:#6b7280;'>Scorecard dipendenti, analisi AI e report. "
    "Usa un dataset di test generato automaticamente o carica il tuo CSV.</p>",
    unsafe_allow_html=True,
)

if not artifacts:
    st.error("❌ Nessun modello trovato. Vai alla pagina **🏋️ Training** e addestra prima il modello.")
    st.stop()

st.success(
    f"✓ Modello caricato — ROC-AUC: **{artifacts['metrics']['roc_auc']:.3f}** | "
    f"Trainato il: {artifacts['metrics'].get('trained_at','')[:16].replace('T',' ')}"
)
st.markdown("---")


# ─────────────────────────────────────────────
# SCORING FUNCTION
# ─────────────────────────────────────────────
def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    model        = artifacts["model"]
    encoders     = artifacts["encoders"]
    feature_cols = artifacts["feature_cols"]

    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            known = set(le.classes_)
            df[col + "_enc"] = df[col].apply(
                lambda x: le.transform([x])[0] if str(x) in known else 0
            )

    # Assicura che tutte le feature cols esistano
    missing_cols = [c for c in feature_cols if c not in df.columns]
    for c in missing_cols:
        df[c] = 0  # fallback per colonne mancanti

    X = df[feature_cols].values.astype(float)
    proba = model.predict_proba(X)[:, 1]

    df["churn_score"] = (proba * 100).round(1)
    df["risk_level"]  = pd.cut(
        df["churn_score"], bins=[0, 30, 55, 75, 100],
        labels=["Basso", "Medio", "Alto", "Critico"]
    ).astype(str)

    def identify_cause(row):
        causes = []
        if row.get("ore_settimanali_medie", 0) > 50:
            causes.append(("Overload lavorativo", row["ore_settimanali_medie"] - 40))
        if row.get("anni_senza_promozione", 0) > 3:
            causes.append(("Stagnazione carriera", row["anni_senza_promozione"] * 5))
        if row.get("crescita_stipendio_percentuale", 10) < 2:
            causes.append(("Crescita stipendio insufficiente",
                           (2 - row["crescita_stipendio_percentuale"]) * 8))
        if row.get("giorni_ultimo_checkin_hr", 0) > 180:
            causes.append(("Scarso engagement HR", row["giorni_ultimo_checkin_hr"] / 20))
        if row.get("manager_score", 10) < 4:
            causes.append(("Rapporto con manager critico", (5 - row["manager_score"]) * 6))
        if row.get("enps_score", 10) < 4:
            causes.append(("Basso engagement aziendale", (5 - row["enps_score"]) * 5))
        return sorted(causes, key=lambda x: x[1], reverse=True)[0][0] if causes else "Profilo da monitorare"

    df["causa_principale"] = df.apply(identify_cause, axis=1)
    return df.sort_values("churn_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# SIDEBAR — sorgente dati + AI settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂️ Sorgente dati")
    data_source = st.radio(
        "Scegli dataset",
        ["🧪 Genera dataset di test", "📤 Carica CSV"],
        index=0,
    )

    df_scored = None

    if data_source == "🧪 Genera dataset di test":
        st.markdown("---")
        st.markdown("**Parametri test set**")
        n_test    = st.slider("Dipendenti", 50, 500, 150, 50)
        test_seed = st.number_input("Seed", value=99, step=1)

        gen_test = st.button("🧪 Genera test set", width="stretch")
        if gen_test:
            with st.spinner("Generazione..."):
                test_config = {
                    "n_employees":       n_test,
                    "churn_rate_target": 0.30,
                }
                df_test = generate_dataset(config=test_config, schema=schema, seed=int(test_seed))
                df_scored = score_dataframe(df_test)
                df_scored.to_csv("data/hr_scored.csv", index=False)
                st.session_state.df_scored = df_scored
                st.success(f"✓ {n_test} dipendenti generati e scorati")

    else:
        uploaded = st.file_uploader("Carica CSV", type=["csv"])
        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                is_valid, errors, warnings = validate_dataset(df_upload)
                for w in warnings:
                    st.warning(f"⚠ {w}")
                for e in errors:
                    st.error(f"❌ {e}")
                if is_valid:
                    df_clean  = impute_missing(df_upload)
                    df_scored = score_dataframe(df_clean)
                    df_scored.to_csv("data/hr_scored.csv", index=False)
                    st.session_state.df_scored = df_scored
                    st.success(f"✓ {len(df_scored)} dipendenti scorati")
            except Exception as ex:
                st.error(f"Errore: {ex}")

        # Template download
        from core.schema_manager import generate_template_csv
        tmpl = generate_template_csv(schema)
        st.download_button("⬇️ Template CSV", data=tmpl.encode(),
                           file_name="template_hr_churn.csv", mime="text/csv",
                           width="stretch")

    # Recupera da session state o file
    if df_scored is None:
        if "df_scored" in st.session_state and st.session_state.df_scored is not None:
            df_scored = st.session_state.df_scored
        elif os.path.exists("data/hr_scored.csv"):
            df_scored = pd.read_csv("data/hr_scored.csv")
            st.session_state.df_scored = df_scored

    st.markdown("---")
    st.markdown("### 🤖 Modello AI")

    from utils.llm_engine import SUGGESTED_MODELS, PROVIDER_ENV_KEYS, get_provider_from_model

    all_models = [m for models in SUGGESTED_MODELS.values() for m in models]
    active_model = st.selectbox("Modello LLM", all_models, index=0)
    custom_model = st.text_input("Modello custom", placeholder="es. gemini/gemini-2.0-flash")
    if custom_model.strip():
        active_model = custom_model.strip()

    provider = get_provider_from_model(active_model)
    env_vars = PROVIDER_ENV_KEYS.get(provider, [])
    env_var_list = [env_vars] if isinstance(env_vars, str) else env_vars

    api_key_input = ""
    if env_var_list:
        api_key_input = st.text_input(
            f"API Key ({provider})", type="password",
            placeholder=f"{' / '.join(env_var_list)}",
        )
        if api_key_input:
            for ev in env_var_list:
                os.environ[ev] = api_key_input
    else:
        st.caption("Ollama — nessuna key necessaria")

    top_n = st.slider("Top N da analizzare con AI", 5, 50, 15, 5)

    use_anonymization = st.toggle("🔐 Anonimizza PII", value=True,
                                   help="Rimuove nome/cognome prima di inviare al LLM")

    st.markdown("---")
    company_name = st.text_input("Nome azienda (per report)", value="Azienda")


# ─────────────────────────────────────────────
# GUARD — dataset non ancora caricato
# ─────────────────────────────────────────────
if df_scored is None:
    st.markdown("""
    <div style='text-align:center; padding:5rem 2rem; color:#4b5563;'>
        <div style='font-size:3rem;'>🔮</div>
        <div style='font-size:1.1rem; font-family:Syne,sans-serif; font-weight:700;
                    color:#6b7280; margin-top:1rem;'>Nessun dataset caricato</div>
        <div style='font-size:0.88rem; margin-top:0.5rem;'>
            Genera un dataset di test o carica un CSV dalla sidebar
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_dash, tab_lista, tab_ai, tab_report = st.tabs([
    "📊 Dashboard", "👥 Classifica", "🤖 Analisi AI", "📄 Report"
])


# ══════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════
with tab_dash:
    risk_counts = df_scored["risk_level"].value_counts()
    total       = len(df_scored)
    avg_score   = df_scored["churn_score"].mean()
    n_critico   = risk_counts.get("Critico", 0)
    n_alto      = risk_counts.get("Alto",    0)

    # KPI
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, num, label, color in [
        (k1, str(total),              "Dipendenti",      "#e2e8f0"),
        (k2, str(n_critico),          "Critici",         "#ef4444"),
        (k3, str(n_alto),             "Alti",            "#f97316"),
        (k4, str(n_critico + n_alto), "Totale a rischio","#e94560"),
        (k5, f"{avg_score:.0f}/100",  "Score medio",     "#06b6d4"),
    ]:
        with col:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-number' style='color:{color};'>"
                f"{num}</div><div class='kpi-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Distribuzione rischio")
        pie_data = df_scored["risk_level"].value_counts().reset_index()
        pie_data.columns = ["Livello", "N"]
        fig = go.Figure(go.Pie(
            labels=pie_data["Livello"], values=pie_data["N"],
            marker_colors=[RISK_COLORS.get(l, "#666") for l in pie_data["Livello"]],
            hole=0.55, textfont_size=12, textfont_color="#e2e8f0",
        ))
        fig.update_layout(**PLOTLY_THEME, height=260,
                          legend=dict(font=dict(color="#94a3b8")))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown("#### Critici + Alti per dipartimento")
        if "dipartimento" in df_scored.columns:
            dept_risk = (
                df_scored[df_scored["risk_level"].isin(["Critico","Alto"])]
                .groupby(["dipartimento","risk_level"]).size().reset_index(name="n")
            )
            fig2 = px.bar(dept_risk, x="n", y="dipartimento", color="risk_level",
                          orientation="h", color_discrete_map=RISK_COLORS,
                          labels={"n":"","dipartimento":"","risk_level":"Rischio"})
            fig2.update_layout(**PLOTLY_THEME, height=260,
                               yaxis=dict(categoryorder="total ascending"),
                               legend=dict(font=dict(color="#94a3b8")))
            st.plotly_chart(fig2, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Score medio per dipartimento")
        if "dipartimento" in df_scored.columns:
            dept_score = df_scored.groupby("dipartimento")["churn_score"].mean().sort_values()
            fig3 = go.Figure(go.Bar(
                x=dept_score.values, y=dept_score.index, orientation="h",
                marker=dict(color=dept_score.values,
                            colorscale=[[0,"#22c55e"],[0.5,"#eab308"],[1,"#ef4444"]],
                            cmin=0, cmax=100),
                text=[f"{v:.0f}" for v in dept_score.values], textposition="outside",
                textfont=dict(color="#94a3b8", size=11),
            ))
            fig3.update_layout(**PLOTLY_THEME, height=260,
                               xaxis=dict(range=[0, 110]))
            st.plotly_chart(fig3, width="stretch")

    with c4:
        st.markdown("#### Top cause di rischio")
        cause_counts = (
            df_scored[df_scored["risk_level"].isin(["Critico","Alto"])]
            ["causa_principale"].value_counts().head(6).reset_index()
        )
        cause_counts.columns = ["Causa","N"]
        fig4 = px.bar(cause_counts, x="N", y="Causa", orientation="h",
                      color="N", color_continuous_scale=["#f97316","#e94560"])
        fig4.update_layout(**PLOTLY_THEME, height=260,
                           coloraxis_showscale=False,
                           yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig4, width="stretch")

    # Scatter ore vs score
    st.markdown("#### Ore settimanali vs Churn score")
    hover_cols = ["nome","cognome","dipartimento","ruolo"] if "nome" in df_scored.columns else ["dipartimento","ruolo"]
    fig5 = px.scatter(
        df_scored, x="ore_settimanali_medie", y="churn_score",
        color="risk_level", size="stipendio_annuo" if "stipendio_annuo" in df_scored.columns else None,
        color_discrete_map=RISK_COLORS, hover_data=hover_cols, opacity=0.75,
        labels={"ore_settimanali_medie":"Ore/sett","churn_score":"Score","risk_level":"Rischio"},
    )
    fig5.update_layout(**PLOTLY_THEME, height=320,
                       legend=dict(font=dict(color="#94a3b8")))
    st.plotly_chart(fig5, width="stretch")


# ══════════════════════════════════════════════
# TAB 2 — CLASSIFICA
# ══════════════════════════════════════════════
with tab_lista:
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        filter_risk = st.multiselect("Rischio", ["Critico","Alto","Medio","Basso"],
                                     default=["Critico","Alto"])
    with fc2:
        if "dipartimento" in df_scored.columns:
            filter_dept = st.multiselect("Dipartimento",
                                         sorted(df_scored["dipartimento"].unique()))
        else:
            filter_dept = []
    with fc3:
        sort_by = st.selectbox("Ordina", ["Score ↓","Score ↑","Nome"])

    df_f = df_scored.copy()
    if filter_risk: df_f = df_f[df_f["risk_level"].isin(filter_risk)]
    if filter_dept: df_f = df_f[df_f["dipartimento"].isin(filter_dept)]
    if sort_by == "Score ↑":   df_f = df_f.sort_values("churn_score", ascending=True)
    elif sort_by == "Nome" and "cognome" in df_f.columns:
        df_f = df_f.sort_values("cognome")

    st.caption(f"{len(df_f)} dipendenti")

    show_cols = [c for c in ["nome","cognome","dipartimento","ruolo",
                              "churn_score","risk_level","causa_principale",
                              "ore_settimanali_medie","anni_senza_promozione",
                              "manager_score","enps_score"] if c in df_f.columns]
    st.dataframe(
        df_f[show_cols],
        width="stretch", height=500, hide_index=True,
        column_config={"churn_score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%.0f"
        )},
    )

    st.download_button(
        "⬇️ Esporta lista filtrata (CSV)",
        data=df_f.to_csv(index=False).encode(),
        file_name="hr_classifica.csv", mime="text/csv",
    )


# ══════════════════════════════════════════════
# TAB 3 — ANALISI AI
# ══════════════════════════════════════════════
with tab_ai:
    st.markdown("#### Analisi AI dei dipendenti a più alto rischio")

    key_ready = (not env_var_list) or any(os.environ.get(ev) for ev in env_var_list) or bool(api_key_input)

    st.markdown(
        f"<p style='color:#6b7280; font-size:0.82rem;'>Modello: <code>{active_model}</code> | "
        f"Anonimizzazione: {'ON 🔐' if use_anonymization else 'OFF ⚠️'}</p>",
        unsafe_allow_html=True,
    )

    if not key_ready:
        st.warning(f"⚠ Inserisci la API key per **{provider}** nella sidebar.")
    else:
        if st.button(f"🤖 Analizza top {top_n} dipendenti", width="content"):
            from utils.llm_engine import analyze_employee
            progress = st.progress(0, text="Avvio analisi AI...")
            results  = []
            top_df   = df_scored.head(top_n)

            try:
                for i, (_, row) in enumerate(top_df.iterrows()):
                    emp = row.to_dict()
                    progress.progress(
                        int(i / top_n * 100),
                        text=f"[{i+1}/{top_n}] {emp.get('nome','')} {emp.get('cognome','')}..."
                    )
                    analysis = analyze_employee(
                        emp, model=active_model,
                        api_key=api_key_input or None,
                        use_anonymization=use_anonymization,
                    )

                    # Mostra errore immediato se fallback
                    if not analysis.get("llm_ok") and analysis.get("error"):
                        progress.empty()
                        st.error(f"❌ Errore LLM: {analysis['error']}")
                        st.stop()

                    results.append({"employee": emp, "analysis": analysis})

                progress.progress(100, text="Completato!")
                time.sleep(0.4)
                progress.empty()
                st.session_state.llm_results = results
                st.success(f"✓ {len(results)} dipendenti analizzati con `{active_model}`")

            except Exception as ex:
                progress.empty()
                import traceback
                st.error(f"❌ Errore: {ex}")
                st.code(traceback.format_exc())

    # Mostra risultati
    llm_results = st.session_state.get("llm_results")
    if llm_results:
        for i, result in enumerate(llm_results, 1):
            emp = result["employee"]
            ana = result["analysis"]
            risk  = str(emp.get("risk_level","Medio"))
            score = emp.get("churn_score", 0)

            with st.expander(
                f"#{i}  {emp.get('nome','')} {emp.get('cognome','')}  —  "
                f"{emp.get('dipartimento','')} · {emp.get('ruolo','')}  "
                f"[Score: {score:.0f}]",
                expanded=(i <= 3),
            ):
                ci, ca = st.columns([1, 2])
                with ci:
                    st.markdown(
                        f"<div class='emp-card' style='border-left:3px solid {score_color(score)};'>"
                        f"<div style='font-family:Syne,sans-serif; font-weight:700;'>"
                        f"{emp.get('nome','')} {emp.get('cognome','')}</div>"
                        f"<div style='font-size:0.82rem; color:#94a3b8;'>"
                        f"{emp.get('ruolo','')} · {emp.get('dipartimento','')}</div>"
                        f"<div style='font-family:DM Mono,monospace; font-size:1.8rem; "
                        f"font-weight:700; color:{score_color(score)}; margin:0.6rem 0;'>"
                        f"{score:.0f}<span style='font-size:1rem;color:#6b7280;'>/100</span></div>"
                        f"{badge_html(risk)}</div>",
                        unsafe_allow_html=True,
                    )
                    for label, val in [
                        ("Età",           f"{emp.get('eta','N/D')} anni"),
                        ("Anzianità",     f"{emp.get('anni_in_azienda',0):.1f} anni"),
                        ("Stipendio",     f"€{emp.get('stipendio_annuo',0):,}"),
                        ("Ore/sett",      f"{emp.get('ore_settimanali_medie',0):.0f}h"),
                        ("HR check-in",   f"{emp.get('giorni_ultimo_checkin_hr','N/D')} gg fa"),
                        ("Manager",       f"{emp.get('manager_score','N/D')}/10"),
                        ("eNPS",          f"{emp.get('enps_score','N/D')}/10"),
                    ]:
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between; "
                            f"padding:3px 0; border-bottom:1px solid #1a1d27; font-size:0.85rem;'>"
                            f"<span style='color:#6b7280;'>{label}</span>"
                            f"<span style='font-weight:500;'>{val}</span></div>",
                            unsafe_allow_html=True,
                        )

                with ca:
                    if ana.get("llm_ok"):
                        urgenza = ana.get("urgenza","media")
                        urg_color = {"immediata":"#ef4444","alta":"#f97316",
                                     "media":"#eab308","bassa":"#22c55e"}.get(urgenza,"#94a3b8")

                        st.markdown("<span style='font-size:0.68rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#06b6d4;'>ANALISI</span>", unsafe_allow_html=True)
                        st.markdown(ana.get("sintesi",""))

                        st.markdown("<span style='font-size:0.68rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#06b6d4;'>CAUSA PRINCIPALE</span>", unsafe_allow_html=True)
                        st.markdown(f"**{ana.get('causa_principale','')}**")

                        cause_sec = ana.get("cause_secondarie", [])
                        if cause_sec:
                            st.markdown("<span style='font-size:0.68rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#06b6d4;'>CAUSE SECONDARIE</span>", unsafe_allow_html=True)
                            st.markdown("  ·  ".join(cause_sec))

                        st.markdown("<span style='font-size:0.68rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#06b6d4;'>AZIONE IMMEDIATA</span>", unsafe_allow_html=True)
                        st.info(ana.get("azione_immediata",""))

                        st.markdown("<span style='font-size:0.68rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#06b6d4;'>AZIONE MEDIO TERMINE</span>", unsafe_allow_html=True)
                        st.markdown(ana.get("azione_medio_termine",""))

                        segnali = ana.get("segnali_positivi",[])
                        if segnali:
                            st.markdown("<span style='font-size:0.68rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#06b6d4;'>PUNTI DI FORZA</span>", unsafe_allow_html=True)
                            st.success("  ·  ".join(segnali))

                        st.markdown("<span style='font-size:0.68rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#06b6d4;'>URGENZA</span>", unsafe_allow_html=True)
                        st.markdown(
                            f"<span style='font-weight:800; font-size:1rem; color:{urg_color}; "
                            f"text-transform:uppercase; letter-spacing:0.08em;'>{urgenza}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning("⚠ Analisi AI non disponibile — fallback")
                        st.markdown(ana.get("sintesi",""))
                        st.markdown(f"**Azione:** {ana.get('azione_immediata','')}")
    elif key_ready:
        st.info("Clicca il bottone sopra per avviare l'analisi AI.")


# ══════════════════════════════════════════════
# TAB 4 — REPORT
# ══════════════════════════════════════════════
with tab_report:
    st.markdown("#### Genera report scaricabili")

    llm_results = st.session_state.get("llm_results")
    if not llm_results:
        st.info("ℹ Esegui prima l'analisi AI per includere i suggerimenti nei report. "
                "Puoi comunque generare un report base con i soli dati ML.")

    def make_fallback_results(df, n):
        return [{"employee": row.to_dict(), "analysis": {
            "sintesi":              f"Score {row['churn_score']:.0f}/100. {row.get('causa_principale','')}.",
            "causa_principale":     row.get("causa_principale","N/D"),
            "cause_secondarie":     [], "azione_immediata": "Check-in HR urgente",
            "azione_medio_termine": "Revisione condizioni", "segnali_positivi": [],
            "urgenza": "alta", "llm_ok": False,
        }} for _, row in df.head(n).iterrows()]

    r1, r2 = st.columns(2)

    with r1:
        st.markdown("""
        <div class='emp-card'>
            <div style='font-size:2rem; margin-bottom:6px;'>📄</div>
            <div style='font-family:Syne,sans-serif; font-weight:700;'>Report PDF</div>
            <div style='color:#6b7280; font-size:0.84rem; margin-top:4px;'>
                Copertina · Riepilogo rischi · Schede per dipendente · Analisi AI
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📄 Genera PDF", width="stretch"):
            with st.spinner("Generazione PDF..."):
                os.makedirs("reports", exist_ok=True)
                from utils.report_generator import generate_pdf
                from datetime import datetime
                results_r = llm_results or make_fallback_results(df_scored, top_n)
                path = f"reports/hr_churn_{company_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                try:
                    generate_pdf(df_scored, results_r, path, company_name)
                    with open(path,"rb") as f:
                        st.download_button("⬇️ Scarica PDF", data=f.read(),
                                           file_name=os.path.basename(path),
                                           mime="application/pdf", width="stretch")
                except Exception as ex:
                    st.error(f"Errore PDF: {ex}")

    with r2:
        st.markdown("""
        <div class='emp-card'>
            <div style='font-size:2rem; margin-bottom:6px;'>📊</div>
            <div style='font-family:Syne,sans-serif; font-weight:700;'>Report Excel</div>
            <div style='color:#6b7280; font-size:0.84rem; margin-top:4px;'>
                Dashboard KPI · Classifica completa · Analisi AI · Grafici
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📊 Genera Excel", width="stretch"):
            with st.spinner("Generazione Excel..."):
                os.makedirs("reports", exist_ok=True)
                from utils.report_generator import generate_excel
                from datetime import datetime
                results_r = llm_results or make_fallback_results(df_scored, top_n)
                path = f"reports/hr_churn_{company_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                try:
                    generate_excel(df_scored, results_r, path, company_name)
                    with open(path,"rb") as f:
                        st.download_button("⬇️ Scarica Excel", data=f.read(),
                                           file_name=os.path.basename(path),
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                           width="stretch")
                except Exception as ex:
                    st.error(f"Errore Excel: {ex}")

    # Metriche anteprima
    st.markdown("---")
    st.markdown("#### Anteprima contenuto report")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dipendenti", len(df_scored))
    m2.metric("Analisi AI", len(llm_results) if llm_results else 0)
    m3.metric("Critici", df_scored["risk_level"].value_counts().get("Critico",0))
    m4.metric("Score medio", f"{df_scored['churn_score'].mean():.1f}")
