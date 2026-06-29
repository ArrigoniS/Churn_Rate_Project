"""
Pagina 3 — Analisi & Scoring / Analysis & Scoring
"""
import os, sys, time, traceback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schema_manager  import load_schema, generate_template_csv
from core.model_trainer   import load_training_artifacts
from core.dataset_builder import generate_dataset
from utils.validator      import validate_dataset, impute_missing
from utils.i18n           import t, render_lang_toggle

st.set_page_config(page_title="Analysis — HR Churn", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }
.stButton > button { background:#e94560 !important; color:white !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important; }
[data-testid="stSidebar"] { background: #13151c !important; }
.kpi-card { background:#13151c; border:1px solid #252836; border-radius:12px; padding:1.2rem 1.4rem; text-align:center; }
.kpi-number { font-family:'DM Mono',monospace; font-size:2.6rem; font-weight:800; line-height:1; margin-bottom:4px; }
.kpi-label  { font-size:0.72rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; }
.emp-card   { background:#13151c; border:1px solid #252836; border-radius:10px; padding:1rem 1.2rem; margin-bottom:6px; }
.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; }
.badge-critico { background:rgba(239,68,68,0.15); color:#ef4444; border:1px solid rgba(239,68,68,0.3); }
.badge-alto    { background:rgba(249,115,22,0.15); color:#f97316; border:1px solid rgba(249,115,22,0.3); }
.badge-medio   { background:rgba(234,179,8,0.15);  color:#eab308; border:1px solid rgba(234,179,8,0.3); }
.badge-basso   { background:rgba(34,197,94,0.15);  color:#22c55e; border:1px solid rgba(34,197,94,0.3); }
</style>
""", unsafe_allow_html=True)

render_lang_toggle()

PLOTLY_THEME = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#94a3b8", size=12),
                    margin=dict(l=10, r=10, t=30, b=10))
RISK_COLORS = {"Critico":"#ef4444","Alto":"#f97316","Medio":"#eab308","Basso":"#22c55e"}

def badge_html(risk):
    return f'<span class="badge badge-{risk.lower()}">{risk}</span>'

def score_color(s):
    if s >= 75: return "#ef4444"
    if s >= 55: return "#f97316"
    if s >= 30: return "#eab308"
    return "#22c55e"

schema    = load_schema()
artifacts = load_training_artifacts()

st.markdown(t("anl_title"))
st.markdown(f"<p style='color:#6b7280;'>{t('anl_sub')}</p>", unsafe_allow_html=True)

if not artifacts:
    st.error(t("no_model_error"))
    st.stop()

st.success(f"✓ {t('anl_model_loaded')} — ROC-AUC: **{artifacts['metrics']['roc_auc']:.3f}** | {artifacts['metrics'].get('trained_at','')[:16].replace('T',' ')}")
st.markdown("---")

def score_dataframe(df):
    model = artifacts["model"]
    encoders = artifacts["encoders"]
    feature_cols = artifacts["feature_cols"]
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            known = set(le.classes_)
            df[col+"_enc"] = df[col].apply(lambda x: le.transform([x])[0] if str(x) in known else 0)
    for c in [c for c in feature_cols if c not in df.columns]:
        df[c] = 0
    proba = model.predict_proba(df[feature_cols].values.astype(float))[:,1]
    df["churn_score"] = (proba*100).round(1)
    df["risk_level"]  = pd.cut(df["churn_score"], bins=[0,30,55,75,100],
                                labels=["Basso","Medio","Alto","Critico"]).astype(str)
    def identify_cause(row):
        causes = []
        if row.get("ore_settimanali_medie",0)>50:       causes.append(("Overload lavorativo", row["ore_settimanali_medie"]-40))
        if row.get("anni_senza_promozione",0)>3:         causes.append(("Stagnazione carriera", row["anni_senza_promozione"]*5))
        if row.get("crescita_stipendio_percentuale",10)<2: causes.append(("Crescita stipendio insufficiente",(2-row["crescita_stipendio_percentuale"])*8))
        if row.get("giorni_ultimo_checkin_hr",0)>180:    causes.append(("Scarso engagement HR", row["giorni_ultimo_checkin_hr"]/20))
        if row.get("manager_score",10)<4:                causes.append(("Rapporto con manager critico",(5-row["manager_score"])*6))
        if row.get("enps_score",10)<4:                   causes.append(("Basso engagement aziendale",(5-row["enps_score"])*5))
        return sorted(causes, key=lambda x:x[1], reverse=True)[0][0] if causes else "Profilo da monitorare"
    df["causa_principale"] = df.apply(identify_cause, axis=1)
    return df.sort_values("churn_score", ascending=False).reset_index(drop=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown(t("anl_source"))
    opt_test = "🧪 " + t("anl_test_gen")
    opt_csv  = "📤 " + t("anl_upload_csv")
    data_source = st.radio(t("anl_source_choice"), [opt_test, opt_csv], index=0)
    df_scored = None

    if opt_test in data_source:
        st.markdown("---")
        st.markdown(t("anl_test_params"))
        n_test    = st.slider(t("anl_employees"), 50, 500, 150, 50)
        test_seed = st.number_input("Seed", value=99, step=1)
        if st.button(t("anl_gen_test_btn"), width="stretch"):
            with st.spinner(t("anl_generating")):
                df_test   = generate_dataset(config={"n_employees":n_test,"churn_rate_target":0.30}, schema=schema, seed=int(test_seed))
                df_scored = score_dataframe(df_test)
                st.session_state["df_scored"] = df_scored
                # df_scored.to_csv("data/hr_scored.csv", index=False)  # solo locale
                st.session_state.df_scored = df_scored
                st.success(f"✓ {n_test} {t('anl_gen_success')}")
    else:
        uploaded = st.file_uploader(t("anl_upload_label"), type=["csv"])
        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                is_valid, errors, warnings = validate_dataset(df_upload)
                for w in warnings: st.warning(f"⚠ {w}")
                for e in errors:   st.error(f"❌ {e}")
                if is_valid:
                    df_clean  = impute_missing(df_upload)
                    df_scored = score_dataframe(df_clean)
                    st.session_state["df_scored"] = df_scored
                # df_scored.to_csv("data/hr_scored.csv", index=False)  # solo locale
                    st.session_state.df_scored = df_scored
                    st.success(f"✓ {len(df_scored)} {t('anl_scored_success')}")
            except Exception as ex:
                st.error(f"{t('error_prefix')} {ex}")
        st.download_button(t("anl_download_tmpl"), data=generate_template_csv(schema).encode(),
                           file_name="template_hr_churn.csv", mime="text/csv", width="stretch")

    if df_scored is None:
        if "df_scored" in st.session_state and st.session_state.df_scored is not None:
            df_scored = st.session_state.df_scored
        elif (st.session_state.get("df_scored") is not None or os.path.exists("data/hr_scored.csv")):
            df_scored = st.session_state.get("df_scored")
            if df_scored is None and os.path.exists("data/hr_scored.csv"):
                df_scored = pd.read_csv("data/hr_scored.csv")
            st.session_state.df_scored = df_scored

    st.markdown("---")
    st.markdown(t("anl_ai_section"))
    from utils.llm_engine import SUGGESTED_MODELS, PROVIDER_ENV_KEYS, get_provider_from_model
    all_models   = [m for models in SUGGESTED_MODELS.values() for m in models]
    active_model = st.selectbox("Model", all_models, index=0)
    custom_model = st.text_input(t("anl_model_custom"), placeholder=t("anl_model_custom_ph"))
    if custom_model.strip(): active_model = custom_model.strip()
    provider     = get_provider_from_model(active_model)
    env_vars     = PROVIDER_ENV_KEYS.get(provider, [])
    env_var_list = [env_vars] if isinstance(env_vars, str) else env_vars
    api_key_input = ""
    if env_var_list:
        api_key_input = st.text_input(f"{t('anl_api_key')} ({provider})", type="password",
                                       placeholder=" / ".join(env_var_list))
        if api_key_input:
            for ev in env_var_list: os.environ[ev] = api_key_input
    else:
        st.caption(t("anl_ollama_note"))
    top_n = st.slider(t("anl_top_n"), 5, 50, 15, 5)
    use_anonymization = st.toggle("🔐 " + t("anl_anon_toggle"), value=True, help=t("anl_anon_help"))
    st.markdown("---")
    company_name = st.text_input(t("anl_company"), value="Company")

# ── GUARD ──
if df_scored is None:
    st.markdown(
        f"<div style='text-align:center;padding:5rem 2rem;color:#4b5563;'>"
        f"<div style='font-size:3rem;'>🔮</div>"
        f"<div style='font-size:1.1rem;font-family:Syne,sans-serif;font-weight:700;color:#6b7280;margin-top:1rem;'>{t('anl_no_dataset')}</div>"
        f"<div style='font-size:0.88rem;margin-top:0.5rem;'>{t('anl_no_dataset_sub')}</div></div>",
        unsafe_allow_html=True)
    st.stop()

# ── TABS ──
tab_dash, tab_lista, tab_ai, tab_report = st.tabs([
    t("anl_tab_dash"), t("anl_tab_list"), t("anl_tab_ai"), t("anl_tab_report")
])

# ── TAB DASHBOARD ──
with tab_dash:
    risk_counts = df_scored["risk_level"].value_counts()
    total       = len(df_scored)
    n_critico   = risk_counts.get("Critico",0)
    n_alto      = risk_counts.get("Alto",0)
    avg_score   = df_scored["churn_score"].mean()

    k1,k2,k3,k4,k5 = st.columns(5)
    for col, num, label, color in [
        (k1, str(total),              t("anl_kpi_employees"), "#e2e8f0"),
        (k2, str(n_critico),          t("anl_kpi_critical"),  "#ef4444"),
        (k3, str(n_alto),             t("anl_kpi_high"),      "#f97316"),
        (k4, str(n_critico+n_alto),   t("anl_kpi_at_risk"),   "#e94560"),
        (k5, f"{avg_score:.0f}/100",  t("anl_kpi_avg_score"), "#06b6d4"),
    ]:
        with col:
            st.markdown(f"<div class='kpi-card'><div class='kpi-number' style='color:{color};'>{num}</div><div class='kpi-label'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(t("anl_chart_pie"))
        pie_data = df_scored["risk_level"].value_counts().reset_index()
        pie_data.columns = ["Level","N"]
        fig = go.Figure(go.Pie(labels=pie_data["Level"], values=pie_data["N"],
            marker_colors=[RISK_COLORS.get(l,"#666") for l in pie_data["Level"]],
            hole=0.55, textfont_size=12, textfont_color="#e2e8f0"))
        fig.update_layout(**PLOTLY_THEME, height=260, legend=dict(font=dict(color="#94a3b8")))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown(t("anl_chart_dept"))
        if "dipartimento" in df_scored.columns:
            dept_risk = df_scored[df_scored["risk_level"].isin(["Critico","Alto"])].groupby(["dipartimento","risk_level"]).size().reset_index(name="n")
            fig2 = px.bar(dept_risk, x="n", y="dipartimento", color="risk_level", orientation="h",
                          color_discrete_map=RISK_COLORS, labels={"n":"","dipartimento":"","risk_level":"Risk"})
            fig2.update_layout(**PLOTLY_THEME, height=260, yaxis=dict(categoryorder="total ascending"), legend=dict(font=dict(color="#94a3b8")))
            st.plotly_chart(fig2, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(t("anl_chart_score_dept"))
        if "dipartimento" in df_scored.columns:
            dept_score = df_scored.groupby("dipartimento")["churn_score"].mean().sort_values()
            fig3 = go.Figure(go.Bar(x=dept_score.values, y=dept_score.index, orientation="h",
                marker=dict(color=dept_score.values, colorscale=[[0,"#22c55e"],[0.5,"#eab308"],[1,"#ef4444"]], cmin=0, cmax=100),
                text=[f"{v:.0f}" for v in dept_score.values], textposition="outside", textfont=dict(color="#94a3b8",size=11)))
            fig3.update_layout(**PLOTLY_THEME, height=260, xaxis=dict(range=[0,110]))
            st.plotly_chart(fig3, width="stretch")
    with c4:
        st.markdown(t("anl_chart_causes"))
        cause_counts = df_scored[df_scored["risk_level"].isin(["Critico","Alto"])]["causa_principale"].value_counts().head(6).reset_index()
        cause_counts.columns = ["Cause","N"]
        fig4 = px.bar(cause_counts, x="N", y="Cause", orientation="h", color="N", color_continuous_scale=["#f97316","#e94560"])
        fig4.update_layout(**PLOTLY_THEME, height=260, coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig4, width="stretch")
    st.markdown(t("anl_chart_scatter"))
    hover_cols = ["nome","cognome","dipartimento","ruolo"] if "nome" in df_scored.columns else ["dipartimento","ruolo"]
    fig5 = px.scatter(df_scored, x="ore_settimanali_medie", y="churn_score", color="risk_level",
                      size="stipendio_annuo" if "stipendio_annuo" in df_scored.columns else None,
                      color_discrete_map=RISK_COLORS, hover_data=hover_cols, opacity=0.75,
                      labels={"ore_settimanali_medie":"Hours/week","churn_score":"Score","risk_level":"Risk"})
    fig5.update_layout(**PLOTLY_THEME, height=320, legend=dict(font=dict(color="#94a3b8")))
    st.plotly_chart(fig5, width="stretch")

# ── TAB CLASSIFICA ──
with tab_lista:
    fc1, fc2, fc3 = st.columns([2,2,1])
    with fc1:
        filter_risk = st.multiselect(t("anl_filter_risk"), ["Critico","Alto","Medio","Basso"], default=["Critico","Alto"])
    with fc2:
        filter_dept = st.multiselect(t("anl_filter_dept"), sorted(df_scored["dipartimento"].unique()) if "dipartimento" in df_scored.columns else [])
    with fc3:
        sort_by = st.selectbox(t("anl_sort_by"), [t("anl_sort_desc"), t("anl_sort_asc"), t("anl_sort_name")])

    df_f = df_scored.copy()
    if filter_risk: df_f = df_f[df_f["risk_level"].isin(filter_risk)]
    if filter_dept: df_f = df_f[df_f["dipartimento"].isin(filter_dept)]
    if t("anl_sort_asc") in sort_by: df_f = df_f.sort_values("churn_score", ascending=True)
    elif t("anl_sort_name") in sort_by and "cognome" in df_f.columns: df_f = df_f.sort_values("cognome")

    st.caption(f"{len(df_f)} {t('anl_kpi_employees').lower()}")
    show_cols = [c for c in ["nome","cognome","dipartimento","ruolo","churn_score","risk_level","causa_principale","ore_settimanali_medie","anni_senza_promozione","manager_score","enps_score"] if c in df_f.columns]
    st.dataframe(df_f[show_cols], width="stretch", height=500, hide_index=True,
                 column_config={"churn_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f")})
    st.download_button(t("anl_export_csv"), data=df_f.to_csv(index=False).encode(), file_name="hr_rankings.csv", mime="text/csv")

# ── TAB AI ──
with tab_ai:
    st.markdown(t("anl_ai_title"))
    key_ready = (not env_var_list) or any(os.environ.get(ev) for ev in env_var_list) or bool(api_key_input)
    anon_status = t("anl_ai_anon_on") if use_anonymization else t("anl_ai_anon_off")
    st.markdown(f"<p style='color:#6b7280;font-size:0.82rem;'>{t('anl_ai_model_active')} <code>{active_model}</code> | {anon_status}</p>", unsafe_allow_html=True)

    if not key_ready:
        st.warning(f"⚠ {t('anl_ai_key_warning')} **{provider}** {t('anl_ai_key_warn2')}")
    else:
        if st.button(f"{t('anl_ai_btn')} {top_n} {t('anl_ai_btn2')}", width="content"):
            from utils.llm_engine import analyze_employee
            progress = st.progress(0, text=t("anl_ai_starting"))
            results  = []
            try:
                for i, (_, row) in enumerate(df_scored.head(top_n).iterrows()):
                    emp = row.to_dict()
                    progress.progress(int(i/top_n*100), text=f"[{i+1}/{top_n}] {emp.get('nome','')} {emp.get('cognome','')}...")
                    analysis = analyze_employee(emp, model=active_model, api_key=api_key_input or None, use_anonymization=use_anonymization)
                    if not analysis.get("llm_ok") and analysis.get("error"):
                        progress.empty()
                        st.error(f"{t('anl_ai_error')} {analysis['error']}")
                        st.stop()
                    results.append({"employee": emp, "analysis": analysis})
                progress.progress(100, text="Done!")
                time.sleep(0.4)
                progress.empty()
                st.session_state.llm_results = results
                st.success(f"✓ {len(results)} {t('anl_ai_success')} `{active_model}`")
            except Exception as ex:
                progress.empty()
                st.error(f"{t('anl_ai_error')} {ex}")
                st.code(traceback.format_exc())

    llm_results = st.session_state.get("llm_results")
    if llm_results:
        for i, result in enumerate(llm_results, 1):
            emp   = result["employee"]
            ana   = result["analysis"]
            risk  = str(emp.get("risk_level","Medio"))
            score = emp.get("churn_score",0)
            with st.expander(f"#{i}  {emp.get('nome','')} {emp.get('cognome','')}  —  {emp.get('dipartimento','')} · {emp.get('ruolo','')}  [Score: {score:.0f}]", expanded=(i<=3)):
                ci, ca = st.columns([1,2])
                with ci:
                    st.markdown(
                        f"<div class='emp-card' style='border-left:3px solid {score_color(score)};'>"
                        f"<div style='font-family:Syne,sans-serif;font-weight:700;'>{emp.get('nome','')} {emp.get('cognome','')}</div>"
                        f"<div style='font-size:0.82rem;color:#94a3b8;'>{emp.get('ruolo','')} · {emp.get('dipartimento','')}</div>"
                        f"<div style='font-family:DM Mono,monospace;font-size:1.8rem;font-weight:700;color:{score_color(score)};margin:0.6rem 0;'>{score:.0f}<span style='font-size:1rem;color:#6b7280;'>/100</span></div>"
                        f"{badge_html(risk)}</div>", unsafe_allow_html=True)
                    for label, val in [
                        ("Age" if "en" in str(st.session_state.get("lang","it")) else "Età", f"{emp.get('eta','N/D')}"),
                        ("Seniority" if "en" in str(st.session_state.get("lang","it")) else "Anzianità", f"{emp.get('anni_in_azienda',0):.1f} yr"),
                        ("Salary" if "en" in str(st.session_state.get("lang","it")) else "Stipendio", f"€{emp.get('stipendio_annuo',0):,}"),
                        ("Hours/wk", f"{emp.get('ore_settimanali_medie',0):.0f}h"),
                        ("HR check-in", f"{emp.get('giorni_ultimo_checkin_hr','N/D')} d"),
                        ("Manager", f"{emp.get('manager_score','N/D')}/10"),
                        ("eNPS", f"{emp.get('enps_score','N/D')}/10"),
                    ]:
                        st.markdown(f"<div style='display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1a1d27;font-size:0.85rem;'><span style='color:#6b7280;'>{label}</span><span style='font-weight:500;'>{val}</span></div>", unsafe_allow_html=True)
                with ca:
                    if ana.get("llm_ok"):
                        urgenza   = ana.get("urgenza","media")
                        urg_color = {"immediata":"#ef4444","alta":"#f97316","media":"#eab308","bassa":"#22c55e","immediate":"#ef4444","high":"#f97316","medium":"#eab308","low":"#22c55e"}.get(urgenza,"#94a3b8")
                        for lbl_key, content, is_info, is_success in [
                            (t("anl_analysis_lbl"),    ana.get("sintesi",""),           False, False),
                            (t("anl_main_cause_lbl"),  f"**{ana.get('causa_principale','')}**", False, False),
                            (t("anl_action_imm_lbl"),  ana.get("azione_immediata",""),  True,  False),
                            (t("anl_action_mt_lbl"),   ana.get("azione_medio_termine",""), False, False),
                        ]:
                            st.markdown(f"<span style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#06b6d4;'>{lbl_key}</span>", unsafe_allow_html=True)
                            if is_info: st.info(content)
                            else: st.markdown(content)

                        cause_sec = ana.get("cause_secondarie",[])
                        if cause_sec:
                            st.markdown(f"<span style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#06b6d4;'>{t('anl_sec_causes_lbl')}</span>", unsafe_allow_html=True)
                            st.markdown("  ·  ".join(cause_sec))

                        segnali = ana.get("segnali_positivi",[])
                        if segnali:
                            st.markdown(f"<span style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#06b6d4;'>{t('anl_strengths_lbl')}</span>", unsafe_allow_html=True)
                            st.success("  ·  ".join(segnali))

                        st.markdown(f"<span style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#06b6d4;'>{t('anl_urgency_lbl')}</span>", unsafe_allow_html=True)
                        st.markdown(f"<span style='font-weight:800;font-size:1rem;color:{urg_color};text-transform:uppercase;letter-spacing:0.08em;'>{urgenza}</span>", unsafe_allow_html=True)
                    else:
                        st.warning(t("anl_ai_fallback"))
                        st.markdown(ana.get("sintesi",""))
                        st.markdown(f"{t('anl_ai_action')} {ana.get('azione_immediata','')}")
    elif key_ready:
        st.info(t("anl_ai_cta"))

# ── TAB REPORT ──
with tab_report:
    st.markdown(t("anl_report_title"))
    llm_results = st.session_state.get("llm_results")
    if not llm_results:
        st.info(t("anl_report_no_ai"))

    def make_fallback_results(df, n):
        return [{"employee": row.to_dict(), "analysis": {
            "sintesi": f"Score {row['churn_score']:.0f}/100. {row.get('causa_principale','')}.",
            "causa_principale": row.get("causa_principale","N/D"),
            "cause_secondarie":[], "azione_immediata":"Urgent HR check-in",
            "azione_medio_termine":"Review conditions", "segnali_positivi":[],
            "urgenza":"alta","llm_ok":False,
        }} for _, row in df.head(n).iterrows()]

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"<div class='emp-card'><div style='font-size:2rem;margin-bottom:6px;'>📄</div><div style='font-family:Syne,sans-serif;font-weight:700;'>{t('anl_pdf_title')}</div><div style='color:#6b7280;font-size:0.84rem;margin-top:4px;'>{t('anl_pdf_desc')}</div></div>", unsafe_allow_html=True)
        if st.button(t("anl_pdf_btn"), width="stretch"):
            with st.spinner(t("anl_pdf_generating")):
                os.makedirs("reports", exist_ok=True)
                from utils.report_generator import generate_pdf
                from datetime import datetime
                results_r = llm_results or make_fallback_results(df_scored, top_n)
                path = f"reports/hr_churn_{company_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                try:
                    generate_pdf(df_scored, results_r, path, company_name)
                    with open(path,"rb") as f:
                        st.download_button(t("anl_pdf_download"), data=f.read(), file_name=os.path.basename(path), mime="application/pdf", width="stretch")
                except Exception as ex:
                    st.error(f"{t('error_prefix')} {ex}")

    with r2:
        st.markdown(f"<div class='emp-card'><div style='font-size:2rem;margin-bottom:6px;'>📊</div><div style='font-family:Syne,sans-serif;font-weight:700;'>{t('anl_xlsx_title')}</div><div style='color:#6b7280;font-size:0.84rem;margin-top:4px;'>{t('anl_xlsx_desc')}</div></div>", unsafe_allow_html=True)
        if st.button(t("anl_xlsx_btn"), width="stretch"):
            with st.spinner(t("anl_xlsx_generating")):
                os.makedirs("reports", exist_ok=True)
                from utils.report_generator import generate_excel
                from datetime import datetime
                results_r = llm_results or make_fallback_results(df_scored, top_n)
                path = f"reports/hr_churn_{company_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                try:
                    generate_excel(df_scored, results_r, path, company_name)
                    with open(path,"rb") as f:
                        st.download_button(t("anl_xlsx_download"), data=f.read(), file_name=os.path.basename(path), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
                except Exception as ex:
                    st.error(f"{t('error_prefix')} {ex}")

    st.markdown("---")
    st.markdown(t("anl_report_preview"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("anl_metric_employees"), len(df_scored))
    m2.metric(t("anl_metric_ai"), len(llm_results) if llm_results else 0)
    m3.metric(t("anl_metric_critical"), df_scored["risk_level"].value_counts().get("Critico",0))
    m4.metric(t("anl_metric_avg"), f"{df_scored['churn_score'].mean():.1f}")
