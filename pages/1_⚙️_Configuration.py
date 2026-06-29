"""
Pagina 1 — Configurazione Dataset / Dataset Configuration
"""
import os, io, json, re
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
from utils.i18n import t, render_lang_toggle

st.set_page_config(page_title="Configuration — HR Churn", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1300px !important; }
.stButton > button { background:#e94560 !important; color:white !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important; }
[data-testid="stSidebar"] { background: #13151c !important; }
</style>
""", unsafe_allow_html=True)

render_lang_toggle()

if "schema" not in st.session_state:
    st.session_state.schema = load_schema()
schema = st.session_state.schema

st.markdown(t("cfg_title"))
st.markdown(f"<p style='color:#6b7280;'>{t('cfg_sub')}</p>", unsafe_allow_html=True)
st.markdown("---")

tab_sim, tab_csv, tab_schema = st.tabs([t("cfg_tab_sim"), t("cfg_tab_csv"), t("cfg_tab_schema")])

# ── TAB 1: DATASET SIMULATO ──
with tab_sim:
    st.markdown(t("cfg_sim_title"))
    st.markdown(f"<p style='color:#6b7280;font-size:0.88rem;'>{t('cfg_sim_sub')}</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(t("cfg_gen_params"))
        n_employees        = st.slider(t("cfg_n_employees"), 100, 2000, 600, 50)
        churn_target       = st.slider(t("cfg_churn_target"), 10, 60, 30, 5, help=t("cfg_churn_help"))
        seed               = st.number_input(t("cfg_seed"), value=42, step=1)
        st.markdown(t("cfg_employee_profile"))
        age_min, age_max   = st.slider(t("cfg_age_range"), 18, 70, (22, 62))
        sen_min, sen_max   = st.slider(t("cfg_seniority"), 1, 20, (0, 12))
        overwork_pct       = st.slider(t("cfg_overwork"), 5, 60, 25, 5, help=t("cfg_overwork_help"))
        low_eng_pct        = st.slider(t("cfg_low_eng"), 5, 50, 20, 5, help=t("cfg_low_eng_help"))
        salary_growth_mean = st.slider(t("cfg_salary_growth"), 0.0, 15.0, 4.5, 0.5)

    with col2:
        st.markdown(t("cfg_dept_title"))
        st.caption(t("cfg_dept_caption"))
        dept_config = {}
        for dept, vals in DEFAULT_CONFIG["departments"].items():
            c1, c2, c3 = st.columns([2,1,1])
            with c1:
                st.markdown(f"<div style='padding-top:8px;font-size:0.88rem;'>{dept}</div>", unsafe_allow_html=True)
            with c2:
                weight = st.number_input(f"Peso##{dept}", 1, 100, vals["weight"], 1, label_visibility="collapsed")
            with c3:
                salary = st.number_input(f"Stipendio##{dept}", 15000, 200000, vals["base_salary"], 1000, label_visibility="collapsed", format="%d")
            dept_config[dept] = {"weight": weight, "base_salary": salary}

    st.markdown("---")
    preview_cols = st.columns(5)
    for col, (label, val, color) in zip(preview_cols, [
        (t("anl_kpi_employees"), str(n_employees), "#e2e8f0"),
        ("Churn target", f"{churn_target}%", "#e94560"),
        ("Overload", f"{overwork_pct}%", "#f97316"),
        ("Low engagement", f"{low_eng_pct}%", "#eab308"),
        ("Seed", str(seed), "#06b6d4"),
    ]):
        with col:
            st.markdown(
                f"<div style='background:#13151c;border:1px solid #252836;border-radius:10px;padding:0.8rem;text-align:center;'>"
                f"<div style='font-size:1.4rem;font-weight:800;color:{color};font-family:DM Mono,monospace;'>{val}</div>"
                f"<div style='font-size:0.7rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;'>{label}</div></div>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(t("cfg_gen_btn"), width="content"):
        config = {
            "n_employees": n_employees, "churn_rate_target": churn_target/100,
            "departments": dept_config, "age_range": [age_min, age_max],
            "seniority_range": [sen_min, sen_max], "overwork_pct": overwork_pct/100,
            "low_engagement_pct": low_eng_pct/100, "salary_growth_mean": salary_growth_mean, "salary_growth_std": 3.0,
        }
        with st.spinner(t("cfg_generating")):
            os.makedirs("data", exist_ok=True)
            df = generate_dataset(config=config, schema=schema, seed=int(seed))
            st.session_state["df_train_raw"] = df
    # df.to_csv("data/hr_dataset.csv", index=False)  # solo locale
            st.session_state.df_train = df
            st.session_state.dataset_config = config
        st.success(f"{t('cfg_gen_success')}: **{len(df)} {t('anl_kpi_employees').lower()}** | Churn rate: **{df['churn_label'].mean():.1%}**")
        st.markdown(t("cfg_preview"))
        st.dataframe(df.head(), width="stretch", hide_index=True)
        st.download_button(t("cfg_download_dataset"), data=df.to_csv(index=False).encode(), file_name="hr_dataset_simulated.csv", mime="text/csv")
        st.markdown(t("cfg_dept_dist"))
        st.bar_chart(df["dipartimento"].value_counts())

# ── TAB 2: CARICA CSV ──
with tab_csv:
    st.markdown(f"### {t('cfg_tab_csv')}")
    col_up, col_tmpl = st.columns([2,1])

    with col_tmpl:
        st.markdown(t("cfg_template_title"))
        st.markdown(f"<p style='color:#6b7280;font-size:0.85rem;'>{t('cfg_template_sub')}</p>", unsafe_allow_html=True)
        st.download_button(t("cfg_download_template"), data=generate_template_csv(schema).encode(),
                           file_name="template_hr_churn.csv", mime="text/csv", width="stretch")
        st.markdown(t("cfg_schema_title"))
        all_cols = {**schema.get("columns",{}), **schema.get("custom_columns",{})}
        st.dataframe(pd.DataFrame([{"Column":c,"Type":m["type"],"Role":m["role"],"Desc":m["desc"]} for c,m in all_cols.items()]),
                     width="stretch", hide_index=True, height=350)

    with col_up:
        st.markdown(t("cfg_upload_title"))
        uploaded = st.file_uploader(t("cfg_upload_label"), type=["csv"], help=t("cfg_upload_help"))
        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                st.markdown(f"**Rows:** {len(df_upload)} | **Columns:** {df_upload.shape[1]}")
                is_valid, errors, warnings = validate_dataset(df_upload)
                for w in warnings: st.warning(f"⚠ {w}")
                for e in errors:   st.error(f"❌ {e}")
                if is_valid:
                    df_clean = impute_missing(df_upload)
                    os.makedirs("data", exist_ok=True)
                    st.session_state["df_train_raw"] = df_clean
                    # df_clean.to_csv("data/hr_dataset.csv", index=False)  # solo locale
                    st.session_state.df_train = df_clean
                    st.success(f"✓ {len(df_clean)} {t('anl_scored_success')}")
                    st.dataframe(df_clean.head(), width="stretch", hide_index=True)
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric(t("anl_kpi_employees"), len(df_clean))
                    if "churn_label" in df_clean.columns:
                        sc2.metric("Churn rate", f"{df_clean['churn_label'].mean():.1%}")
                    sc3.metric("Columns", df_clean.shape[1])
            except Exception as ex:
                st.error(f"{t('error_prefix')} {ex}")

# ── TAB 3: GESTIONE COLONNE ──
with tab_schema:
    st.markdown(f"### {t('cfg_tab_schema')}")
    col_existing, col_add = st.columns([1,1])

    with col_existing:
        st.markdown(t("cfg_default_cols"))
        st.dataframe(pd.DataFrame([
            {"Column":col,"Type":meta["type"],"Role":meta["role"],
             "Range":f"[{meta.get('min','')},{meta.get('max','')}]" if "min" in meta else "—","Desc":meta["desc"]}
            for col,meta in DEFAULT_COLUMNS.items()
        ]), width="stretch", hide_index=True, height=380)

        st.markdown(t("cfg_custom_cols"))
        custom_cols = schema.get("custom_columns", {})
        if custom_cols:
            st.dataframe(pd.DataFrame([
                {"Column":col,"Type":meta["type"],"Role":meta["role"],"Desc":meta["desc"]}
                for col,meta in custom_cols.items()
            ]), width="stretch", hide_index=True)
            to_remove = st.selectbox(t("cfg_remove_col"), options=[t("cfg_remove_select")] + list(custom_cols.keys()))
            if to_remove != t("cfg_remove_select"):
                if st.button(f"{t('cfg_remove_btn')} '{to_remove}'", type="secondary"):
                    schema = remove_custom_column(schema, to_remove)
                    save_schema(schema)
                    st.session_state.schema = schema
                    st.success(f"'{to_remove}' {t('cfg_remove_success')}")
                    st.rerun()
        else:
            st.info(t("cfg_no_custom"))

        st.markdown("---")
        if st.button(t("cfg_reset_btn"), type="secondary"):
            reset_schema()
            st.session_state.schema = load_schema()
            st.success(t("cfg_reset_success"))
            st.rerun()

    with col_add:
        st.markdown(t("cfg_add_col_title"))
        st.markdown(f"<p style='color:#6b7280;font-size:0.85rem;'>{t('cfg_add_col_sub')}</p>", unsafe_allow_html=True)
        with st.form("add_column_form"):
            col_name       = st.text_input(t("cfg_col_name"), placeholder=t("cfg_col_name_ph"), help=t("cfg_col_name_help"))
            col_type       = st.selectbox(t("cfg_col_type"), COLUMN_TYPES)
            col_role       = st.selectbox(t("cfg_col_role"), COLUMN_ROLES, help=t("cfg_col_role_help"))
            col_desc       = st.text_input(t("cfg_col_desc"), placeholder=t("cfg_col_desc_ph"))
            is_categorical = st.checkbox(t("cfg_col_categorical"))
            c1, c2 = st.columns(2)
            with c1: col_min = st.number_input(t("cfg_col_min"), value=0.0) if not is_categorical else None
            with c2: col_max = st.number_input(t("cfg_col_max"), value=100.0) if not is_categorical else None
            cat_options = None
            if is_categorical:
                cat_opts_str = st.text_input(t("cfg_col_options"), placeholder=t("cfg_col_options_ph"))
                cat_options  = [v.strip() for v in cat_opts_str.split(",") if v.strip()]
            submitted = st.form_submit_button(t("cfg_add_btn"))
            if submitted:
                if not col_name:
                    st.error(t("cfg_err_name_empty"))
                elif not re.match(r'^[a-z][a-z0-9_]*$', col_name):
                    st.error(t("cfg_err_name_invalid"))
                elif col_name in {**schema.get("columns",{}), **schema.get("custom_columns",{})}:
                    st.error(f"'{col_name}' {t('cfg_err_name_exists')}")
                else:
                    schema = add_custom_column(schema, col_name, col_type, col_role, col_desc,
                                               min_val=col_min if not is_categorical else None,
                                               max_val=col_max if not is_categorical else None,
                                               categorical=is_categorical)
                    if is_categorical and cat_options:
                        schema["custom_columns"][col_name]["options"] = cat_options
                    save_schema(schema)
                    st.session_state.schema = schema
                    st.success(t("cfg_add_success"))
                    st.rerun()

        st.markdown("---")
        st.markdown(t("cfg_roles_guide"))
        for role, desc_key in [("feature",t("cfg_role_feature")),("pii",t("cfg_role_pii")),
                                ("id",t("cfg_role_id")),("target",t("cfg_role_target")),("ignore",t("cfg_role_ignore"))]:
            st.markdown(
                f"<div style='display:flex;gap:8px;margin-bottom:4px;'>"
                f"<code style='background:#1a1d27;padding:2px 8px;border-radius:4px;font-size:0.8rem;'>{role}</code>"
                f"<span style='font-size:0.85rem;color:#94a3b8;'>{desc_key}</span></div>",
                unsafe_allow_html=True)

    st.markdown("---")
    sch_c1, sch_c2 = st.columns([2,1])
    with sch_c1:
        n_def  = len(schema.get("columns",{}))
        n_cust = len(schema.get("custom_columns",{}))
        n_feat = len([c for c,m in {**schema.get("columns",{}),**schema.get("custom_columns",{})}.items() if m.get("role")=="feature"])
        st.markdown(f"**{t('cfg_schema_summary')}** {n_def} {t('cfg_schema_default_n')} **{n_cust} {t('cfg_schema_custom_n')}** | **{n_feat} {t('cfg_schema_feature_n')}**")
    with sch_c2:
        st.download_button(t("cfg_export_schema"), data=json.dumps(schema,indent=2,ensure_ascii=False).encode(),
                           file_name="schema.json", mime="application/json", width="stretch")

st.markdown("---")
st.markdown(t("cfg_status_title"))
s1, s2, s3 = st.columns(3)
with s1:
    if "df_train" in st.session_state and st.session_state.df_train is not None:
        st.success(f"{t('cfg_dataset_ready')} — {len(st.session_state.df_train)} {t('anl_kpi_employees').lower()}")
    elif (st.session_state.get("df_train_raw") is not None or os.path.exists("data/hr_dataset.csv")):
        df_t = pd.read_csv("data/hr_dataset.csv")
        st.session_state.df_train = df_t
        st.success(f"{t('cfg_dataset_ready')} — {len(df_t)} {t('anl_kpi_employees').lower()}")
    else:
        st.warning(t("cfg_dataset_missing"))
with s2:
    n_cust = len(schema.get("custom_columns",{}))
    if n_cust > 0: st.info(f"🧩 {n_cust} {t('cfg_custom_n')}")
    else: st.info(t("cfg_no_custom_info"))
with s3:
    if (st.session_state.get("training_artifacts") is not None or os.path.exists("models/churn_model.pkl")): st.success(t("cfg_model_ready"))
    else: st.warning(t("go_to_training"))
