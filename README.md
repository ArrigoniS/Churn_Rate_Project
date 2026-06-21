# Churn_Rate_Project
This Repository contains a simple dashboard that let the user analyze dataset of employee using a churn rate ML model and an AI overview for the most critical ones. It's also possible to do a Demo with fake dataset created at the moment.

Employee churn risk prediction system, featuring an interactive dashboard and automated reports.

## Stack

| Layer | Tool |
| --- | --- |
| ML model | XGBoost + scikit-learn |
| Narrative LLM | Ollama (local) / Groq API (production) |
| Dashboard | Streamlit |
| PDF Report | ReportLab |
| Excel Report | openpyxl |
| Deploy | Streamlit Cloud (free) |


## Project structure

```
hr_churn/
├── app.py                    # Streamlit Dashboard (next steps)
├── train_model.py            # XGBoost training pipeline
├── requirements.txt
├── data/
│   ├── generate_dataset.py   # Fake dataset generator with Faker
│   ├── hr_dataset.csv        # Generated dataset (auto-created)
│   ├── hr_scored.csv         # Dataset with scores (auto-created)
│   └── template_dataset.csv  # Template for custom upload
├── models/
│   ├── churn_model.pkl       # Trained model (auto-created)
│   ├── encoders.pkl          # LabelEncoder (auto-created)
│   ├── feature_cols.pkl      # Feature order (auto-created)
│   └── metrics.json          # Evaluation metrics (auto-created)
├── utils/
│   ├── validator.py          # Custom CSV validation
│   ├── llm_engine.py         # Ollama/Groq wrapper (next step)
│   └── report_generator.py   # PDF + Excel (next step)
└── README.md

```

## Features used by the model

| Feature | Description |
| --- | --- |
| eta | Employee age |
| anni_in_azienda | Tenure (years at company) |
| stipendio_annuo | Current gross annual salary |
| num_promozioni | Total promotions received |
| anni_senza_promozione | Years since last promotion |
| crescita_stipendio_percentuale | Average annual salary growth % |
| ore_settimanali_medie | Work overload (average weekly hours) |
| giorni_ultimo_checkin_hr | Frequency of HR check-ins |
| manager_score | Manager feedback score (1-10) |
| enps_score | Employee NPS (1-10) |
| giorni_assenza_anno | Days of absence |
| num_progetti_anno | Number of managed projects |
| ferie_residue | Unused vacation days |
| dipartimento | Department (encoded) |
| ruolo | Role (encoded) |

## Risk levels

| Score | Level | Suggested action |
| --- | --- | --- |
| 0–30 | Low | Periodic monitoring |
| 30–55 | Medium | Proactive HR check-in |
| 55–75 | High | Interview within 2 weeks |
| 75–100 | Critical | Immediate action |
