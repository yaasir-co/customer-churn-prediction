
from pathlib import Path
import io, json, math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "customer_churn.csv"
MODEL_DIR = BASE / "models"

st.set_page_config(
    page_title="Churn Intelligence Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
  --bg:#050a14;
  --panel:#0b1322;
  --panel2:#101a2d;
  --line:rgba(255,255,255,.08);
  --muted:#8ea0b8;
  --text:#f4f8ff;
  --violet:#8b5cf6;
  --cyan:#22d3ee;
  --green:#34d399;
  --red:#fb7185;
  --amber:#fbbf24;
}
.stApp {
  background:
    radial-gradient(circle at 8% 0%, rgba(139,92,246,.17), transparent 28%),
    radial-gradient(circle at 92% 2%, rgba(34,211,238,.10), transparent 24%),
    linear-gradient(180deg,#050a14,#07101c 45%,#050a14);
}
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#07101d,#081321);
  border-right:1px solid var(--line);
}
[data-testid="stSidebar"] * { color:#e6eef9; }
.hero {
  position:relative; overflow:hidden;
  padding:34px 36px;
  border:1px solid var(--line);
  border-radius:28px;
  background:
    linear-gradient(135deg,rgba(139,92,246,.20),rgba(34,211,238,.06) 55%,rgba(255,255,255,.025));
  box-shadow:0 28px 80px rgba(0,0,0,.28);
  margin-bottom:22px;
}
.hero:after {
  content:"";position:absolute;width:240px;height:240px;border-radius:50%;
  right:-80px;top:-100px;background:rgba(34,211,238,.10);filter:blur(10px);
}
.hero-kicker {
  display:inline-block;padding:6px 10px;border:1px solid rgba(255,255,255,.10);
  border-radius:999px;font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#b9c7db;
}
.hero h1 { font-size:clamp(38px,6vw,72px);line-height:.96;margin:16px 0 14px;letter-spacing:-2.5px; }
.hero p { max-width:900px;color:#9fb0c7;font-size:17px;margin:0; }
.glow {
  background:linear-gradient(90deg,#fff,#c4b5fd 42%,#67e8f9);
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.lux-card {
  border:1px solid var(--line);border-radius:22px;
  background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.025));
  padding:20px;box-shadow:0 16px 50px rgba(0,0,0,.20);
}
.result-high {
  border:1px solid rgba(251,113,133,.32);background:rgba(251,113,133,.07);
  border-radius:22px;padding:22px;
}
.result-low {
  border:1px solid rgba(52,211,153,.32);background:rgba(52,211,153,.07);
  border-radius:22px;padding:22px;
}
.pill {
  display:inline-block;padding:5px 9px;border-radius:999px;margin:2px;
  border:1px solid rgba(255,255,255,.09);background:rgba(255,255,255,.035);
  color:#b9c7db;font-size:12px;
}
.note {
  border-left:3px solid #8b5cf6;padding:12px 14px;background:rgba(139,92,246,.07);
  border-radius:0 14px 14px 0;color:#c4cee0;
}
[data-testid="stMetric"] {
  border:1px solid var(--line);padding:13px 15px;border-radius:18px;background:rgba(255,255,255,.025);
}
.stButton>button, .stDownloadButton>button, a[data-testid="stLinkButton"] {
  border-radius:13px!important;font-weight:700!important;
}
hr { border-color:rgba(255,255,255,.08)!important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_meta():
    return json.loads((MODEL_DIR / "model_metrics.json").read_text(encoding="utf-8"))

@st.cache_data
def load_importance():
    return pd.read_csv(MODEL_DIR / "feature_importance.csv")

@st.cache_data
def load_thresholds():
    return pd.read_csv(MODEL_DIR / "threshold_metrics.csv")

@st.cache_resource
def load_models():
    meta = load_meta()
    result = {}
    for name in meta["metrics"]:
        slug = name.lower().replace("-", "").replace(" ", "_")
        result[name] = joblib.load(MODEL_DIR / f"{slug}.joblib")
    return result

df = load_data()
meta = load_meta()
models = load_models()
importance = load_importance()
threshold_table = load_thresholds()
best_name = meta["best_model"]
best_model = models[best_name]

PAGES = [
    "◈ Executive Overview",
    "◎ Live Prediction",
    "◇ Scenario Simulator",
    "⇧ Batch Prediction",
    "▦ Model Arena",
    "∿ Mathematics Lab",
    "◫ Exploratory Analysis",
    "⌁ Feature Intelligence",
    "⚙ Threshold Studio",
    "↗ ML Engineer Learning Path",
    "▶ Tutorials",
    "⌘ Interview Preparation",
    "⌂ Production Architecture",
    "⚖ Responsible AI",
    "↓ Download Center",
]

with st.sidebar:
    st.markdown("## ◈ Churn Intelligence")
    st.caption("AI/ML Engineer Learning Studio")
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown("**Best model on this dataset**")
    st.success(best_name)
    st.caption("Selected by held-out ROC-AUC, not by marketing preference.")
    st.divider()
    st.caption("Python · Pandas · Scikit-learn · Plotly · Streamlit")

def hero(kicker, title, text):
    st.markdown(
        f"""<div class="hero"><span class="hero-kicker">{kicker}</span>
        <h1>{title}</h1><p>{text}</p></div>""",
        unsafe_allow_html=True
    )

def model_metric_table():
    rows = []
    for name, m in meta["metrics"].items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC-AUC": m["roc_auc"],
            "Log Loss": m["log_loss"],
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)

def customer_form(prefix="main"):
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12, key=f"{prefix}_tenure")
        monthly = st.slider("Monthly charges ($)", 18.0, 150.0, 82.0, 1.0, key=f"{prefix}_monthly")
        contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"], key=f"{prefix}_contract")
        internet = st.selectbox("Internet service", ["DSL","Fiber optic","No"], key=f"{prefix}_internet")
        payment = st.selectbox("Payment method", ["Electronic check","Mailed check","Bank transfer","Credit card"], key=f"{prefix}_payment")
        paperless = st.selectbox("Paperless billing", ["Yes","No"], key=f"{prefix}_paperless")
    with c2:
        tech = st.selectbox("Tech support", ["No","Yes"], key=f"{prefix}_tech")
        security = st.selectbox("Online security", ["No","Yes"], key=f"{prefix}_security")
        protection = st.selectbox("Device protection", ["No","Yes"], key=f"{prefix}_protection")
        streaming_tv = st.selectbox("Streaming TV", ["No","Yes"], key=f"{prefix}_tv")
        streaming_movies = st.selectbox("Streaming movies", ["No","Yes"], key=f"{prefix}_movies")
        phone = st.selectbox("Phone service", ["Yes","No"], key=f"{prefix}_phone")
    with c3:
        multiple = st.selectbox("Multiple lines", ["No","Yes"], key=f"{prefix}_multiple")
        senior = st.selectbox("Senior citizen", ["No","Yes"], key=f"{prefix}_senior")
        partner = st.selectbox("Partner", ["No","Yes"], key=f"{prefix}_partner")
        dependents = st.selectbox("Dependents", ["No","Yes"], key=f"{prefix}_dependents")
        default_total = float(round(monthly * max(tenure,1), 2))
        total = st.number_input("Total charges ($)", 0.0, 12000.0, default_total, 10.0, key=f"{prefix}_total")

    return pd.DataFrame([{
        "tenure_months":tenure,
        "monthly_charges":monthly,
        "total_charges":total,
        "contract_type":contract,
        "internet_service":internet,
        "tech_support":tech,
        "online_security":security,
        "device_protection":protection,
        "streaming_tv":streaming_tv,
        "streaming_movies":streaming_movies,
        "phone_service":phone,
        "multiple_lines":multiple,
        "payment_method":payment,
        "paperless_billing":paperless,
        "senior_citizen":senior,
        "partner":partner,
        "dependents":dependents,
    }])

def explain_profile(row):
    r = row.iloc[0]
    reasons = []
    if r["contract_type"] == "Month-to-month": reasons.append(("Higher risk","Month-to-month contract"))
    if r["contract_type"] == "Two year": reasons.append(("Lower risk","Two-year contract"))
    if r["internet_service"] == "Fiber optic": reasons.append(("Higher risk","Fiber-optic service pattern"))
    if r["tech_support"] == "No": reasons.append(("Higher risk","No tech support"))
    if r["online_security"] == "No": reasons.append(("Higher risk","No online security"))
    if r["payment_method"] == "Electronic check": reasons.append(("Higher risk","Electronic-check payment"))
    if r["tenure_months"] < 12: reasons.append(("Higher risk","Short tenure"))
    if r["tenure_months"] > 48: reasons.append(("Lower risk","Long tenure"))
    if r["monthly_charges"] > 100: reasons.append(("Higher risk","High monthly charges"))
    if not reasons: reasons.append(("Neutral","Mixed customer profile"))
    return reasons

if page == "◈ Executive Overview":
    hero("Project Command Center","Customer Churn <span class='glow'>Intelligence Studio</span>",
         "A complete end-to-end machine-learning learning environment: business framing, data, mathematics, model training, evaluation, explainability, deployment, responsible AI and interview preparation.")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Dataset", f"{meta['dataset_rows']:,}")
    c2.metric("Churn rate", f"{meta['churn_rate']*100:.1f}%")
    c3.metric("Models", len(meta["metrics"]))
    c4.metric("Best ROC-AUC", f"{meta['metrics'][best_name]['roc_auc']:.3f}")
    c5.metric("Best model", best_name)

    st.subheader("What you gain from this project")
    a,b,c = st.columns(3)
    with a:
        st.markdown("""<div class="lux-card"><h3>ML Engineering</h3>
        <p>Preprocessing pipelines, model training, serialization, inference, batch scoring, thresholding and model comparison.</p></div>""", unsafe_allow_html=True)
    with b:
        st.markdown("""<div class="lux-card"><h3>Data Science</h3>
        <p>EDA, business metrics, class imbalance, feature analysis, model evaluation and trade-off interpretation.</p></div>""", unsafe_allow_html=True)
    with c:
        st.markdown("""<div class="lux-card"><h3>Production Thinking</h3>
        <p>Monitoring, drift, human review, versioning, API architecture, governance and responsible deployment.</p></div>""", unsafe_allow_html=True)

    st.subheader("Model leaderboard")
    board = model_metric_table()
    st.dataframe(board.style.format({
        "Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}",
        "F1":"{:.3f}","ROC-AUC":"{:.3f}","Log Loss":"{:.3f}"
    }), use_container_width=True, hide_index=True)

    st.markdown(f"""<div class="note"><b>Why {best_name} is highlighted:</b>
    it achieved the highest ROC-AUC on the held-out test split in this project. In a real business,
    the preferred model can change if recall, precision, latency, interpretability or operating cost matters more.</div>""", unsafe_allow_html=True)

elif page == "◎ Live Prediction":
    hero("Inference","Live <span class='glow'>Churn Prediction</span>",
         "Create a customer profile, score it with any trained model, compare probabilities and translate the result into business language.")

    row = customer_form("predict")
    c1,c2 = st.columns([1,1])
    with c1:
        selected_model = st.selectbox("Inference model", list(models.keys()), index=list(models.keys()).index(best_name))
        threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)
    with c2:
        st.markdown("**Decision rule**")
        st.latex(r"\hat{y} = 1 \quad \text{if} \quad P(\mathrm{churn}\mid x)\ge \tau")

    if st.button("Run prediction", type="primary", use_container_width=True):
        probs = {name: float(model.predict_proba(row)[0,1]) for name,model in models.items()}
        p = probs[selected_model]
        high = p >= threshold
        css = "result-high" if high else "result-low"
        label = "High churn risk" if high else "Lower churn risk"
        st.markdown(f"""<div class="{css}">
        <div style="color:#9fb0c7">Decision</div>
        <div style="font-size:28px;font-weight:800">{label}</div>
        <div style="font-size:54px;font-weight:900;margin-top:8px">{p*100:.1f}%</div>
        <div style="color:#9fb0c7">predicted probability using {selected_model}</div>
        </div>""", unsafe_allow_html=True)

        st.subheader("Cross-model comparison")
        compare = pd.DataFrame({"Model":probs.keys(),"Churn Probability":probs.values()}).sort_values("Churn Probability", ascending=False)
        fig = px.bar(compare, x="Churn Probability", y="Model", orientation="h", text_auto=".1%")
        fig.update_layout(height=330, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Profile signals")
        for typ,reason in explain_profile(row):
            st.write(f"**{typ}:** {reason}")

elif page == "◇ Scenario Simulator":
    hero("Counterfactual Thinking","Scenario <span class='glow'>Simulator</span>",
         "Change one customer characteristic at a time and observe how predicted churn changes. This teaches sensitivity analysis and counterfactual reasoning.")

    base = customer_form("scenario")
    model_name = st.selectbox("Model", list(models.keys()), index=list(models.keys()).index(best_name), key="scenario_model")
    base_p = float(models[model_name].predict_proba(base)[0,1])

    scenarios = []
    for contract in ["Month-to-month","One year","Two year"]:
        temp = base.copy(); temp.loc[0,"contract_type"] = contract
        scenarios.append(("Contract → "+contract, float(models[model_name].predict_proba(temp)[0,1])))
    for support in ["No","Yes"]:
        temp = base.copy(); temp.loc[0,"tech_support"] = support
        scenarios.append(("Tech support → "+support, float(models[model_name].predict_proba(temp)[0,1])))
    for security in ["No","Yes"]:
        temp = base.copy(); temp.loc[0,"online_security"] = security
        scenarios.append(("Security → "+security, float(models[model_name].predict_proba(temp)[0,1])))

    st.metric("Base churn probability", f"{base_p:.1%}")
    scen = pd.DataFrame(scenarios, columns=["Scenario","Probability"])
    scen["Change"] = scen["Probability"] - base_p
    fig = px.bar(scen.sort_values("Probability"), x="Probability", y="Scenario", orientation="h",
                 hover_data={"Change":":.1%"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Counterfactual changes are demonstrations of model behaviour, not causal proof.")

elif page == "⇧ Batch Prediction":
    hero("Operational ML","Batch <span class='glow'>Prediction</span>",
         "Upload many customers in CSV format, generate churn probabilities, classify by threshold and export the scored file.")

    st.download_button("Download sample CSV", (BASE/"data"/"sample_batch.csv").read_bytes(),
                       file_name="sample_batch.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    batch_model = st.selectbox("Batch model", list(models.keys()), index=list(models.keys()).index(best_name))
    batch_threshold = st.slider("Batch threshold",0.10,0.90,0.50,0.01,key="batch_threshold")
    if uploaded:
        bdf = pd.read_csv(uploaded)
        required = [c for c in df.columns if c not in ["customer_id","churn"]]
        missing = [c for c in required if c not in bdf.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            scored = bdf.copy()
            scored["churn_probability"] = models[batch_model].predict_proba(scored[required])[:,1]
            scored["risk_class"] = np.where(scored["churn_probability"] >= batch_threshold, "High risk","Lower risk")
            st.dataframe(scored.head(50), use_container_width=True)
            st.download_button("Download scored customers", scored.to_csv(index=False).encode(),
                               file_name="churn_scored_customers.csv", mime="text/csv")

elif page == "▦ Model Arena":
    hero("Evaluation","Model <span class='glow'>Arena</span>",
         "Compare algorithms on the same held-out test set and learn why the best model depends on the business objective.")

    board = model_metric_table()
    st.dataframe(board.style.format({
        "Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}",
        "F1":"{:.3f}","ROC-AUC":"{:.3f}","Log Loss":"{:.3f}"
    }), use_container_width=True, hide_index=True)

    metric = st.selectbox("Compare by metric", ["ROC-AUC","F1","Recall","Precision","Accuracy"])
    fig = px.bar(board.sort_values(metric), x=metric, y="Model", orientation="h", text_auto=".3f")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("When each model is useful")
    st.markdown("""
- **Logistic Regression** — strong baseline, fast, interpretable coefficients and easy probability output.
- **Random Forest** — captures nonlinear interactions and is robust on many tabular datasets.
- **Gradient Boosting** — often strong on structured/tabular prediction problems by sequentially correcting errors.
- **K-Nearest Neighbors** — intuitive instance-based learning, but inference becomes expensive as data grows.
""")
    st.markdown("**Best on this dataset by ROC-AUC:** " + best_name)

elif page == "∿ Mathematics Lab":
    hero("Foundations","Mathematics <span class='glow'>Behind Churn Prediction</span>",
         "Understand the probability, optimization and evaluation formulas instead of treating machine learning as a black box.")

    tab1,tab2,tab3,tab4,tab5 = st.tabs(["Logistic Regression","Loss Function","Confusion Matrix","Metrics","Thresholds"])
    with tab1:
        st.markdown("### Linear score")
        st.latex(r"z = w_0 + w_1x_1 + w_2x_2 + \cdots + w_nx_n")
        st.markdown("### Sigmoid transformation")
        st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}")
        st.write("The sigmoid maps any real-valued score into a probability between 0 and 1.")
        z = st.slider("Try z", -8.0,8.0,0.0,.1)
        st.metric("σ(z)", f"{1/(1+math.exp(-z)):.4f}")
    with tab2:
        st.markdown("### Binary cross-entropy / log loss")
        st.latex(r"\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}[y_i\log(p_i)+(1-y_i)\log(1-p_i)]")
        st.write("Wrong confident predictions are penalized heavily. Lower log loss is better.")
    with tab3:
        st.markdown("### Confusion matrix")
        cm = meta["metrics"][best_name]["confusion_matrix"]
        st.dataframe(pd.DataFrame(cm,index=["Actual Stay","Actual Churn"],columns=["Pred Stay","Pred Churn"]),use_container_width=True)
        st.write("TP = correctly identified churners, TN = correctly identified stayers, FP = false alarms, FN = missed churners.")
    with tab4:
        st.latex(r"\mathrm{Precision}=\frac{TP}{TP+FP}")
        st.latex(r"\mathrm{Recall}=\frac{TP}{TP+FN}")
        st.latex(r"F_1=2\cdot\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}")
        st.latex(r"\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}")
        st.write("ROC-AUC measures ranking quality across many thresholds; it does not choose the operating threshold for you.")
    with tab5:
        st.latex(r"\hat{y}=1 \;\text{if}\; p\ge\tau")
        st.write("Lower threshold → more customers flagged → usually higher recall and lower precision.")
        st.write("Higher threshold → fewer customers flagged → usually higher precision and lower recall.")

elif page == "◫ Exploratory Analysis":
    hero("Data Science","Exploratory <span class='glow'>Data Analysis</span>",
         "Inspect churn patterns before modeling. EDA helps you understand distributions, data quality and potential business signals.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing values", int(df.isna().sum().sum()))
    c4.metric("Churn", f"{df['churn'].mean():.1%}")

    st.subheader("Churn by contract")
    contract_rates = df.groupby("contract_type",as_index=False)["churn"].mean()
    st.plotly_chart(px.bar(contract_rates,x="contract_type",y="churn",text_auto=".1%"),use_container_width=True)

    st.subheader("Tenure vs monthly charges")
    sample = df.sample(min(2500,len(df)),random_state=4).copy()
    sample["Churn"] = sample["churn"].map({0:"Stay",1:"Churn"})
    st.plotly_chart(px.scatter(sample,x="tenure_months",y="monthly_charges",color="Churn",opacity=.55),use_container_width=True)

    st.subheader("Churn by payment method")
    pay_rates = df.groupby("payment_method",as_index=False)["churn"].mean().sort_values("churn")
    st.plotly_chart(px.bar(pay_rates,x="churn",y="payment_method",orientation="h",text_auto=".1%"),use_container_width=True)

elif page == "⌁ Feature Intelligence":
    hero("Explainability","Feature <span class='glow'>Intelligence</span>",
         "Use model-agnostic permutation importance to see which original inputs most affect predictive performance.")

    topn = st.slider("Number of features",5,min(17,len(importance)),12)
    chart = importance.head(topn).sort_values("importance")
    st.plotly_chart(px.bar(chart,x="importance",y="feature",orientation="h"),use_container_width=True)
    st.markdown("""
**How permutation importance works:** shuffle one feature at a time. If model performance drops a lot,
that feature was important to the model. This is useful across many model types, but it still does **not**
prove that the feature causes churn.
""")

elif page == "⚙ Threshold Studio":
    hero("Decision Science","Threshold <span class='glow'>Studio</span>",
         "A classifier produces probabilities. Business policy converts probabilities into actions. Tune the threshold and observe precision–recall trade-offs.")

    t = st.slider("Operating threshold",0.10,0.90,0.50,0.05)
    nearest = threshold_table.iloc[(threshold_table["threshold"]-t).abs().argsort()[:1]].iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",f"{nearest['accuracy']:.3f}")
    c2.metric("Precision",f"{nearest['precision']:.3f}")
    c3.metric("Recall",f"{nearest['recall']:.3f}")
    c4.metric("F1",f"{nearest['f1']:.3f}")

    fig = go.Figure()
    for col in ["precision","recall","f1","accuracy"]:
        fig.add_trace(go.Scatter(x=threshold_table["threshold"],y=threshold_table[col],mode="lines+markers",name=col.title()))
    fig.add_vline(x=t,line_dash="dash")
    fig.update_layout(xaxis_title="Threshold",yaxis_title="Score",height=430)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("""
**Business interpretation**
- If missing a churner is expensive, optimize for **recall** and consider a lower threshold.
- If retention offers are costly, optimize for **precision** and consider a higher threshold.
- Do not choose 0.50 automatically just because it is the default.
""")

elif page == "↗ ML Engineer Learning Path":
    hero("Career Learning","ML Engineer <span class='glow'>Learning Path</span>",
         "Use this single project to learn the end-to-end skills expected from practical AI/ML engineers.")

    modules = [
        ("01","Business framing","Define churn, prediction horizon, intervention, false-positive cost and success metrics."),
        ("02","Data understanding","Schema, target leakage, missing values, class distribution, data types and data contracts."),
        ("03","EDA","Univariate/bivariate analysis, segments, correlations, outliers and hypothesis generation."),
        ("04","Preprocessing","Train/test split, scaling, one-hot encoding, pipelines and avoiding leakage."),
        ("05","Modeling","Baseline models, linear vs nonlinear models, hyperparameters and class imbalance."),
        ("06","Evaluation","Accuracy, precision, recall, F1, ROC-AUC, log loss and confusion matrices."),
        ("07","Thresholding","Convert probabilities into actions using business costs and operating points."),
        ("08","Explainability","Feature importance, counterfactual testing, limitations and non-causal interpretation."),
        ("09","Deployment","Serialization, app/API inference, dependency management and environment reproducibility."),
        ("10","MLOps","Versioning, monitoring, drift, retraining triggers, observability and rollback."),
        ("11","Responsible AI","Privacy, fairness, governance, human review and documentation."),
        ("12","Communication","Explain model value to product, sales, operations and nontechnical stakeholders."),
    ]
    for num,title,body in modules:
        with st.expander(f"{num} — {title}"):
            st.write(body)

    st.subheader("Portfolio evidence you can show")
    st.markdown("""
- Data pipeline and reproducible preprocessing
- Four-model comparison
- Business-aware threshold tuning
- Interactive prediction UI
- Batch inference
- Explainability
- Model card and production architecture
- Responsible-AI considerations
- Interview-ready mathematical explanation
""")

elif page == "▶ Tutorials":
    hero("Guided Learning","Step-by-Step <span class='glow'>Tutorials</span>",
         "Follow the project from first principles to deployment. Each lesson is designed to be explainable in an interview.")

    lessons = [
        ("Tutorial 1 — What is customer churn?",
         "Churn is a customer leaving or cancelling a service. A churn model estimates risk early enough for a company to intervene."),
        ("Tutorial 2 — Define the target",
         "Choose what counts as churn and over what prediction horizon. Never use information that would only be known after the churn event."),
        ("Tutorial 3 — Split before learning",
         "Create train and test sets before fitting scalers, encoders or models. Otherwise information from the test set can leak into training."),
        ("Tutorial 4 — Encode categorical features",
         "Algorithms need numerical input. One-hot encoding creates binary columns for categories while preserving non-ordinal meaning."),
        ("Tutorial 5 — Scale numeric features",
         "StandardScaler transforms values using z=(x−μ)/σ. Scaling is especially important for distance- and coefficient-based methods."),
        ("Tutorial 6 — Train a baseline",
         "Start with logistic regression. A simple, transparent baseline tells you whether more complex models actually add value."),
        ("Tutorial 7 — Compare models fairly",
         "Use the same data split and evaluation metrics. Compare performance, interpretability, speed and operational complexity."),
        ("Tutorial 8 — Tune thresholds",
         "The model predicts probability. Your business selects a threshold based on the cost of false positives and false negatives."),
        ("Tutorial 9 — Explain predictions",
         "Use global importance for overall behaviour and profile-level reasoning for individual cases. Avoid claiming correlation is causation."),
        ("Tutorial 10 — Deploy",
         "Serialize the fitted pipeline, keep feature order/schema stable, load once, validate inputs and return probabilities."),
        ("Tutorial 11 — Monitor",
         "Track input drift, outcome drift, model performance, latency, errors and business KPIs after deployment."),
        ("Tutorial 12 — Retrain responsibly",
         "Retrain only with validated newer labels, compare against the current champion model and document changes before promotion."),
    ]
    for title,body in lessons:
        with st.expander(title):
            st.write(body)

elif page == "⌘ Interview Preparation":
    hero("Career Readiness","Interview <span class='glow'>Preparation</span>",
         "Practice explaining not just what you coded, but why each machine-learning decision was made.")

    qa = [
        ("Why did you use a pipeline?","To keep preprocessing and the model together, reduce leakage risk, guarantee consistent inference and simplify deployment."),
        ("Why is accuracy not enough?","With class imbalance, a model can look accurate while missing many churners. Precision, recall, F1 and ROC-AUC expose different failure modes."),
        ("What does ROC-AUC mean?","It measures how well the model ranks positives above negatives across thresholds. 0.5 is random ranking; 1.0 is perfect ranking."),
        ("Why might logistic regression still be valuable?","It is fast, interpretable, stable and provides calibrated-style probability outputs; it is an excellent baseline."),
        ("What is data leakage?","Using information during training that would not be available at prediction time, causing unrealistically strong evaluation results."),
        ("How would you productionize this?","Serve a versioned model behind an API, validate schemas, log predictions, monitor drift/performance, protect customer data and retrain through a controlled pipeline."),
        ("What would you monitor?","Feature distributions, missingness, prediction distribution, latency, errors, calibration, recall/precision when labels arrive, and business retention outcomes."),
        ("How would you improve the model?","Collect better labels/features, tune hyperparameters, test gradient boosting libraries, calibrate probabilities, engineer temporal features and validate across time."),
    ]
    for q,a in qa:
        with st.expander(q):
            st.write(a)

elif page == "⌂ Production Architecture":
    hero("System Design","Production <span class='glow'>Architecture</span>",
         "See how a portfolio Streamlit demo evolves into a real machine-learning system.")

    st.graphviz_chart("""
    digraph G {
      rankdir=LR;
      node [shape=box, style="rounded,filled", fillcolor="#111827", fontcolor="white", color="#374151"];
      CRM [label="CRM / Billing Data"];
      ETL [label="Data Validation + ETL"];
      FS [label="Feature Pipeline"];
      Train [label="Training Pipeline"];
      Reg [label="Model Registry"];
      API [label="Prediction API"];
      App [label="Retention App"];
      Mon [label="Monitoring"];
      CRM -> ETL -> FS -> Train -> Reg -> API -> App;
      API -> Mon;
      App -> Mon;
      Mon -> Train [label="retrain trigger"];
    }
    """)

    st.markdown("""
### Production checklist
- Version datasets, code and models.
- Validate input schema at inference.
- Keep preprocessing inside the deployed pipeline.
- Use a model registry/champion–challenger process.
- Log prediction version, timestamp and confidence.
- Monitor data drift, performance and latency.
- Add authentication and authorization.
- Encrypt customer data and minimize stored PII.
- Define rollback and retraining procedures.
- Test the retention intervention itself, not only the model.
""")

elif page == "⚖ Responsible AI":
    hero("Governance","Responsible <span class='glow'>AI</span>",
         "Predictive performance is only one part of a deployable ML system. Privacy, fairness, transparency and human review also matter.")

    st.markdown("""
### Key considerations
**Privacy:** use only necessary customer data, protect identifiers and follow applicable privacy requirements.

**Fairness:** test performance across relevant groups before using predictions to allocate offers or service treatment.

**Human review:** churn probabilities should support retention teams, not become an unquestioned automated decision.

**Explainability:** distinguish model associations from causal conclusions.

**Monitoring:** a model can become stale as customer behaviour, pricing and products change.

**Documentation:** maintain a model card describing training data, metrics, assumptions, limitations and intended use.
""")

elif page == "↓ Download Center":
    hero("Resources","Project <span class='glow'>Download Center</span>",
         "Use the included resources for learning, testing and portfolio development.")

    st.download_button("Download full dataset", (BASE/"data"/"customer_churn.csv").read_bytes(),
                       file_name="customer_churn.csv", mime="text/csv")
    st.download_button("Download sample batch file", (BASE/"data"/"sample_batch.csv").read_bytes(),
                       file_name="sample_batch.csv", mime="text/csv")
    st.download_button("Download model metrics", (BASE/"models"/"model_metrics.json").read_bytes(),
                       file_name="model_metrics.json", mime="application/json")
    st.download_button("Download feature importance", (BASE/"models"/"feature_importance.csv").read_bytes(),
                       file_name="feature_importance.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### Need the full source-code package or customization?")
    st.link_button(
        "WhatsApp 076 100 0618",
        "https://wa.me/94761000618?text=Hello%2C%20I%20am%20interested%20in%20the%20Customer%20Churn%20AI%20project%20source%20code%2C%20documentation%20and%20customization.",
        use_container_width=True
    )

st.markdown("---")
st.caption("Educational portfolio project. Synthetic data is used in this package. Real production use requires organization-specific validation, governance and monitoring.")
