
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random, json, os
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -------------------------------
# Setup
# -------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

st.set_page_config(page_title="Hackathon ML Demo", layout="wide")
st.title("🚀 Hackathon ML Demo with Streamlit")

# -------------------------------
# Conversation UI
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def chat_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me about the datasets, models, or results...")
if prompt:
    chat_message("user", prompt)
    with st.chat_message("assistant"):
        st.markdown(f"Processing your request: **{prompt}**")

# -------------------------------
# Generate synthetic datasets
# -------------------------------
@st.cache_data
def generate_datasets():
    # Classification
    X_cls, y_cls = make_classification(
        n_samples=2000, n_features=20, n_informative=6,
        n_redundant=4, n_classes=2, weights=[0.7,0.3],
        class_sep=1.2, flip_y=0.02, random_state=SEED
    )
    cls_cols = [f"feat_{i}" for i in range(X_cls.shape[1])]
    df_cls = pd.DataFrame(X_cls, columns=cls_cols)
    df_cls["target"] = y_cls
    df_cls["region"] = np.random.choice(["APAC","EMEA","AMER"], size=len(df_cls))
    df_cls["channel"] = np.random.choice(["web","mobile","api"], size=len(df_cls))

    # Regression
    X_reg, y_reg = make_regression(
        n_samples=1500, n_features=10, n_informative=5,
        noise=10.0, random_state=SEED
    )
    reg_cols = [f"rfeat_{i}" for i in range(X_reg.shape[1])]
    df_reg = pd.DataFrame(X_reg, columns=reg_cols)
    df_reg["target"] = y_reg
    df_reg["segment"] = np.random.choice(["A","B","C"], size=len(df_reg))

    return df_cls, df_reg

df_cls, df_reg = generate_datasets()

st.sidebar.header("📊 Dataset Explorer")
dataset_choice = st.sidebar.radio("Choose dataset", ["Classification", "Regression"])

if dataset_choice == "Classification":
    st.subheader("Classification Dataset")
    st.write(df_cls.head())
    st.bar_chart(df_cls["target"].value_counts())
elif dataset_choice == "Regression":
    st.subheader("Regression Dataset")
    st.write(df_reg.head())
    st.line_chart(df_reg["target"].head(100))

# -------------------------------
# Train models
# -------------------------------
def train_classification(df):
    df = pd.get_dummies(df, columns=["region","channel"], drop_first=True)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=SEED,stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_clf = LogisticRegression(max_iter=1000,class_weight="balanced",random_state=SEED)
    log_clf.fit(X_train_scaled,y_train)
    y_pred_log = log_clf.predict(X_test_scaled)
    y_proba_log = log_clf.predict_proba(X_test_scaled)[:,1]

    rf_clf = RandomForestClassifier(n_estimators=200,random_state=SEED)
    rf_clf.fit(X_train,y_train)
    y_pred_rf = rf_clf.predict(X_test)
    y_proba_rf = rf_clf.predict_proba(X_test)[:,1]

    return {
        "logistic": (y_test,y_pred_log,y_proba_log),
        "rf": (y_test,y_pred_rf,y_proba_rf),
        "rf_model": rf_clf
    }

def train_regression(df):
    df = pd.get_dummies(df, columns=["segment"], drop_first=True)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled,y_train)
    y_pred_lin = lin_reg.predict(X_test_scaled)

    rf_reg = RandomForestRegressor(n_estimators=200,random_state=SEED)
    rf_reg.fit(X_train,y_train)
    y_pred_rf = rf_reg.predict(X_test)

    return {
        "linear": (y_test,y_pred_lin),
        "rf": (y_test,y_pred_rf),
        "rf_model": rf_reg
    }

if dataset_choice == "Classification":
    results = train_classification(df_cls)
    y_test,y_pred,y_proba = results["rf"]
    acc = accuracy_score(y_test,y_pred)
    st.metric("RandomForest Accuracy", f"{acc:.3f}")
elif dataset_choice == "Regression":
    results = train_regression(df_reg)
    y_test,y_pred = results["rf"]
    mse = mean_squared_error(y_test,y_pred)
    rmse = mse ** 0.5
    st.metric("RandomForest RMSE", f"{rmse:.2f}")

    # -------------------------------
# Historical Incident Analysis Visualization
# -------------------------------
st.header("📊 Historical Incident Analysis")

# Synthetic incident dataset
incident_data = {
    "IncidentID": [f"INC-{i}" for i in range(1, 11)],
    "ErrorType": ["DB", "API", "Config", "Network", "DB", "API", "Config", "Network", "DB", "API"],
    "Downtime": [30, 45, 20, 60, 25, 50, 15, 70, 40, 55],
    "Resolution": ["Rollback", "Config Fix", "Scaling", "Failover",
                   "Rollback", "Config Fix", "Scaling", "Failover",
                   "Rollback", "Config Fix"]
}
df_inc = pd.DataFrame(incident_data)

tab1, tab2, tab3 = st.tabs(["Scatter Similarity", "Resolution Frequency", "Flow Diagram"])

# Scatter Plot
with tab1:
    st.subheader("Incident Similarity (ErrorType vs Downtime)")
    fig1 = px.scatter(
        df_inc,
        x="ErrorType",
        y="Downtime",
        color="Resolution",
        hover_data=["IncidentID", "Resolution", "Downtime"],
        title="Scatter plot of incidents"
    )
    # Highlight current incident (example: INC-5)
    fig1.add_scatter(x=["DB"], y=[25], mode="markers",
                     marker=dict(color="black", size=12, symbol="star"),
                     name="Current Incident")
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("💡 Hover over points to see IncidentID, Resolution, and Downtime — justifying similarity.")

# Bar Chart
with tab2:
    st.subheader("Resolution Frequency")
    res_counts = df_inc["Resolution"].value_counts().reset_index()
    res_counts.columns = ["Resolution", "Count"]
    fig2 = px.bar(
        res_counts,
        x="Resolution",
        y="Count",
        text="Count",
        hover_data=["Resolution", "Count"],
        title="Frequency of resolution types"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("💡 Hover shows exact counts, justifying which resolution strategy is most common.")

# Flow Diagram (Sankey)
with tab3:
    st.subheader("Incident → Similar Cases → Resolution → Recommendation")
    fig3 = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Incident", "Similar Cases", "Rollback", "Config Fix", "Scaling", "Failover", "Recommendation"],
            color=["blue","orange","green","red","purple","brown","grey"]
        ),
        link=dict(
            source=[0,0,1,1,1,1],
            target=[1,2,3,4,5,6],
            value=[10,4,3,2,1,5],
            hovertemplate="From %{source.label} → %{target.label}<br>Count: %{value}<extra></extra>"
        )
    ))
    fig3.update_layout(title_text="Flow of incident analysis", font_size=12)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("💡 Hover over links to see how many incidents flowed into each resolution, justifying recommendations.")
    # -------------------------------
# Conversational UI
# -------------------------------
st.header("💬 Conversational Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me about the datasets, models, or incident analysis...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        # Simple router
        if "regression target" in prompt.lower():
            subset = df_reg["target"].head(100).reset_index()
            subset.columns = ["Index", "Target"]
            import plotly.express as px
            fig = px.line(subset, x="Index", y="Target", markers=True,
                          hover_data=["Target"],
                          title="First 100 Regression Target Values")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 Hover over points to see exact target values.")
        elif "accuracy" in prompt.lower():
            st.markdown(f"RandomForest Accuracy: **{accuracy_score(*results['rf'][:2]):.3f}**")
        elif "incident analysis" in prompt.lower():
            st.markdown("Switch to the Incident Analysis tabs above for scatter, bar, and flow diagrams.")
        else:
            st.markdown(f"Processing your request: **{prompt}**")