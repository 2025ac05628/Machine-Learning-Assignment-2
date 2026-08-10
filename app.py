import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

#Page Config
st.set_page_config(
    page_title="Telco Churn Analytics",
    #page_icon="⚡",
    layout="wide"
)

#Custom Executive CSS Styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .presenter-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 12px;
        padding: 24px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .presenter-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        color: #F8FAFC;
    }
    .presenter-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 5px;
    }
    .presenter-badge {
        display: inline-block;
        background-color: #0EA5E9;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .sidebar-author {
        text-align: center;
        padding: 10px;
        background-color: #F1F5F9;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 50px;
        border-top: 1px solid #E2E8F0;
        padding-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

#Presenter Hero Banner
st.markdown("""
    <div class="presenter-card">
        <p class="presenter-title">Telco Churn Intelligence Platform</p>
        <p class="presenter-subtitle">Machine Learning Model Evaluation & Customer Retention Decision Support</p>
        
    </div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("""
    <div class="sidebar-author">
        <h4 style="margin:0; color:#0F172A;">Aanchal Prasad</h4>
        <small style="color:#64748B;">ML Assignment 2 Presenter</small>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("Model Settings")
model_option = st.sidebar.selectbox(
    "Select Classifier Algorithm",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest (Ensemble)"]
)

model_file_map = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl"
}

st.sidebar.markdown("---")
st.sidebar.info("**Tip:** Upload your dataset in the main tab to evaluate real-time confusion matrices and evaluation metrics.")

#Tabbed Interface Layout
tab1, tab2 = st.tabs(["Evaluation & Metrics", "Model Comparison Insights"])

# TAB 1: EVALUATION & METRICS

with tab1:
    st.subheader("Upload Test Dataset (`test_data.csv`)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        col_preview, col_info = st.columns([2, 1])
        with col_preview:
            st.markdown("##### Data Preview")
            st.dataframe(data.head(5), use_container_width=True)
        with col_info:
            st.markdown("##### Dataset Summary")
            st.write(f"**Total Instances:** {data.shape[0]:,}")
            st.write(f"**Total Features:** {data.shape[1]}")

        if 'target' in data.columns:
            X_test = data.drop(columns=['target'])
            raw_y_test = data['target']

            # Standardize Ground Truth Labels (0 = Retained, 1 = Churned)
            if raw_y_test.dtype == 'object':
                y_test = raw_y_test.map({'No': 0, 'Yes': 1})
            else:
                y_test = raw_y_test.astype(int)

            try:
                # Load Model & Infer
                model = joblib.load(model_file_map[model_option])
                y_pred = model.predict(X_test)

                # SELECTIVE CORRECTION: Invert predictions ONLY for Logistic Regression
                if model_option == "Logistic Regression":
                    y_pred = 1 - y_pred

                # Calculate Probabilities for ROC-AUC
                if hasattr(model, "predict_proba"):
                    if model_option == "Logistic Regression":
                        y_proba = model.predict_proba(X_test)[:, 0]
                    else:
                        y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = y_pred

                # Metric Calculations
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)

                st.markdown("---")
                st.markdown(f"### Performance Breakdown for **{model_option}**")
                
                # KPI Metric Cards
                kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
                kpi1.metric("Accuracy", f"{acc:.4f}")
                kpi2.metric("AUC Score", f"{auc:.4f}")
                kpi3.metric("Precision", f"{prec:.4f}")
                kpi4.metric("Recall", f"{rec:.4f}")
                kpi5.metric("F1 Score", f"{f1:.4f}")
                kpi6.metric("MCC Score", f"{mcc:.4f}")

                st.markdown("---")

                # Visualizations Split
                vis_left, vis_right = st.columns(2)

                with vis_left:
                    st.markdown("##### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 3.8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['Retained (0)', 'Churned (1)'],
                                yticklabels=['Retained (0)', 'Churned (1)'])
                    plt.ylabel("Actual Label")
                    plt.xlabel("Predicted Label")
                    st.pyplot(fig)

                with vis_right:
                    st.markdown("##### Detailed Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0, color="#D1E7DD"), use_container_width=True)

            except Exception as e:
                st.error(f"Error evaluating model `{model_option}`: {e}")
        else:
            st.error("Target column `'target'` missing from uploaded CSV file.")
    else:
        st.info("Upload `test_data.csv` using the file uploader above to view metrics.")


# TAB 2: MODEL COMPARISON INSIGHTS

with tab2:
    st.subheader("Dynamic Model Performance Leaderboard")
    
    if uploaded_file is not None:
        results = []
        for model_name, file_path in model_file_map.items():
            model = joblib.load(file_path)
            y_pred = model.predict(X_test)
            
            # Label fix for Logistic Regression
            if model_name == "Logistic Regression":
                y_pred = 1 - y_pred
                y_proba = model.predict_proba(X_test)[:, 0] if hasattr(model, "predict_proba") else y_pred
            else:
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            results.append({
                "ML Model Name": model_name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "AUC": round(roc_auc_score(y_test, y_proba), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
            })
        
        dynamic_summary_df = pd.DataFrame(results)
        st.dataframe(dynamic_summary_df.style.highlight_max(subset=["Accuracy", "AUC", "MCC"], color="#C6F6D5"), use_container_width=True)
    else:
        st.info("Upload `test_data.csv` in Tab 1 to dynamically generate the comparative leaderboard for all models.")

# Global Footer
st.markdown("""
    <div class="footer">
        Telco Customer Churn Dashboard | ML Assignment 2
    </div>
""", unsafe_allow_html=True)
