# ============================================================
# 🧬 COVID-19 Clinical Trials Predictor & Insight Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# ⚙️ Page Configuration
# ------------------------------------------------------------
st.set_page_config(page_title="COVID-19 Trials Predictor", layout="wide")
st.title("🧬 COVID-19 Clinical Trials – Prediction & Insight Dashboard")

# ------------------------------------------------------------
# 🧠 Load Pre-trained Models (Directly)
# ------------------------------------------------------------
try:
    rf_model = joblib.load("rf_covid_trials.pkl")
    lr_model = joblib.load("lr_covid_trials.pkl")
    scaler = joblib.load("scaler_covid_trials.pkl")
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {e}")
    st.stop()

# ------------------------------------------------------------
# 📥 Sidebar Controls
# ------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])
model_choice = st.sidebar.selectbox("🤖 Choose Model", ["RandomForest", "LinearRegression"])
show_eda = st.sidebar.checkbox("📊 Show EDA Visualizations")
manual_input = st.sidebar.checkbox("✍️ Manual Prediction Input")

# ============================================================
# 🧩 MAIN APP LOGIC
# ============================================================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")

    # 🧾 Data Preview
    st.write("### 🔍 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------------
    # 🧹 Data Cleaning
    # --------------------------------------------------------
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Extract Country if available
    if "Locations" in df.columns:
        df["Country"] = df["Locations"].fillna("Missing").apply(lambda x: str(x).split(",")[-1].strip())
    else:
        df["Country"] = "Unknown"

    # Fill missing Enrollment values
    if "Enrollment" in df.columns:
        df["Enrollment"].fillna(df["Enrollment"].median(), inplace=True)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype(str)
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])

    # --------------------------------------------------------
    # 🧠 Feature Setup
    # --------------------------------------------------------
    target = "Enrollment" if "Enrollment" in df.columns else None
    X = df.select_dtypes(include=np.number).copy()

    if target and target in X.columns:
        y = X[target]
        X = X.drop(columns=[target])
    else:
        y = None

    # Scaling
    try:
        X_cols_train = getattr(scaler, "feature_names_in_", None)
        if X_cols_train is not None:
            for col in X_cols_train:
                if col not in X.columns:
                    X[col] = 0
            X = X[X_cols_train]
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.warning(f"⚠️ Scaling skipped due to mismatch: {e}")
        X_scaled = X.copy()

    # --------------------------------------------------------
    # 🤖 Prediction
    # --------------------------------------------------------
    model = rf_model if model_choice == "RandomForest" else lr_model
    predictions = model.predict(X if model_choice == "RandomForest" else X_scaled)
    df["Predicted_Enrollment"] = predictions

    # --------------------------------------------------------
    # 🎯 Prediction Summary
    # --------------------------------------------------------
    st.subheader("🎯 Predicted Results Summary")
    st.write("These are the predicted participant enrollments for each clinical trial.")
    st.dataframe(df[["Predicted_Enrollment"]].head(10))

    # --------------------------------------------------------
    # 📈 Model Evaluation
    # --------------------------------------------------------
    if target:
        mae = np.mean(np.abs(df["Enrollment"] - df["Predicted_Enrollment"]))
        rmse = np.sqrt(np.mean((df["Enrollment"] - df["Predicted_Enrollment"]) ** 2))
        r2 = 1 - np.sum((df["Enrollment"] - df["Predicted_Enrollment"]) ** 2) / np.sum(
            (df["Enrollment"] - np.mean(df["Enrollment"])) ** 2
        )

        st.markdown("### 📊 Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("📏 MAE", f"{mae:.2f}")
        col2.metric("📉 RMSE", f"{rmse:.2f}")
        col3.metric("📈 R²", f"{r2:.3f}")
        st.success("✅ Model evaluation completed successfully!")

    # --------------------------------------------------------
    # 🧠 Insights
    # --------------------------------------------------------
    st.markdown("### 🧠 Insights & Results Interpretation")
    avg_pred = np.mean(df["Predicted_Enrollment"])
    high_pred = df["Predicted_Enrollment"].max()
    low_pred = df["Predicted_Enrollment"].min()

    st.info(f"""
    📊 **Summary of Predictions:**
    - Average Predicted Enrollment: **{avg_pred:.2f}**
    - Highest Predicted Enrollment: **{high_pred:.2f}**
    - Lowest Predicted Enrollment: **{low_pred:.2f}**

    💡 **Interpretation:**
    - Model predicts how many participants will enroll in a trial.
    - Higher predicted enrollments → large-scale studies.
    - Lower predicted enrollments → small or early-phase trials.
    """)

    # --------------------------------------------------------
    # 📈 Visualizations
    # --------------------------------------------------------
    st.subheader("📊 Prediction Visualizations")

    # Histogram
    fig1 = px.histogram(df, x="Predicted_Enrollment", nbins=40,
                        title="Distribution of Predicted Enrollment")
    st.plotly_chart(fig1, use_container_width=True)

    # Scatter: Actual vs Predicted
    if target:
        fig2 = px.scatter(df, x="Enrollment", y="Predicted_Enrollment",
                          title="Actual vs Predicted Enrollment (Trendline)", trendline="ols")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("🔹 Each dot represents a clinical trial. Dots closer to the diagonal mean accurate predictions.")

    # Feature Importance
    if model_choice == "RandomForest":
        st.subheader("🌍 Top 10 Important Features")
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="coolwarm")
        plt.title("Top 10 Most Influential Features on Enrollment")
        st.pyplot(fig)
        st.caption("📘 Longer bars = more influence on enrollment prediction.")

    # --------------------------------------------------------
    # 💾 Download Predictions
    # --------------------------------------------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predicted Results CSV",
        data=csv,
        file_name="COVID_Trial_Predictions.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # 📊 Optional EDA
    # --------------------------------------------------------
    if show_eda:
        st.subheader("📊 Exploratory Data Analysis (EDA)")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) > 1:
            st.write("#### 🔹 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=False)
            st.pyplot(fig)
            st.caption("🧩 Bright red/blue = strong correlation between variables.")

        if "Country" in df.columns:
            top_countries = df["Country"].value_counts().head(10)
            fig = px.bar(top_countries, x=top_countries.index, y=top_countries.values,
                         title="Top 10 Countries by Trial Count",
                         labels={"x": "Country", "y": "Count"})
            st.plotly_chart(fig, use_container_width=True)

        if "Enrollment" in df.columns:
            fig = px.histogram(df, x="Enrollment", nbins=40, title="Actual Enrollment Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.success("✅ EDA Completed Successfully!")

    # --------------------------------------------------------
    # ✍️ Manual Input
    # --------------------------------------------------------
    if manual_input:
        st.subheader("🧮 Manual Prediction Input")
        try:
            user_inputs = {}
            for feature in X.columns:
                user_inputs[feature] = st.number_input(f"{feature}", value=float(X[feature].mean()))
            user_df = pd.DataFrame([user_inputs])

            try:
                user_scaled = scaler.transform(user_df)
            except:
                user_scaled = user_df

            pred_val = model.predict(user_df if model_choice == "RandomForest" else user_scaled)[0]
            st.success(f"✅ Predicted Enrollment for Input Data: **{pred_val:.2f}**")

            st.info("""
            🔍 **Interpretation:**
            - The predicted number represents expected trial enrollment.
            - Adjust values to see impact on predictions.
            """)
        except Exception as e:
            st.error(f"Manual input failed: {e}")

else:
    st.info("⬆️ Please upload a CSV dataset to begin analysis.")

# ------------------------------------------------------------
# 🧭 Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Developed with ❤️ | COVID Clinical Trials ML Dashboard | v2.0")
