import streamlit as st
import pandas as pd
import joblib

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="AquiferGuard",
    page_icon="💧",
    layout="wide"
)

# 2. LOAD MODEL AND DATASET
model = joblib.load("models/aquiferguard_model.pkl")

df = pd.read_csv("dataset/raw/cgwb_punjab_groundwater_quality_raw.csv")
df = df.dropna(subset=["Uranium(mg/L)"])
df = df.drop_duplicates()
df["Uranium_Risk"] = (df["Uranium(mg/L)"] > 0.03).astype(int)

features = [
    "Potential of Hydrogen (pH)",
    "Electric Conductivity (μS/cm)",
    "Bicarbonate (mg/L)",
    "Chloride (mg/L)",
    "Sodium (mg/L)"
]

# 3. INITIALIZE PREDICTION HISTORY
if "history" not in st.session_state:
    st.session_state.history = []

# 4. SIDEBAR
st.sidebar.title("💧 AquiferGuard")
st.sidebar.write("Groundwater Uranium Risk Classification")
st.sidebar.markdown("---")
st.sidebar.info(
    "Academic machine learning project using groundwater "
    "quality data from the CGWB dataset."
)

# 5. MAIN PAGE
st.title("💧 AquiferGuard")
st.subheader("Groundwater Uranium Risk Prediction")
st.write(
    "Explore groundwater quality records and estimate the uranium "
    "risk category using a Decision Tree classification model."
)

# 6. DASHBOARD STATISTICS
st.markdown("### Groundwater Dataset Overview")

total = len(df)
elevated = int((df["Uranium_Risk"] == 1).sum())
not_elevated = int((df["Uranium_Risk"] == 0).sum())

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", f"{total:,}")
col2.metric("Elevated Samples", f"{elevated:,}")
col3.metric("Not Elevated Samples", f"{not_elevated:,}")

st.markdown("---")

# 7. DISTRICT-WISE CHART
st.subheader("District-wise Elevated Uranium Samples")

district_data = (
    df[df["Uranium_Risk"] == 1]
    .groupby("District")
    .size()
    .sort_values(ascending=False)
)

st.bar_chart(district_data)

# 8. DECISION TREE FEATURE IMPORTANCE
st.markdown("---")
st.subheader("Groundwater Parameter Feature Importance")
st.write(
    "This chart shows how much each input parameter contributed to "
    "the splits made by the trained Decision Tree."
)

# Support both a standalone Decision Tree and a Pipeline.
tree_model = model

if hasattr(model, "steps"):
    tree_model = model.steps[-1][1]

if hasattr(tree_model, "feature_importances_"):
    importance_values = tree_model.feature_importances_

    if len(importance_values) == len(features):
        importance_df = pd.DataFrame({
            "Groundwater Parameter": features,
            "Importance": importance_values
        }).sort_values("Importance", ascending=False)

        st.bar_chart(
            importance_df.set_index("Groundwater Parameter")["Importance"]
        )

        st.caption(
            "Feature importance represents each feature's relative contribution "
            "to the tree's impurity reduction. It does not show whether a feature "
            "increases or decreases risk, prove causation, or establish water safety."
        )
    else:
        st.warning(
            "The saved model's number of input features does not match "
            "the five groundwater parameters used by this application."
        )
else:
    st.warning(
        "The loaded model does not expose Decision Tree feature importance. "
        "Check that aquiferguard_model.pkl contains a trained Decision Tree."
    )

# 9. DATASET PREVIEW
st.markdown("---")
st.subheader("Groundwater Sample Records")

if st.checkbox("Show Dataset"):
    st.dataframe(
        df[["Station", "District", "Uranium(mg/L)", "Uranium_Risk"]],
        use_container_width=True
    )

# 10. GROUNDWATER INPUT FORM
st.markdown("---")
st.subheader("Enter Groundwater Parameters")
st.write("Enter the water chemistry values to estimate the risk category.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Water Chemistry**")
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=8.0)
    ec = st.number_input(
        "Electrical Conductivity (μS/cm)", min_value=0.0, value=1000.0
    )
    bicarbonate = st.number_input(
        "Bicarbonate (mg/L)", min_value=0.0, value=400.0
    )

with col2:
    st.markdown("**Additional Parameters**")
    chloride = st.number_input(
        "Chloride (mg/L)", min_value=0.0, value=100.0
    )
    sodium = st.number_input(
        "Sodium (mg/L)", min_value=0.0, value=150.0
    )

# 11. PREDICTION
st.markdown("---")

if st.button("Predict Uranium Risk", type="primary"):
    input_data = pd.DataFrame(
        [[ph, ec, bicarbonate, chloride, sodium]],
        columns=features
    )

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    result = "Elevated" if prediction == 1 else "Not Elevated"

    st.session_state.history.append({
        "pH": ph,
        "Electrical Conductivity": ec,
        "Bicarbonate": bicarbonate,
        "Chloride": chloride,
        "Sodium": sodium,
        "Prediction": result,
        "Estimated Elevated-Class Probability (%)": round(probability * 100, 2)
    })

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Elevated Uranium Risk")
    else:
        st.success("✅ Not Elevated Uranium Risk")

    st.metric(
        "Model-Estimated Elevated-Class Probability",
        f"{probability * 100:.2f}%"
    )
    st.caption(
        "Project classification threshold: uranium concentration greater "
        "than 0.03 mg/L. The displayed probability is the model's estimate, "
        "not a laboratory measurement."
    )

# 12. PREDICTION HISTORY AND DOWNLOAD
st.markdown("---")
st.subheader("Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df, use_container_width=True)

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction History (CSV)",
        data=csv_data,
        file_name="aquiferguard_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No predictions made yet. Make a prediction to create history.")

# 13. DISCLAIMER
st.markdown("---")
st.caption(
    "Disclaimer: AquiferGuard is an academic machine learning project. "
    "Its predictions are estimates and do not replace laboratory "
    "groundwater testing or official water-quality assessment."
)
