import joblib
import pandas as pd

# Load the complete preprocessing and classification pipeline
model = joblib.load("models/aquiferguard_model.pkl")

feature_names = [
    "Potential of Hydrogen (pH)",
    "Electric Conductivity (μS/cm)",
    "Bicarbonate (mg/L)",
    "Chloride (mg/L)",
    "Sodium (mg/L)"
]

# Test 1: Complete input
complete_input = pd.DataFrame(
    [[8.0, 1000.0, 400.0, 100.0, 150.0]],
    columns=feature_names
)

# Test 2: Input with one missing measurement
input_with_missing_value = pd.DataFrame(
    [[float("nan"), 1000.0, 400.0, 100.0, 150.0]],
    columns=feature_names
)

for test_name, sample in [
    ("Complete input", complete_input),
    ("Input with one missing value", input_with_missing_value)
]:
    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    assert prediction in [0, 1], f"{test_name}: invalid prediction."
    assert len(probabilities) == 2, f"{test_name}: expected two class probabilities."
    assert abs(float(probabilities.sum()) - 1.0) < 1e-6, (
        f"{test_name}: probabilities do not sum to 1."
    )

    category = "Elevated" if prediction == 1 else "Not Elevated"

    print(f"PASS: {test_name}")
    print("Predicted category:", category)
    print(
        "Estimated elevated-class probability:",
        round(float(probabilities[1]) * 100, 2),
        "%"
    )
    print()

print("All AquiferGuard pipeline tests passed.")
