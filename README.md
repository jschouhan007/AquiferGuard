# AquiferGuard

AquiferGuard is an academic machine learning project that explores groundwater uranium measurements and classifies samples into two project-defined categories: **Elevated** and **Not Elevated**.

The application uses a Decision Tree Classifier and provides a Streamlit dashboard for exploring the dataset and making sample predictions.

## Project features

- Explore groundwater sample records from the dataset.
- View total, elevated, and not-elevated sample counts.
- View district-wise counts of elevated samples.
- Enter groundwater chemistry parameters and receive a model prediction.
- Review prediction history during the current Streamlit session.
- Download prediction history as a CSV file.

## Technology

- Python
- Pandas and NumPy
- scikit-learn
- Streamlit
- Joblib
- Matplotlib and Seaborn
- Jupyter Notebook

## Project structure

```text
AquiferGuard/
│
├── app/
│   └── app.py
│
├── dataset/
│   ├── raw/
│   │   └── cgwb_punjab_groundwater_quality_raw.csv
│   └── processed/
│       └── aquiferguard_cleaned.csv
│
├── models/
│   └── aquiferguard_model.pkl
│
├── notebooks/
│   └── 01_simple_aquiferguard.ipynb
│
├── reports/
├── requirements.txt
└── .gitignore
```

## Setup

### 1. Open the project directory

Open a terminal in the main `AquiferGuard` directory.

### 2. Create a virtual environment (recommended)

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

## Run the application

From the main `AquiferGuard` directory, run:

```powershell
streamlit run app/app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## Retrain the model

1. Open `notebooks/01_simple_aquiferguard.ipynb`.
2. Run the notebook cells in order.
3. The notebook prepares the data, trains and evaluates the Decision Tree model, and saves the model to:

```text
models/aquiferguard_model.pkl
```

Restart the Streamlit application after retraining so it loads the updated model.

## Dataset

The project uses groundwater quality data published through India's National Water Data Portal, associated with the Central Ground Water Board (CGWB).

Dataset page:
https://nwdp.nwic.gov.in/dataset/ground-water-quality-manual-chemical-parameters-cgwb-f-gfg

The raw dataset should remain in `dataset/raw/`. The notebook writes the cleaned dataset to `dataset/processed/`.

## Project classification threshold

For this academic project, a uranium concentration **greater than 0.03 mg/L** is labelled as `Elevated`; values at or below 0.03 mg/L are labelled `Not Elevated`.

This is the project's chosen classification threshold. It should not be interpreted as a statement that a water sample is safe or unsafe under a particular regulation.

## Important limitation

AquiferGuard produces machine learning estimates from the selected input parameters. Its output is not a laboratory measurement, medical assessment, or official water-quality certification. Actual groundwater quality should be established through appropriate laboratory testing and applicable official standards.

## Notes

- Run commands from the main project directory so the relative paths in `app/app.py` work.
- Keep the raw dataset unchanged.
- Do not commit `.venv/`, secrets, or temporary files to Git.
