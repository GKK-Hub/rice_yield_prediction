# 🌾 Rice Yield Prediction

A machine learning project to predict rice yield using weather and environmental factors such as rainfall, precipitation, evapotranspiration, and water deficit.  

The project uses data preprocessing, model training, hyperparameter optimization, and SHAP-based interpretation to generate insights that align with agronomic knowledge.

# Project Structure
```
rice_yield_prediction
├─ data
│  └─ raw
│     ├─ actual_evapotranspiration.csv
│     ├─ area_production_yield.csv
│     ├─ irrigated_area.csv
│     ├─ maximum_temperature.csv
│     ├─ minimum_temperature.csv
│     ├─ potential_evapotranspiration.csv
│     ├─ precipitation.csv
│     ├─ rainfall.csv
│     └─ water_deficit.csv
├─ notebooks
│  ├─ eda.qmd
│  └─ optimize.qmd
├─ pyproject.toml
├─ README.md
├─ src
│  └─ rice_yield
│     ├─ clean
│     │  ├─ combine.py
│     │  └─ process.py
│     ├─ cli.py
│     ├─ models
│     │  ├─ mlflow_utils.py
│     │  ├─ optuna_utils.py
│     │  └─ shap_utils.py
│     ├─ utils
│     │  ├─ dataframe_utils.py
│     │  ├─ notebook_utils.py
│     │  ├─ paths.py
│     │  ├─ plot_utils.py
│     │  └─ py.typed
│     └─ __init__.py
├─ streamlit_app.py
└─ tests
   └─ test_paths.py

```

# Clone
```
git clone https://github.com/GKK-Hub/rice_yield_prediction.git
cd rice_yield_prediction
```

# Create environment
```
python -m venv venv
venv\Scripts\activate
```

# Install dependencies
```
pip install -e .
```

# Data processing
```
# Process individual raw files
rice-yield process

# Combine processed files into a single dataset
rice-yield combine
```

# Explore data and optimize models
```
# Exploratory data analysis
quarto preview notebooks/eda.qmd

# Model training and hyperparameter optimization
quarto preview notebooks/optimize.qmd
```

# Monitor experiments
```
# Go to outputs folder
cd outputs

# Start MLflow UI
mlflow ui

# Start Optuna dashboard (for hyperparameter tuning)
optuna-dashboard "sqlite:///models_db.sqlite3"
```

# Run the streamlit app
```
# Go to outputs folder
cd outputs

# Start MLflow UI
mlflow ui

# Start Optuna dashboard (for hyperparameter tuning)
optuna-dashboard "sqlite:///models_db.sqlite3"
```

# Contact

For questions or suggestions, reach out to:

Name: Gowtham Kumar K

Email: gowthamkumar.kg@gmail.com