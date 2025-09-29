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

# Clone the repository
```
git clone https://github.com/GKK-Hub/rice_yield_prediction.git
cd rice_yield_prediction
```

# Create environment

## Windows
```
python -m venv venv
venv\Scripts\activate
```

## Linux/MacOS
```
python -m venv venv
source venv/bin/activate
```

# Install dependencies
```
pip install -e .
```

# Data processing

Open a new terminal and from the project root, run:

```
# Process individual raw files
rice-yield process
```
Once it's complete, run the following:

# Combine processed files into a single dataset
```
# Combine the cleaned files
rice-yield combine
```

You should see two folders created under data\ after the above code is run.

# Explore data and optimize models

In the same terminal, run:

```
# Exploratory data analysis
quarto preview notebooks/eda.qmd
```

Once the above command re-directs to a browser, read the analysis. 

Now, take a new terminal and run:

# Model training and hyperparameter optimization
```
# Model optimization and SHAP
quarto preview notebooks/optimize.qmd
```
Note: Running optimize.qmd will create the outputs/ folder dynamically.

# Monitor experiments
```
# Go to outputs folder
cd outputs
```
Take a new terminal and run:
```
# Start MLflow UI
mlflow ui
```

If it doesn't re-direct to a browser window, click on the local host url from the terminal output.

Take a new terminal, move to outputs/ folder and run:

```
# Start Optuna dashboard (for hyperparameter tuning)
optuna-dashboard "sqlite:///models_db.sqlite3"
```
If it doesn't re-direct to a browser window, do the same as above.


Finally, take a new terminal and run:

From project root, run

```
streamlit run streamlit_app.py
```

# Contact

For questions or suggestions, reach out to:

Name: Gowtham Kumar K

Email: gowthamkumar.kg@gmail.com