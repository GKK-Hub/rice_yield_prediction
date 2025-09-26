import streamlit as st
import pandas as pd
from rice_yield.utils.notebook_utils import load_model
from rice_yield.utils.paths import get_output_dir, get_data_dir

st.markdown(
    """
    <style>

    /* App background gradient */
    .stApp {
        background: linear-gradient(to bottom, #E0F7FA, #FFFFFF);
        background-attachment: fixed;
        color: #000000;  /* Default text color black */
    }

     /* Override Streamlit's default theme colors */
    .stApp {
        color-scheme: light;
    }

    </style>
    """, unsafe_allow_html=True
)


# Load saved test data and model
X_test = pd.read_csv(get_data_dir("final") / "X_test.csv")
y_test = pd.read_csv(get_data_dir("final") / "y_test.csv")
best_model = load_model(get_output_dir() / "best_model/rf.joblib")
st.title("Rice Yield Prediction 🌾")

test_df = X_test.copy()
test_df['yield'] = y_test.values

# --- Step 2: Auto-fill other features ---
# 1️⃣ Select district and year
district = st.selectbox("Select District", test_df["dist_name"].unique())
# year = st.slider("Select Year", min_value=int(test_df["year"].min()),
#                  max_value=int(test_df["year"].max()),
#                  value=int(test_df["year"].min()), step=1)
years_available = sorted(test_df["year"].unique())
year = st.selectbox("Select Year", years_available)
# 2️⃣ Prepare input DataFrame with all required columns
row = test_df[(test_df["dist_name"] == district) & (test_df["year"] == year)]
y_true = row['year'].iloc[0]

if not row.empty:
    input_df = row.drop(columns=['yield']).copy()
else:
    # If year not in dataset, fill numeric columns with mean for this district
    mean_values = X_test[X_test["dist_name"] == district]\
                  .mean(numeric_only=True)
    input_df = pd.DataFrame([mean_values])
    input_df["dist_name"] = district
    input_df["year"] = year

# 3️⃣ Optional: display numeric features as disabled inputs
for col in input_df.columns:
    if col not in ["dist_name", "year"]:
        st.number_input(f"{col}",
                        value=float(input_df[col].values[0]),
                        disabled=True)

# 4️⃣ Predict
if st.button("Predict"):
    prediction = best_model.predict(input_df)[0]

    # 5️⃣ Show prediction

    st.success(f"""🔮 Predicted Yield: {prediction:.2f} kg/ha |
                   True Yield (with default values): {y_true:.2f} kg/ha""")
