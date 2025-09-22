# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from rice_yield.utils.paths import get_output_dir, get_data_dir

# st.markdown(
#     """
#     <style>
#     /* App background gradient */
#     .stApp {
#         background: linear-gradient(to bottom, #E0F7FA, #FFFFFF);
#         background-attachment: fixed;
#         color: #000000;  /* Default text color black */
#     }

#     /* Sidebar labels and headers */
#     .stSidebar p, 
#     .stSidebar h2, 
#     .stSidebar h3 {
#         color: #000000;
#     }

#     /* Dropdown / Selectbox */
#     div[data-baseweb="select"] > div {
#         background-color: #1f77b4 !important;  /* strong blue */
#         color: white !important;
#     }

#     /* Slider track, handle and labels */
#     .stSlider > div > div > div {
#         background-color: #1f77b4 !important; /* strong blue */
#     }
#     .stSlider label, .stSlider span {
#         color: #000000 !important;  /* slider labels in black */
#     }

#     /* Predict button */
#     div.stButton > button {
#         background-color: #E0F7F; /* light blue */
#         color: white;
#         font-weight: bold;
#     }

#     div.stButton > button:hover {
#         background-color: #1f77b8; /* slightly darker blue on hover */
#         color: white;
#     }

#     /* Chart text labels */
#     .stPlotlyChart text, .stPlotlyChart tspan, .stMarkdown text {
#         fill: #000000 !important; /* ensures chart text is visible in black */
#     }

#     /* Button styling */
#     div.stButton > button:first-child {
#         background-color: #0057b7;  /* strong blue */
#         color: white;               /* font color */
#         font-weight: bold;
#         height: 45px;
#         width: 150px;
#         border-radius: 5px;
#     }

#     /* Success/output message styling */
#     .custom-output {
#         background-color: #cce5ff;  /* pale blue background */
#         color: #003366;             /* strong blue font */
#         padding: 10px;
#         border-radius: 5px;
#         font-weight: bold;
#         margin-top: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

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
    
    /* Force primary color to be your blue */
    :root {
        --primary-color: #0d47a1;
    }

    /* Sidebar labels and headers */
    .stSidebar p, 
    .stSidebar h2, 
    .stSidebar h3 {
        color: #000000;
    }

    /* Number input styling (for year field) */
    .stNumberInput input {
        background-color: #1565c0 !important;  /* dark blue background */
        color: white !important;               /* white text */
        border: 1px solid #1565c0 !important;
        border-radius: 5px !important;
    }

    /* Number input placeholder text */
    .stNumberInput input::placeholder {
        color: #b3d9ff !important;  /* light blue placeholder */
    }

    /* Selectbox styling (for dist_name dropdown) */
    div[data-baseweb="select"] > div {
        background-color: #1565c0 !important;  /* dark blue background */
        color: white !important;               /* white text */
        border: 1px solid #1565c0 !important;
    }

    /* Selectbox selected value text */
    div[data-baseweb="select"] > div > div {
        color: white !important;
    }

    /* Selectbox dropdown arrow */
    div[data-baseweb="select"] svg {
        fill: white !important;
    }

    /* Selectbox dropdown options menu */
    div[data-baseweb="select"] div[role="listbox"] {
        background-color: white !important;
        border: 1px solid #1565c0 !important;
    }

    /* Individual dropdown options */
    div[data-baseweb="select"] div[role="option"] {
        background-color: white !important;
        color: #1565c0 !important;
    }

    /* Dropdown option hover state */
    div[data-baseweb="select"] div[role="option"]:hover {
        background-color: #e3f2fd !important;
        color: #1565c0 !important;
    }

    /* Slider track, handle and labels */
    .stSlider > div > div > div {
        background-color: #1f77b4 !important; /* strong blue */
    }
    .stSlider label, .stSlider span {
        color: #000000 !important;  /* slider labels in black */
    }

    /* Button styling */
    div.stButton > button:first-child {
        background-color: #0057b7;  /* strong blue */
        color: white;               /* font color */
        font-weight: bold;
        height: 45px;
        width: 150px;
        border-radius: 5px;
    }

    div.stButton > button:hover {
        background-color: #1f77b8; /* slightly darker blue on hover */
        color: white;
    }

    /* SUCCESS MESSAGE STYLING - This fixes your issue! */
    div[data-testid="stAlert"] {
        background-color: #e3f2fd !important;  /* pale blue background */
        color: #1565c0 !important;             /* dark blue font */
        border: 1px solid #90caf9 !important; /* light blue border */
        border-radius: 8px !important;
    }

    /* Success message content */
    .stAlert > div {
        background-color: #e3f2fd !important;
        color: #1565c0 !important;
    }

    /* Make sure the success text is dark blue and bold */
    div[data-testid="stAlert"] div {
        color: #1565c0 !important;
        font-weight: 600 !important;
    }

    /* SLIDER CUSTOMIZATION */

    /* Force override Streamlit's default orange slider */
    .stSlider * {
        background-color: #e3f2fd !important;
    }

    /* More specific slider track targeting */
    .stSlider > div > div > div,
    .stSlider > div > div > div > div,
    .stSlider > div > div > div > div > div,
    div[data-testid="stSlider"] div,
    div[data-testid="stSlider"] div div,
    div[data-testid="stSlider"] div div div {
        background-color: #e3f2fd !important;  /* Light blue track */
    }

    /* Target the filled portion specifically */
    .stSlider [role="slider"] ~ div,
    .stSlider div[style*="background"],
    div[data-testid="stSlider"] div[style*="background-color"] {
        background-color: #90caf9 !important;  /* Medium blue filled area */
    }

    /* Slider handle/thumb - multiple targeting approaches */
    .stSlider [role="slider"],
    .stSlider div[role="slider"],
    div[data-testid="stSlider"] [role="slider"],
    .stSlider > div > div > div > div > div[style*="position: absolute"] {
        background-color: #0d47a1 !important;  /* Dark blue handle */
        border: 2px solid #ffffff !important;
        border-radius: 50% !important;
        width: 20px !important;
        height: 20px !important;
    }

    /* Nuclear option - override ALL slider colors */
    .stSlider div,
    .stSlider span,
    .stSlider [class*="slider"],
    div[data-testid="stSlider"] * {
        background: #e3f2fd !important;
    }

    /* But keep the handle dark blue */
    .stSlider [role="slider"],
    div[data-testid="stSlider"] [role="slider"] {
        background: #0d47a1 !important;
        border: 2px solid white !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------
# Load best model & data
# ----------------------
model_path = get_output_dir() / "best_model" / "svr.joblib"
data_path = get_data_dir("final") / "X_train.csv"

best_model = joblib.load(model_path)
X_train = pd.read_csv(data_path)
y_train = pd.read_csv(get_data_dir("final") / "y_train.csv")
df_train = X_train.copy()
df_train["yield"] = y_train.values

st.title("Rice Yield Prediction App 🌾")

st.markdown("Enter values for each feature to predict rice yield.")

# ----------------------
# Generate inputs
# ----------------------
user_input = {}

for col in X_train.columns:
    if col.lower() == "year":
        # Force integer input with max = 2016
        min_val, max_val = int(X_train[col].min()), \
                               max(2016, int(X_train[col].max()))
        mean_val = int(X_train[col].mean())
        user_input[col] = st.number_input(f"{col}",
                                          min_value=min_val,
                                          max_value=max_val,
                                          value=mean_val,
                                          step=1)
    elif X_train[col].dtype == "object":
        # Dropdown for categorical
        user_input[col] = st.selectbox(f"{col}", sorted(X_train[col].unique()))
    else:
        # Slider for numeric
        min_val, max_val = float(X_train[col].min()), float(X_train[col].max())
        mean_val = float(X_train[col].mean())
        user_input[col] = st.slider(f"{col}", min_val, max_val, mean_val)

# ----------------------
# Prediction button
# ----------------------
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = best_model.predict(input_df)
    st.success(f"Predicted Rice Yield: **{prediction[0]:.2f}** kg/ha")
    # st.markdown(f'<div class="custom-success">🔮 Predicted Yield: {prediction:.2f} kg/ha</div>', unsafe_allow_html=True)
    # st.markdown(
    #     f'<div class="custom-output">🔮 Predicted Yield): {prediction:.2f} kg/ha</div>',
    #     unsafe_allow_html=True
    # )

    # --- Trend Chart ---
    district = user_input.get("dist_name")
    year = user_input.get("year")
    st.subheader(f"📈 Yield Trend for {district}")

    # Merge X_train with y_train to get true yields
    df_train_full = X_train.copy()
    df_train_full["yield"] = y_train.values

    # Filter and compute trend
    df_trend = (
        df_train_full[df_train_full["dist_name"] == district]
        .groupby("year")["yield"]
        .mean()
        .reset_index()
    )

    # Show true yields over years
    # st.line_chart(df_trend.set_index("year")["yield"], height=400)
    # Wrap your chart in a container with custom styling
#     st.line_chart(
#     df_trend.set_index("year")["yield"], 
#     height=400,
#     color="#788aa5"  # Dark blue line
# ) 
    import plotly.express as px
    fig = px.line(
    df_trend,
    x="year",
    y="yield",
    markers=True,
    title="Historical Yield Trend"
)
    fig.update_layout(
        plot_bgcolor="#E0F7FA",   # chart background
        paper_bgcolor="#E0F7FA",  # entire chart background
        font_color="black",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray")
    )

    st.plotly_chart(fig, use_container_width=True)


    # If the selected year is in training data, highlight its true yield
    if year in df_trend["year"].values:
        selected_yield = df_trend[df_trend["year"] == year]["yield"].values[0]
        st.markdown(f"✅ **Year {year} (Observed):** {selected_yield:.2f} kg/ha")
    else:
        # If not in training, show the predicted value for reference
        st.markdown(f"No observed data for year {year}.")