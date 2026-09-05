import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bitcoin AI Trend Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

div[data-testid="stMetric"] {
    background-color: #161B22;
    border: 1px solid #30363D;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #AAB2BF;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(
        "model_artifacts/bitcoin_xgboost_model.pkl"
    )

    feature_columns = joblib.load(
        "model_artifacts/feature_columns.pkl"
    )

    metadata = joblib.load(
        "model_artifacts/model_metadata.pkl"
    )

    return model, feature_columns, metadata


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    data = pd.read_csv(
        "model_artifacts/processed_bitcoin_data.csv"
    )

    data["Date"] = pd.to_datetime(data["Date"])

    return data


# ============================================================
# LOAD RESOURCES
# ============================================================

model, feature_columns, metadata = load_model()
df = load_data()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("₿ Bitcoin AI")

    st.caption(
        "Machine Learning Market Trend Predictor"
    )

    st.divider()

    st.subheader("⚙️ Prediction Settings")

    available_dates = df["Date"].dt.date.tolist()

    selected_date = st.selectbox(
        "Select Historical Market Date",
        available_dates,
        index=len(available_dates) - 2
    )

    st.divider()

    st.subheader("🤖 Model Information")

    st.write("**Model:** XGBoost Classifier")
    st.write("**Task:** Binary Classification")
    st.write("**Prediction:** Next-Day Direction")
    st.write("**F1 Score:** 0.6132")

    st.divider()

    st.caption(
        "Week 06 | Machine Learning Capstone"
    )


# ============================================================
# GET SELECTED ROW
# ============================================================

selected_row = df[
    df["Date"].dt.date == selected_date
].iloc[0]


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

input_data = pd.DataFrame(
    [selected_row[feature_columns].values],
    columns=feature_columns
)


# ============================================================
# MODEL PREDICTION
# ============================================================

prediction = model.predict(input_data)[0]

prediction_probability = model.predict_proba(
    input_data
)[0][1]


# ============================================================
# DETERMINE DIRECTION
# ============================================================

if prediction == 1:

    direction = "UP 📈"
    confidence = prediction_probability
    prediction_message = "The model predicts an upward market direction."

else:

    direction = "DOWN 📉"
    confidence = 1 - prediction_probability
    prediction_message = "The model predicts a downward market direction."


# ============================================================
# HEADER
# ============================================================

st.title("₿ Bitcoin Market Trend Predictor")

st.subheader(
    "AI-Powered Next-Day Bitcoin Market Direction Analysis"
)

st.write(
    "An end-to-end Machine Learning system that analyzes historical "
    "Bitcoin market data and technical indicators to predict whether "
    "the market is likely to move UP or DOWN on the following day."
)

st.divider()


# ============================================================
# SELECTED DATE
# ============================================================

st.info(
    f"📅 Currently analyzing market data from: "
    f"**{selected_date.strftime('%B %d, %Y')}**"
)


# ============================================================
# AI PREDICTION
# ============================================================

st.header("🤖 AI Market Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted Direction",
        direction
    )

with col2:
    st.metric(
        "Model Confidence",
        f"{confidence:.2%}"
    )

with col3:
    st.metric(
        "Probability of UP",
        f"{prediction_probability:.2%}"
    )

st.success(prediction_message)


# ============================================================
# PREDICTION PROBABILITY
# ============================================================

st.subheader("📊 Prediction Probability")

st.progress(float(prediction_probability))

probability_col1, probability_col2 = st.columns(2)

with probability_col1:
    st.write(
        f"📈 **UP Probability:** "
        f"{prediction_probability:.2%}"
    )

with probability_col2:
    st.write(
        f"📉 **DOWN Probability:** "
        f"{1 - prediction_probability:.2%}"
    )


# ============================================================
# MARKET OVERVIEW
# ============================================================

st.divider()

st.header("💰 Market Overview")

market_col1, market_col2, market_col3, market_col4 = st.columns(4)

with market_col1:
    st.metric(
        "Open Price",
        f"${selected_row['Open']:,.2f}"
    )

with market_col2:
    st.metric(
        "High Price",
        f"${selected_row['High']:,.2f}"
    )

with market_col3:
    st.metric(
        "Low Price",
        f"${selected_row['Low']:,.2f}"
    )

with market_col4:
    st.metric(
        "Close Price",
        f"${selected_row['Close']:,.2f}"
    )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

st.divider()

st.header("📈 Technical Indicators")

indicator_col1, indicator_col2, indicator_col3, indicator_col4 = st.columns(4)

with indicator_col1:
    if "RSI" in df.columns:
        st.metric(
            "RSI",
            f"{selected_row['RSI']:.2f}"
        )

with indicator_col2:
    if "MACD" in df.columns:
        st.metric(
            "MACD",
            f"{selected_row['MACD']:.2f}"
        )

with indicator_col3:
    if "SMA_7" in df.columns:
        st.metric(
            "SMA 7",
            f"${selected_row['SMA_7']:,.2f}"
        )

with indicator_col4:
    if "SMA_30" in df.columns:
        st.metric(
            "SMA 30",
            f"${selected_row['SMA_30']:,.2f}"
        )


# ============================================================
# ADDITIONAL FEATURES
# ============================================================

st.divider()

st.header("🔍 Additional Market Signals")

excluded_features = [
    "Open",
    "High",
    "Low",
    "Close",
    "RSI",
    "MACD",
    "SMA_7",
    "SMA_30"
]

additional_features = [
    feature
    for feature in feature_columns
    if feature not in excluded_features
]


if additional_features:

    columns = st.columns(3)

    for index, feature in enumerate(additional_features):

        value = selected_row[feature]

        with columns[index % 3]:

            if isinstance(value, (int, float)):
                st.metric(
                    feature,
                    f"{value:.4f}"
                )
            else:
                st.metric(
                    feature,
                    str(value)
                )


# ============================================================
# MODEL FEATURES TABLE
# ============================================================

st.divider()

st.header("🧠 Model Input Features")

st.caption(
    "The following engineered features are used as input for the "
    "XGBoost machine learning model."
)

feature_display = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Value": [
            selected_row[feature]
            for feature in feature_columns
        ]
    }
)

st.dataframe(
    feature_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.header("🎯 About This Project")

st.write("""
This application represents a complete end-to-end Machine Learning workflow.
The project was developed as a Machine Learning Capstone Project.

The workflow includes:

- Data Collection
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Technical Indicator Creation
- Multiple Model Training
- Model Evaluation
- Hyperparameter Tuning
- Final Model Selection
- Model Deployment
""")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🏆 Final Model Performance")

performance_col1, performance_col2, performance_col3 = st.columns(3)

with performance_col1:
    st.metric(
        "Accuracy",
        "50.46%"
    )

with performance_col2:
    st.metric(
        "F1 Score",
        "0.6132"
    )

with performance_col3:
    st.metric(
        "ROC-AUC",
        "0.5233"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning("""
⚠️ **Educational Disclaimer**

This application was developed for educational and research purposes
as part of a Machine Learning Capstone Project.

Cryptocurrency markets are highly volatile and difficult to predict.
The predictions generated by this model should not be considered
financial, investment, or trading advice.
""")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python • Pandas • Scikit-learn • XGBoost • Streamlit "
    "| Week 06 Machine Learning Capstone"
)