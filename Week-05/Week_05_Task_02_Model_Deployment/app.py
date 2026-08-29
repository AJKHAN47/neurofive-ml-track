import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)


# Load trained model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model" / "titanic_model.joblib"
    return joblib.load(model_path)


model = load_model()


# App title
st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger information below to predict whether "
    "the passenger would have survived the Titanic disaster."
)


# Input fields
st.subheader("Passenger Information")

pclass = st.selectbox(
    "Passenger Class",
    options=[1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    options=["Female", "Male"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Number of Siblings / Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents / Children Aboard",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=32.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    options=[
        "Cherbourg (C)",
        "Queenstown (Q)",
        "Southampton (S)"
    ]
)


# Predict button
if st.button("Predict Survival"):

    # Convert categorical inputs to encoded values
    sex_male = 1 if sex == "Male" else 0

    embarked_q = 1 if embarked == "Queenstown (Q)" else 0
    embarked_s = 1 if embarked == "Southampton (S)" else 0

    # Create input dataframe in EXACT training feature order
    input_data = pd.DataFrame(
        [[
            pclass,
            age,
            sibsp,
            parch,
            fare,
            sex_male,
            embarked_q,
            embarked_s
        ]],
        columns=[
            'Pclass',
            'Age',
            'SibSp',
            'Parch',
            'Fare',
            'Sex_male',
            'Embarked_Q',
            'Embarked_S'
        ]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.divider()

    if prediction == 1:
        st.success("Prediction: The passenger is likely to SURVIVE! 🎉")
    else:
        st.error("Prediction: The passenger is unlikely to survive.")