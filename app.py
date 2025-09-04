import pandas as pd
import joblib as jl
import streamlit as st

model = jl.load("model.pkl")
scaler = jl.load("scaler.pkl")
columns = jl.load("columns.pkl")

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

st.title("🚢 Titanic Survivor Predictor")
st.markdown("""
Predict whether a passenger survived the Titanic disaster based on their details.
Fill out the fields below and click **Predict**.
""")

st.sidebar.header("Passenger Details")

title = st.sidebar.selectbox("Title", ["Mr","Mrs","Master","Miss","Rare"])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
ticket_type = st.sidebar.selectbox("Ticket Type", ["A","C","F","L","P","S","U","W"])
pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3])
cabin = st.sidebar.selectbox("Cabin", ["A","B","C","D","E","F","G","T","U"])
embarked = st.sidebar.selectbox("Embarked", ["S","C","Q"])
fare = st.sidebar.number_input("Fare", 0, 1000, 50)

input_df = pd.DataFrame([{
    "Title": title,
    "Sex": 1 if sex=="Male" else 0,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "TicketType": ticket_type,
    "Pclass": pclass,
    "Cabin": cabin,
    "Embarked": embarked,
    "Fare": fare
}])

numerical_columns = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)

st.subheader("Passenger Details")

col1, col2, col3 = st.columns(3)
col1.metric("Title", title)
col1.metric("SibSp", sibsp)
col1.metric("Pclass", pclass)

col2.metric("Sex", sex)
col2.metric("Parch", parch)
col2.metric("Cabin", cabin)

col3.metric("Age", age)
col3.metric("TicketType", ticket_type)
col3.metric("Embarked", embarked)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.markdown("❌ **Prediction:** Not Survived", unsafe_allow_html=True)
        st.error("This passenger likely did not survive.")
    else:
        st.markdown("✅ **Prediction:** Survived", unsafe_allow_html=True)
        st.success("This passenger likely survived!")