
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.title("🔮 Customer Churn Prediction System")

uploaded = st.file_uploader("Upload churn_data.csv", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("Using sample churn_data.csv")
    df = pd.read_csv("churn_data.csv")

# Encode categorical
df_enc = df.copy()
for col in df_enc.select_dtypes(include=['object']).columns:
    if col != "customerID":
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

X = df_enc.drop(["customerID","Churn"], axis=1)
y = df_enc["Churn"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.subheader("Preview")
st.dataframe(df.head())

if st.button("Predict on Sample"):
    preds = model.predict(X)
    df['Predicted_Churn'] = preds
    st.write(df[['customerID','Churn','Predicted_Churn']].head(20))
