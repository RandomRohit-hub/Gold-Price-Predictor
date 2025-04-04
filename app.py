import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

st.set_page_config(page_title="Gold Price Predictor", page_icon="💰", layout="wide")

st.title("Gold Price Predictor 💸")
st.markdown("This app analyzes gold price data and predicts future prices using Machine Learning. ")

# Load data
data_file = 'gld_price_data.csv'
gold_data = pd.read_csv(data_file)

st.subheader("📄 Dataset Overview")
st.dataframe(gold_data.head())

with st.expander("More about the dataset 📊"):
    st.write("**Shape:**", gold_data.shape)
    st.write("**Missing Values:**")
    st.write(gold_data.isnull().sum())
    st.write("**Statistical Summary:**")
    st.write(gold_data.describe())

# Correlation heatmap
st.subheader("🔄 Feature Correlation")
correlation = gold_data.select_dtypes(include=['number']).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='Blues', fmt=".1f", ax=ax)
st.pyplot(fig)

# Target correlation
st.subheader("📊 Correlation with Target (GLD)")
st.bar_chart(correlation['GLD'])

# Check column names
available_features = gold_data.columns.tolist()

# Safe feature selection
selected_features = [col for col in ['SPX', 'SLV'] if col in available_features]
X = gold_data[selected_features]
Y = gold_data['GLD']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Predict and evaluate
predictions = model.predict(X_test)
r2 = metrics.r2_score(Y_test, predictions)
mae = metrics.mean_absolute_error(Y_test, predictions)

st.subheader("📊 Model Performance")
st.write(f"**R-squared Score:** {r2:.2f}")
st.write(f"**Mean Absolute Error:** {mae:.2f}")

# Interactive prediction
st.subheader("🧰 Predict Gold Price")
spx = st.slider("SPX (S&P 500 Index)", float(X['SPX'].min()), float(X['SPX'].max()), float(X['SPX'].mean())) if 'SPX' in X else 0
slt = st.slider("SLV (Silver Price)", float(X['SLV'].min()), float(X['SLV'].max()), float(X['SLV'].mean())) if 'SLV' in X else 0

user_input = np.array([[spx, slt]])
user_prediction = model.predict(user_input)

st.success(f"Predicted Gold Price: ${user_prediction[0]:.2f} 💰")

st.markdown("---")
