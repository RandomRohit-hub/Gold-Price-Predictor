# 🪙 Gold Price Predictor 💸

https://gold-price-predictor-yeatwztuwqnq7eqrbfgnyt.streamlit.app/



This is a web-based app built with **Streamlit** that predicts gold prices based on financial indicators like the S&P 500 index and Silver price. The app uses a **Random Forest Regressor** model for accurate price forecasting.

## 🚀 Features

- 📊 Visualizes correlations between gold and other financial indicators
- 🧠 Predicts gold prices using machine learning
- 🧰 Interactive sliders for real-time prediction input
- 📈 Displays model evaluation metrics like R² Score and MAE
- ✅ Clean UI with emojis and expanders for better user experience

---

## 📂 Dataset

The dataset (`gld_price_data.csv`) includes historical prices for:

- **GLD** (Gold ETF)
- **SPX** (S&P 500 Index)
- **SLV** (Silver Price)
- Possibly other features like Oil, USD, etc. (ignored if unavailable)

---

## 🛠 What I Did

### ✅ Data Preprocessing

- Loaded and cleaned the dataset using **pandas**
- Handled missing values and dropped non-numeric columns for correlation
- Visualized correlation matrix using **Seaborn** heatmap

### ✅ Feature Selection

- Selected features based on high correlation with the target `GLD` value
- Dynamically handled missing features (`Oil`, `USD`, etc.)

### ✅ Model Building

- Split data into training and testing sets
- Trained a **Random Forest Regressor** from `sklearn.ensemble`
- Evaluated the model using:
  - **R² Score**
  - **Mean Absolute Error (MAE)**

### ✅ Streamlit App

- Added clean UI with Streamlit components like:
  - Sliders for user input
  - Charts and heatmaps
  - Expanders for data insight
- Wrapped everything in an interactive app for real-time gold price prediction

---

## ▶️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/gold-price-predictor.git
   cd gold-price-predictor
