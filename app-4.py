import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Title
st.title("📈 Stock Price Movement Prediction - AAPL (Apple)")

# --- Data Section ---
st.header("1️⃣ Data Description")

# Download stock data
@st.cache_data
def load_data():
    df = yf.download('AAPL', start='2020-01-01', end='2024-12-31')
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    return df

df = load_data()
st.dataframe(df.head())

# --- Visualization Section ---
st.header("2️⃣ Visualizations")

st.subheader("📊 Apple Stock Closing Price")
fig1, ax1 = plt.subplots()
ax1.plot(df['Close'], label="Close Price")
ax1.set_title("Apple Stock Price (Close)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
st.pyplot(fig1)

st.subheader("📉 Daily Return Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df['Return'], bins=50, kde=True, ax=ax2)
ax2.set_title("Distribution of Daily Returns")
st.pyplot(fig2)

# --- Model & Prediction Section ---
st.header("3️⃣ Model & Predictions")

features = ['Return', 'MA10', 'MA50', 'Volatility']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Confusion matrix
st.subheader("🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
st.pyplot(fig3)

# Classification report
st.subheader("📄 Classification Report")
report = classification_report(y_test, y_pred, output_dict=False)
st.text(report)
