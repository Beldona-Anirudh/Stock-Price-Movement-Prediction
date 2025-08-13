
# Stock Price Movement Prediction

This project is all about trying to predict whether Apple’s stock price will go up or down the next day. First, it pulls real historical data for Apple (from 2020 to 2024) using the yfinance library. Then, it does some smart data preparation calculating things like daily returns and moving averages, which are useful signals for market trends.




## What This Project Does

Automatically downloads stock data (Apple, 2020–2024) using yfinance.

Calculates daily returns and moving averages classic trading indicators.

Creates clear visualizations of price trends, return patterns, and feature correlations.

Trains a machine learning model to predict next-day price movement (up or down).

Shows which indicators matter most using feature importance plots.
## Tools & Libraries Used

-> Python – Core programming language.

-> Jupyter Notebook – Interactive coding and analysis.

-> yfinance – Fetching historical stock market data.

-> Pandas – Data manipulation and cleaning.

-> Matplotlib / Seaborn – Creating graphs and visualizations.

-> Scikit-learn – Machine learning model building and evaluation.
## Binary Prediction

This project is a Binary Prediction, meaning the model predicts one of two possible outcomes for the next day’s stock price:

    1 (Up) → If the closing price is expected to be higher than today’s.

    0 (Down) → If the closing price is expected to be lower or the  same.
## Tech Stack

-> Programming & Environment

    Python 
    Jupyter Notebook


-> Libraries & Frameworks

    yfinance
    Pandas
    Matplotlib
    Seaborn
    Scikit-learn

-> Machine Learning Approach

    Binary Classification (Up = 1, Down = 0)
    Train-Test Split
    Accuracy, Confusion Matrix, Feature Importance
## How to Run It

Clone the repo

    git clone https://github.com/Beldona-Anirudh/Stock-Price-Movement-Prediction
    


Install the Streamlit

    pip install Streamlit.

Change the Directory

    Switch the cmd directory to the directory where your file is located.

Run 

    Now run the app file with the extension .py .
## Results at a Glance

Accuracy: 0.4834710743801653

Top Predictors: 
    
    Moving averages, daily returns

Visual Highlights:

    Price trend chart over 4 years

    Heatmap showing feature correlations

    Bar chart ranking most important features