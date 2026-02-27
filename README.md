# trader-behavior-insights
# Trader Performance vs Market Sentiment

## Overview

This project analyzes how Bitcoin Fear & Greed sentiment impacts trader behavior and performance on Hyperliquid. 

An interactive Streamlit dashboard is built to compare:
- Daily PnL across sentiment regimes
- Win rate differences
- Trade frequency and exposure shifts
- Trader segmentation (Exposure, Frequency, Consistency)

---

## Methodology

1. Data Cleaning & Preparation
   - Converted timestamps to daily level
   - Standardized column names
   - Merged trader data with sentiment data on date
   - Created daily trader-level metrics:
     - Daily PnL
     - Win rate
     - Average trade size
     - Trade count
     - Long/Short ratio
     - Max drawdown proxy

2. Sentiment Analysis
   - Compared performance across Fear, Neutral, and Greed regimes
   - Evaluated distribution shifts in PnL and win rate

3. Trader Segmentation
   - High vs Low Exposure traders
   - Frequent vs Infrequent traders
   - Consistent Winners vs Inconsistent vs Net Losers

4. Modeling & Insights
   - Random Forest for feature importance
   - K-Means clustering for behavioral archetypes

---

## Key Insights

- Fear regimes increase PnL volatility and drawdowns.
- High-exposure traders suffer larger losses during Fear periods.
- Consistent winners remain profitable across regimes and perform strongly during Extreme Fear.
- Trading activity increases during volatile sentiment periods.

---

## Strategy Recommendations

1. Reduce Exposure During Fear  
   When Fear & Greed Index < 40, high-exposure traders should reduce position sizes by 30–50% to limit drawdowns.

2. Conditional Aggression for Skilled Traders  
   Consistent winners may maintain or increase activity during Extreme Fear, while inconsistent traders should reduce leverage.

---

## Setup Instructions

### 1. Clone Repository
git clone https://github.com/MGaul6/trader-behavior-insights.git

cd trader-behavior-insights

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Add Dataset

Place the following files inside the `data/` folder:

- sentiment_data.csv
- trader_data.csv

### 4. Run the Application
streamlit run app.py

The dashboard will open automatically in your browser.
