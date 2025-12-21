import pandas as pd
import os

# Load data
df = pd.read_csv("./data/sentiment_dataset.csv")
print(f"Loaded {len(df)} rows")

# Rename timestamp to date
df['date'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None).dt.date
df['date'] = pd.to_datetime(df['date'])

# Sort
df = df.sort_values(['ticker', 'date'])

# Calculate 7-day rolling average
df['rolling_sentiment'] = df.groupby('ticker')['sentiment_confidence']\
    .rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)

# Save
output_path = "./outputs/rolling_sentiment_indices.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Saved to: {output_path}")
print(f"Rows: {len(df)}")
print(f"Tickers: {df['ticker'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")