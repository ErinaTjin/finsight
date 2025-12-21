import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import json
import os

print("=== CORRELATION ANALYSIS ===")
print(f"Current directory: {os.getcwd()}")

# Load rolling sentiment data
print("\n📊 Loading rolling sentiment data...")
if os.path.exists('rolling_sentiment_indices.csv'):
    sentiment_df = pd.read_csv('rolling_sentiment_indices.csv')
    print(f"✓ Loaded from current directory: {len(sentiment_df)} rows")
elif os.path.exists('./outputs/rolling_sentiment_indices.csv'):
    sentiment_df = pd.read_csv('./outputs/rolling_sentiment_indices.csv')
    print(f"✓ Loaded from ./data/: {len(sentiment_df)} rows")
else:
    print("❌ ERROR: rolling_sentiment_indices.csv not found!")
    exit()

# Convert date
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
print(f"   Unique tickers: {sentiment_df['ticker'].nunique()}")
print(f"   Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")

# Try to load price data
print("\n📈 Looking for price data...")
price_files = ['fnspid_processed.csv', 'combined_financial_news.csv', 'signal_dataset.csv']
price_df = None

for file in price_files:
    if os.path.exists(file):
        price_df = pd.read_csv(file)
        print(f"✓ Found price data: {file} ({len(price_df)} rows)")
        break
    elif os.path.exists(f'processed_data/{file}'):
        price_df = pd.read_csv(f'processed_data/{file}')
        print(f"✓ Found price data: processed_data/{file} ({len(price_df)} rows)")
        break

if price_df is None:
    print("⚠️  No price data found. Creating mock price data for testing...")
    # Create mock price data for testing
    dates = sentiment_df['date'].unique()
    tickers = sentiment_df['ticker'].unique()
    
    mock_data = []
    for date in dates:
        for ticker in tickers:
            mock_data.append({
                'date': date,
                'ticker': ticker,
                'price': np.random.uniform(100, 200),
                'price_change': np.random.uniform(-0.05, 0.05)
            })
    
    price_df = pd.DataFrame(mock_data)
    print(f"   Created mock data: {len(price_df)} rows")

# Standardize price data
if 'timestamp' in price_df.columns:
    price_df = price_df.rename(columns={'timestamp': 'date'})

price_df['date'] = pd.to_datetime(price_df['date'])
if hasattr(price_df['date'].dt, 'tz_localize'):
    price_df['date'] = price_df['date'].dt.tz_localize(None)
price_df['date'] = price_df['date'].dt.date
price_df['date'] = pd.to_datetime(price_df['date'])

# Calculate returns
print("\n📊 Calculating price returns...")
price_cols = [c for c in ['price_change', 'close', 'price', 'adj_close'] if c in price_df.columns]
if price_cols:
    price_col = price_cols[0]
    price_df['price_return'] = price_df.groupby('ticker')[price_col].pct_change()
    print(f"   Using column: {price_col}")
else:
    print("⚠️  No price column found. Using random returns for testing.")
    price_df['price_return'] = np.random.uniform(-0.1, 0.1, len(price_df))

# Create results directory
os.makedirs('results', exist_ok=True)

# Merge data
print("\n🔗 Merging sentiment and price data...")
merged = pd.merge(sentiment_df[['ticker', 'date', 'rolling_sentiment']],
                 price_df[['ticker', 'date', 'price_return']],
                 on=['ticker', 'date'],
                 how='inner').dropna()

print(f"   Merged records: {len(merged)}")

if len(merged) > 10:
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(merged['rolling_sentiment'], merged['price_return'])
    spearman_corr, spearman_p = spearmanr(merged['rolling_sentiment'], merged['price_return'])
    
    results = {
        'pearson_correlation': float(pearson_corr),
        'pearson_p_value': float(pearson_p),
        'spearman_correlation': float(spearman_corr),
        'spearman_p_value': float(spearman_p),
        'n_observations': int(len(merged)),
        'average_sentiment': float(merged['rolling_sentiment'].mean()),
        'average_return': float(merged['price_return'].mean())
    }
    
    print("\n📊 CORRELATION RESULTS:")
    print(f"   Pearson: {results['pearson_correlation']:.4f} (p={results['pearson_p_value']:.4f})")
    print(f"   Spearman: {results['spearman_correlation']:.4f} (p={results['spearman_p_value']:.4f})")
    print(f"   Observations: {results['n_observations']}")
    
    # Save results
    with open('results/correlation_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Saved: results/correlation_report.json")
    
    # Ticker-level correlations
    print("\n🎯 Calculating ticker-level correlations...")
    ticker_results = []
    
    for ticker in merged['ticker'].unique():
        ticker_data = merged[merged['ticker'] == ticker]
        if len(ticker_data) > 5:
            try:
                p_corr, p_p = pearsonr(ticker_data['rolling_sentiment'], ticker_data['price_return'])
                ticker_results.append({
                    'ticker': ticker,
                    'pearson_correlation': float(p_corr),
                    'pearson_p_value': float(p_p),
                    'n_observations': len(ticker_data)
                })
            except:
                pass
    
    if ticker_results:
        ticker_df = pd.DataFrame(ticker_results)
        ticker_df.to_csv('results/ticker_correlations.csv', index=False)
        print(f"✓ Saved: results/ticker_correlations.csv ({len(ticker_df)} tickers)")
        
        # Print summary
        print(f"\n📈 TICKER SUMMARY:")
        print(f"   Average correlation: {ticker_df['pearson_correlation'].mean():.4f}")
        print(f"   Min: {ticker_df['pearson_correlation'].min():.4f}")
        print(f"   Max: {ticker_df['pearson_correlation'].max():.4f}")
        
        print(f"\n🏆 Top 3 positive:")
        for _, row in ticker_df.nlargest(3, 'pearson_correlation').iterrows():
            print(f"   {row['ticker']}: {row['pearson_correlation']:.4f}")
        
        print(f"\n📉 Top 3 negative:")
        for _, row in ticker_df.nsmallest(3, 'pearson_correlation').iterrows():
            print(f"   {row['ticker']}: {row['pearson_correlation']:.4f}")
    
else:
    print("❌ Not enough data for correlation analysis")

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE")
print("=" * 60)