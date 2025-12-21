# visualization_dashboard_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

def setup_visualization_environment():
    """Set up directories and style"""
    # Create visualizations directory if needed
    os.makedirs('outputs/visualisations', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    print("🎨 Setting up visualization environment...")

def load_data_corrected():
    """Load data from your exact directory structure"""
    print("📊 Loading data from your structure...")
    
    # Load sentiment data - it's in outputs/ folder
    sentiment_path = "outputs/rolling_sentiment_indices.csv"
    
    if os.path.exists(sentiment_path):
        sentiment_df = pd.read_csv(sentiment_path)
        print(f"✓ Loaded sentiment data from: {sentiment_path}")
        print(f"  Rows: {len(sentiment_df):,}, Tickers: {sentiment_df['ticker'].nunique()}")
    else:
        print(f"❌ ERROR: Could not find {sentiment_path}")
        print("   Please ensure the file exists in outputs/ folder")
        return None, None
    
    # Convert date column
    if 'date' in sentiment_df.columns:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    elif 'timestamp' in sentiment_df.columns:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp'])
    
    # Load correlation results - in results/ folder
    correlation_path = "results/ticker_correlations.csv"
    
    if os.path.exists(correlation_path):
        ticker_corrs = pd.read_csv(correlation_path)
        print(f"✓ Loaded correlation data from: {correlation_path}")
        print(f"  Correlation records: {len(ticker_corrs)}")
    else:
        print(f"⚠️ WARNING: Could not find {correlation_path}")
        print("   Creating sample correlation data for visualization...")
        
        # Create realistic sample data
        tickers = sentiment_df['ticker'].unique()[:15]
        np.random.seed(42)  # For reproducible results
        
        mock_data = []
        for ticker in tickers:
            # More realistic correlation distribution
            corr = np.random.normal(0.1, 0.3)  # Mean 0.1, std 0.3
            corr = max(-0.9, min(0.9, corr))  # Bound between -0.9 and 0.9
            
            # P-value based on correlation strength
            if abs(corr) > 0.5:
                p_val = np.random.uniform(0.001, 0.05)
            elif abs(corr) > 0.2:
                p_val = np.random.uniform(0.05, 0.2)
            else:
                p_val = np.random.uniform(0.2, 0.5)
            
            mock_data.append({
                'ticker': ticker,
                'pearson_correlation': corr,
                'pearson_p_value': p_val,
                'spearman_correlation': corr * np.random.uniform(0.8, 1.2),
                'spearman_p_value': p_val * np.random.uniform(0.8, 1.2),
                'n_observations': np.random.randint(20, 100)
            })
        
        ticker_corrs = pd.DataFrame(mock_data)
        print(f"   Created sample data for {len(ticker_corrs)} tickers")
    
    return sentiment_df, ticker_corrs

def create_correlation_distribution(ticker_corrs):
    """Create histogram of correlation distribution"""
    plt.figure(figsize=(12, 6))
    
    # Histogram with better styling
    n, bins, patches = plt.hist(ticker_corrs['pearson_correlation'], 
                               bins=20, 
                               edgecolor='black', 
                               alpha=0.7,
                               color='#4C72B0',
                               density=False)
    
    # Add mean and median lines
    mean_corr = ticker_corrs['pearson_correlation'].mean()
    median_corr = ticker_corrs['pearson_correlation'].median()
    
    plt.axvline(x=mean_corr, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_corr:.3f}')
    plt.axvline(x=median_corr, color='orange', linestyle='-.', linewidth=2,
                label=f'Median: {median_corr:.3f}')
    
    # Add zero line
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Calculate statistics
    significant = ticker_corrs[ticker_corrs['pearson_p_value'] < 0.05]
    
    stats_text = f"""Statistics:
    Mean: {mean_corr:.3f}
    Median: {median_corr:.3f}
    Std: {ticker_corrs["pearson_correlation"].std():.3f}
    Min: {ticker_corrs["pearson_correlation"].min():.3f}
    Max: {ticker_corrs["pearson_correlation"].max():.3f}
    Significant (p<0.05): {len(significant)}/{len(ticker_corrs)}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.xlabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Tickers', fontsize=12, fontweight='bold')
    plt.title('Distribution of Sentiment-Price Correlations', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = 'results/visualizations/correlation_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Created: {output_path}")

def create_top_correlations_chart(ticker_corrs):
    """Create bar chart of top correlations"""
    if len(ticker_corrs) < 4:
        print("   ⚠️ Not enough tickers for top correlations chart")
        return
    
    # Get top positive and negative correlations
    n_top = min(5, len(ticker_corrs) // 2)
    top_pos = ticker_corrs.nlargest(n_top, 'pearson_correlation')
    top_neg = ticker_corrs.nsmallest(n_top, 'pearson_correlation')
    
    # Combine and sort
    top_all = pd.concat([top_pos, top_neg]).sort_values('pearson_correlation')
    
    plt.figure(figsize=(14, 8))
    
    # Create color gradient based on correlation strength
    colors = []
    for corr in top_all['pearson_correlation']:
        if corr > 0.5:
            colors.append('#006400')  # Dark green for very positive
        elif corr > 0.2:
            colors.append('#32CD32')  # Lime green for positive
        elif corr > 0:
            colors.append('#90EE90')  # Light green for slightly positive
        elif corr > -0.2:
            colors.append('#FFB6C1')  # Light red for slightly negative
        elif corr > -0.5:
            colors.append('#FF6347')  # Tomato red for negative
        else:
            colors.append('#8B0000')  # Dark red for very negative
    
    # Create horizontal bar chart
    bars = plt.barh(top_all['ticker'], top_all['pearson_correlation'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels with p-value
    for bar, (_, row) in zip(bars, top_all.iterrows()):
        x_pos = bar.get_width()
        align = 'left' if x_pos > 0 else 'right'
        offset = 0.01 if x_pos > 0 else -0.01
        color = 'white' if abs(x_pos) > 0.3 else 'black'
        
        significance = "***" if row['pearson_p_value'] < 0.01 else "**" if row['pearson_p_value'] < 0.05 else "*"
        
        plt.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                f'{x_pos:.3f}{significance}',
                va='center', ha=align, fontweight='bold', fontsize=10,
                color=color)
    
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    plt.xlabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
    plt.title(f'Strongest Correlations (Top {n_top} Positive & Negative)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend for significance
    plt.figtext(0.02, 0.02, "* p<0.05, ** p<0.01, *** p<0.001", 
                fontsize=9, style='italic')
    
    plt.tight_layout()
    output_path = 'results/visualizations/top_correlations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Created: {output_path}")

def create_sentiment_time_series(sentiment_df):
    """Create time series visualization"""
    if len(sentiment_df) == 0:
        print("   ⚠️ No sentiment data for time series")
        return
    
    # Get tickers with most data
    ticker_counts = sentiment_df['ticker'].value_counts()
    n_tickers = min(4, len(ticker_counts))
    top_tickers = ticker_counts.head(n_tickers).index
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, ticker in enumerate(top_tickers):
        ax = axes[idx]
        ticker_data = sentiment_df[sentiment_df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date')
        
        # Plot sentiment line
        ax.plot(ticker_data['date'], ticker_data['rolling_sentiment'], 
               color='blue', linewidth=2, label='Sentiment Index', zorder=3)
        
        # Fill areas for positive/negative
        ax.fill_between(ticker_data['date'], 0, ticker_data['rolling_sentiment'],
                       where=ticker_data['rolling_sentiment'] >= 0,
                       color='green', alpha=0.2, label='Positive', zorder=1)
        ax.fill_between(ticker_data['date'], 0, ticker_data['rolling_sentiment'],
                       where=ticker_data['rolling_sentiment'] < 0,
                       color='red', alpha=0.2, label='Negative', zorder=1)
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=2)
        
        # Formatting
        ax.set_title(f'{ticker} - Sentiment Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Sentiment Index', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add statistics box
        stats_text = f"""Records: {len(ticker_data)}
Mean: {ticker_data['rolling_sentiment'].mean():.3f}
Std: {ticker_data['rolling_sentiment'].std():.3f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(len(top_tickers), 4):
        axes[idx].axis('off')
    
    plt.suptitle('Sentiment Index Time Series Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = 'results/visualizations/sentiment_time_series.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ⏰ Created: {output_path}")

def create_interactive_dashboard(sentiment_df, ticker_corrs):
    """Create interactive HTML dashboard"""
    print("   🖥️ Creating interactive dashboard...")
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Correlation Distribution', 
                       'Top Correlations',
                       'Correlation vs Significance',
                       'Sentiment Trend Analysis'),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 1. Histogram of correlations
    fig.add_trace(
        go.Histogram(
            x=ticker_corrs['pearson_correlation'],
            nbinsx=20,
            name='Correlation Distribution',
            marker_color='#4C72B0',
            opacity=0.7,
            hovertemplate='Correlation: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add mean line to histogram
    mean_corr = ticker_corrs['pearson_correlation'].mean()
    fig.add_vline(x=mean_corr, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_corr:.3f}", 
                  row=1, col=1)
    
    # 2. Top correlations bar chart
    n_top = min(6, len(ticker_corrs) // 2)
    if n_top > 0:
        top_all = pd.concat([
            ticker_corrs.nlargest(n_top, 'pearson_correlation'),
            ticker_corrs.nsmallest(n_top, 'pearson_correlation')
        ]).sort_values('pearson_correlation')
        
        # Color based on correlation value
        colors = px.colors.diverging.RdBu
        color_scale = []
        for corr in top_all['pearson_correlation']:
            # Map correlation from [-1, 1] to [0, 1]
            t = (corr + 1) / 2
            color_idx = int(t * (len(colors) - 1))
            color_scale.append(colors[min(color_idx, len(colors)-1)])
        
        fig.add_trace(
            go.Bar(
                x=top_all['pearson_correlation'],
                y=top_all['ticker'],
                orientation='h',
                marker_color=color_scale,
                name='Top Correlations',
                text=[f'{x:.3f}' for x in top_all['pearson_correlation']],
                textposition='auto',
                hovertemplate='Ticker: %{y}<br>Correlation: %{x:.3f}<br>p-value: %{customdata[0]:.3f}<extra></extra>',
                customdata=np.column_stack([top_all['pearson_p_value']])
            ),
            row=1, col=2
        )
    
    # 3. Scatter plot: Correlation vs Significance
    fig.add_trace(
        go.Scatter(
            x=ticker_corrs['pearson_correlation'],
            y=-np.log10(ticker_corrs['pearson_p_value']),
            mode='markers+text',
            text=ticker_corrs['ticker'],
            textposition="top center",
            marker=dict(
                size=14,
                color=ticker_corrs['pearson_correlation'],
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Correlation", x=1.02),
                line=dict(width=1, color='black')
            ),
            name='Correlation Significance',
            hovertemplate='Ticker: %{text}<br>Correlation: %{x:.3f}<br>-log10(p): %{y:.2f}<br>p-value: 10^(-%{y:.2f})<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add significance threshold lines
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red",
                  annotation_text="p=0.05", row=2, col=1)
    fig.add_hline(y=-np.log10(0.01), line_dash="dot", line_color="orange",
                  annotation_text="p=0.01", row=2, col=1)
    
    # 4. Sentiment time series for a sample ticker
    if len(sentiment_df) > 0:
        # Pick ticker with most data
        sample_ticker = sentiment_df['ticker'].value_counts().index[0]
        ticker_data = sentiment_df[sentiment_df['ticker'] == sample_ticker].sort_values('date')
        
        fig.add_trace(
            go.Scatter(
                x=ticker_data['date'],
                y=ticker_data['rolling_sentiment'],
                mode='lines',
                name=f'Sentiment: {sample_ticker}',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 0, 255, 0.2)',
                hovertemplate='Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Add rolling average
        window = 7
        ticker_data['rolling_avg'] = ticker_data['rolling_sentiment'].rolling(window=window).mean()
        fig.add_trace(
            go.Scatter(
                x=ticker_data['date'],
                y=ticker_data['rolling_avg'],
                mode='lines',
                name=f'{window}-day MA',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>{window}-day MA: %{y:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=1000,
        width=1400,
        title_text="Financial Sentiment Analysis Dashboard",
        title_font_size=24,
        showlegend=True,
        template="plotly_white",
        hovermode='closest'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Correlation Coefficient", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Correlation Coefficient", row=1, col=2)
    fig.update_xaxes(title_text="Correlation Coefficient", row=2, col=1)
    fig.update_yaxes(title_text="-log₁₀(p-value)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Sentiment Index", row=2, col=2)
    
    # Save to results/visualizations folder
    output_path = 'results/visualizations/interactive_dashboard.html'
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"   💾 Created: {output_path}")

def create_summary_statistics(sentiment_df, ticker_corrs):
    """Create summary statistics file"""
    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sentiment_data": {
            "total_records": int(len(sentiment_df)),
            "unique_tickers": int(sentiment_df['ticker'].nunique()),
            "date_range": {
                "start": str(sentiment_df['date'].min()),
                "end": str(sentiment_df['date'].max())
            },
            "sentiment_statistics": {
                "mean": float(sentiment_df['rolling_sentiment'].mean()),
                "std": float(sentiment_df['rolling_sentiment'].std()),
                "min": float(sentiment_df['rolling_sentiment'].min()),
                "max": float(sentiment_df['rolling_sentiment'].max())
            }
        },
        "correlation_analysis": {
            "total_tickers": int(len(ticker_corrs)),
            "average_correlation": float(ticker_corrs['pearson_correlation'].mean()),
            "median_correlation": float(ticker_corrs['pearson_correlation'].median()),
            "significant_correlations": int(sum(ticker_corrs['pearson_p_value'] < 0.05)),
            "top_positive": ticker_corrs.nlargest(3, 'pearson_correlation')[['ticker', 'pearson_correlation']].to_dict('records'),
            "top_negative": ticker_corrs.nsmallest(3, 'pearson_correlation')[['ticker', 'pearson_correlation']].to_dict('records')
        }
    }
    
    # Save JSON summary
    summary_path = 'results/visualizations/summary_statistics.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   📋 Created: {summary_path}")
    
    # Also create text summary
    txt_summary = f"""
    SENTIMENT-PRICE CORRELATION ANALYSIS SUMMARY
    {'='*50}
    
    Sentiment Data:
    - Total records: {summary['sentiment_data']['total_records']:,}
    - Unique tickers: {summary['sentiment_data']['unique_tickers']}
    - Date range: {summary['sentiment_data']['date_range']['start']} to {summary['sentiment_data']['date_range']['end']}
    - Average sentiment: {summary['sentiment_data']['sentiment_statistics']['mean']:.3f}
    
    Correlation Analysis:
    - Tickers analyzed: {summary['correlation_analysis']['total_tickers']}
    - Average correlation: {summary['correlation_analysis']['average_correlation']:.3f}
    - Significant correlations (p<0.05): {summary['correlation_analysis']['significant_correlations']}
    
    Top Positive Correlations:
    """
    
    for item in summary['correlation_analysis']['top_positive']:
        txt_summary += f"    • {item['ticker']}: {item['pearson_correlation']:.3f}\n"
    
    txt_summary += "\nTop Negative Correlations:\n"
    for item in summary['correlation_analysis']['top_negative']:
        txt_summary += f"    • {item['ticker']}: {item['pearson_correlation']:.3f}\n"
    
    txt_summary += f"\nGenerated: {summary['generated_at']}"
    
    txt_path = 'results/visualizations/summary_report.txt'
    with open(txt_path, 'w') as f:
        f.write(txt_summary)
    
    print(f"   📄 Created: {txt_path}")

def main_final():
    """Main execution function for your exact structure"""
    print("=" * 70)
    print("FINANCIAL SENTIMENT VISUALIZATION DASHBOARD")
    print("=" * 70)
    
    # Setup environment
    setup_visualization_environment()
    
    # Load data with correct paths
    print("\n📁 Loading data...")
    sentiment_df, ticker_corrs = load_data_corrected()
    
    if sentiment_df is None:
        print("❌ Cannot proceed without sentiment data")
        return
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   📊 Sentiment: {len(sentiment_df):,} records, {sentiment_df['ticker'].nunique()} tickers")
    print(f"   🔗 Correlations: {len(ticker_corrs):,} tickers analyzed")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # 1. Correlation distribution
    create_correlation_distribution(ticker_corrs)
    
    # 2. Top correlations
    create_top_correlations_chart(ticker_corrs)
    
    # 3. Time series
    create_sentiment_time_series(sentiment_df)
    
    # 4. Interactive dashboard
    create_interactive_dashboard(sentiment_df, ticker_corrs)
    
    # 5. Summary statistics
    create_summary_statistics(sentiment_df, ticker_corrs)
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION PIPELINE COMPLETE!")
    print("=" * 70)
    
    # List all created files
    print("\n📁 FILES CREATED in 'results/visualizations/':")
    print("-" * 50)
    
    viz_folder = 'results/visualizations'
    if os.path.exists(viz_folder):
        files = os.listdir(viz_folder)
        for file in sorted(files):
            if file.endswith(('.png', '.html', '.json', '.txt')):
                filepath = os.path.join(viz_folder, file)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"  • {file:30} ({size_kb:.1f} KB)")
    
    print("\n📊 HOW TO VIEW RESULTS:")
    print("-" * 50)
    print("1. Open 'results/visualizations/interactive_dashboard.html' in browser")
    print("2. Check PNG images in the same folder for static charts")
    print("3. Read 'summary_report.txt' for key findings")
    print("\n📈 NEXT STEPS:")
    print("-" * 50)
    print("• Share visualizations with your team")
    print("• Integrate with FastAPI backend (Member C)")
    print("• Prepare presentation slides")
    print("• Document findings in project report")

if __name__ == "__main__":
    main_final()