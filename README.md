### Member B (Shi Ying): Analytics Interpretation & Evaluation

In addition to completing the end-to-end analytics pipeline (dependencies are listed in `requirements.txt`), this role focused on
interpreting results, validating assumptions, and documenting analytical insights
derived from the final evaluation plots.

#### Sentiment Time-Series Analysis
- Aggregated 68 sentiment records for AMD across the period 2024-01-01 to 2024-01-09.
- The sentiment index remained consistently positive, with a mean value of 0.493
  and relatively low volatility (std ≈ 0.050).
- Time-series visualization shows short-term fluctuations but no sharp regime shifts,
  suggesting stable market sentiment over the observed window.
- A 7-day rolling average was used to smooth short-term noise and highlight underlying
  sentiment trends.

#### Sentiment–Price Correlation Results
- Pearson correlation analysis between sentiment index and price movements yielded
  a weak positive correlation for AMD (ρ ≈ 0.069).
- Statistical significance testing showed no significant correlations (p < 0.05),
  indicating that short-term sentiment alone is insufficient to explain price changes
  in this sample.
- The correlation distribution plot confirms limited dispersion, reflecting the
  constrained sample size (single ticker, short time horizon).

#### Key Analytical Takeaways
- While sentiment remained generally positive, its explanatory power over short-term
  price movements was limited in this dataset.
- Results highlight the importance of combining sentiment indicators with additional
  market variables (e.g. volume, volatility, macro signals) for stronger predictive
  modeling.
- Findings reinforce the need for longer time horizons and multi-ticker analysis
  when evaluating sentiment-driven financial signals.

#### Deliverables
- Final evaluation plots: sentiment time-series and correlation distribution.
- Summary statistics and correlation report.
- Interactive visualization dashboard for exploratory analysis.

All analytics deliverables, documentation, and result interpretations were completed
and integrated into the final project evaluation.
