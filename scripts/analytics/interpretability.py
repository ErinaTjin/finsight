"""
SHAP and attention analysis for model interpretability
"""
import shap
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def analyze_sentiment_with_shap(model_path, sample_headlines):
    """
    Use SHAP to explain sentiment predictions
    """
    # Load the trained model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Create SHAP explainer
    explainer = shap.Explainer(model, tokenizer)
    
    # Get SHAP values for sample headlines
    shap_values = explainer(sample_headlines)
    
    # Plot force plot for individual predictions
    shap.plots.text(shap_values)
    
    # Plot summary of important words
    shap.plots.bar(shap_values.abs.mean(0))
    
    return shap_values

def create_attention_heatmaps(headlines_with_attention):
    """
    Visualize attention weights from transformer models
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for idx, (headline, attention) in enumerate(headlines_with_attention[:4]):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(attention, cmap='viridis', aspect='auto')
        ax.set_title(f"Headline: {headline[:50]}...")
        ax.set_xticks(range(len(headline.split())))
        ax.set_xticklabels(headline.split(), rotation=45)
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('attention_heatmaps.png')
    plt.show()