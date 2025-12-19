import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
from datasets import Dataset
import numpy as np

class SentimentClassifier(nn.Module):
    def __init__(self, model_name: str = "yiyanghkust/finbert-tone", num_labels: int = 3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            
        return (loss, logits) if loss is not None else logits

class SentimentTrainer:
    def __init__(self, model_name: str = "yiyanghkust/finbert-tone"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def preprocess_data(self, df: pd.DataFrame):
        """Convert dataframe to tokenized dataset"""
        texts = df['clean_text'].tolist()
        labels = df['clean_sentiment'].tolist()
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        
        return dataset
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1_weighted': f1_score(labels, predictions, average='weighted'),
            'f1_macro': f1_score(labels, predictions, average='macro')
        }
    
    def train(self, train_dataset, val_dataset, output_dir: str = "./models/sentiment"):
        # Initialize model
        model = SentimentClassifier(self.model_name)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        trainer.train()
        
        # Save
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer