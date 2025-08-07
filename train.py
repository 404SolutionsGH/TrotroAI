#!/usr/bin/env python3
"""
Trotro AI Model Training Script

Usage:
    python train.py --data data/qa_pairs.csv --model distilbert --epochs 3
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
from datasets import Dataset
import numpy as np
from loguru import logger
import yaml

# Configure logging
logger.add("training.log", rotation="10 MB")

class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
        
    def __len__(self):
        return len(self.encodings["input_ids"])

def load_config():
    """Load training configuration"""
    default_config = {
        "model_name": "distilbert-base-uncased",
        "batch_size": 16,
        "epochs": 3,
        "learning_rate": 2e-5,
        "max_length": 256,
        "output_dir": "models",
    }
    
    config_path = "config/training_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        default_config.update(config)
    
    return default_config

def load_data(file_path):
    """Load and preprocess training data"""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Basic validation
    required_columns = ["question", "answer"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, val_df

def train_model(train_df, val_df, config):
    """Train the model"""
    logger.info("Initializing tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    # Tokenize data
    train_encodings = tokenizer(
        train_df["question"].tolist(),
        train_df["answer"].tolist(),
        truncation=True,
        padding=True,
        max_length=config["max_length"]
    )
    
    val_encodings = tokenizer(
        val_df["question"].tolist(),
        val_df["answer"].tolist(),
        truncation=True,
        padding=True,
        max_length=config["max_length"]
    )
    
    # Create datasets
    train_dataset = QADataset(train_encodings)
    val_dataset = QADataset(val_encodings)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2  # Binary classification
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    output_dir = os.path.join(config["output_dir"], "trotro_ai_model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train Trotro AI model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Model name or path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--output", type=str, default="models", help="Output directory for the model")
    
    args = parser.parse_args()
    
    # Load config and override with command line args
    config = load_config()
    config.update({
        "model_name": args.model,
        "epochs": args.epochs,
        "output_dir": args.output
    })
    
    # Ensure output directory exists
    os.makedirs(config["output_dir"], exist_ok=True)
    
    try:
        # Load and prepare data
        train_df, val_df = load_data(args.data)
        
        # Train model
        model, tokenizer = train_model(train_df, val_df, config)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
