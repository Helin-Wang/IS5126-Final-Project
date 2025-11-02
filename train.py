#!/usr/bin/env python3
"""
RoBERTa-based Regression Model Training Script

Usage:
    python train_roberta_regression.py \
        --csv_path womens_reviews_llm_latest.csv \
        --text_col "Review Text" \
        --target_col y_norm \
        --output_dir ./roberta_regression_output \
        --batch_size 16 \
        --learning_rate 2e-5 \
        --num_epochs 10
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RoBERTaRegressionModel(nn.Module):
    """RoBERTa-base encoder with regression head using [CLS] pooling."""
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Use [CLS] token (first token) for pooling
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and regression
        pooled = self.dropout(cls_hidden)
        logits = self.regressor(pooled)
        
        # Remove batch dimension if single output
        logits = logits.squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels.float())
        
        return {"loss": loss, "logits": logits}


def load_and_preprocess_data(
    csv_path: str,
    text_col: str,
    target_col: str,
    scale: float = 100.0,
    random_state: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Load CSV data and prepare train/val datasets.
    
    Returns:
        train_dataset, val_dataset with 'text' and 'labels' fields
        Labels are already log1p transformed: log(1 + scale * y_norm)
    """
    df = pd.read_csv(csv_path)
    
    # Extract text and target
    texts = df[text_col].fillna("").astype(str).tolist()
    y = np.clip(df[target_col].astype(float).to_numpy(), 0.0, 1.0)
    
    # Apply log1p transformation: log(1 + scale * y)
    y_transformed = np.log1p(y * scale)
    
    # Train/val split (80/20)
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(texts))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=random_state
    )
    
    train_texts = [texts[i] for i in train_indices]
    train_labels = y_transformed[train_indices].tolist()
    
    val_texts = [texts[i] for i in val_indices]
    val_labels = y_transformed[val_indices].tolist()
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "labels": train_labels,
    })
    
    val_dataset = Dataset.from_dict({
        "text": val_texts,
        "labels": val_labels,
    })
    
    return train_dataset, val_dataset, y[train_indices], y[val_indices]


def safe_expm1(x: np.ndarray, scale: float = 100.0, clip_max: float = 10.0) -> np.ndarray:
    """Safe inverse of log(1 + scale*y): avoid overflow, clip to [0, 1]."""
    x = np.clip(x, None, clip_max)
    out = np.expm1(x) / scale
    out[np.isnan(out)] = 0
    out[np.isinf(out)] = 0
    return np.clip(out, 0, 1.0)


def compute_metrics_factory(y_original: Optional[np.ndarray] = None, scale: float = 100.0):
    """
    Factory function to create compute_metrics function with original y values.
    
    Returns:
        compute_metrics function that can be used by Trainer
    """
    def compute_metrics(eval_pred):
        """
        Compute metrics in both transformed and original domains.
        
        Args:
            eval_pred: tuple of (predictions, labels) from Trainer
        """
        predictions, labels = eval_pred
        
        # Ensure predictions and labels are 1D arrays
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if labels.ndim > 1:
            labels = labels.flatten()
        
        # Transformed domain metrics (what model predicts)
        mse_t = mean_squared_error(labels, predictions)
        rmse_t = np.sqrt(mse_t)
        mae_t = mean_absolute_error(labels, predictions)
        
        # R2 in transformed domain
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2_t = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        metrics = {
            "mse": mse_t,
            "rmse": rmse_t,
            "mae": mae_t,
            "r2": r2_t,
        }
        
        # Original domain metrics (inverse transform)
        if y_original is not None and len(y_original) == len(predictions):
            pred_original = safe_expm1(predictions, scale=scale)
            labels_original = y_original
            
            mse_o = mean_squared_error(labels_original, pred_original)
            rmse_o = np.sqrt(mse_o)
            mae_o = mean_absolute_error(labels_original, pred_original)
            r2_o = r2_score(labels_original, pred_original)
            
            metrics.update({
                "mse_original": mse_o,
                "rmse_original": rmse_o,
                "mae_original": mae_o,
                "r2_original": r2_o,
            })
        
        return metrics
    
    return compute_metrics


class CustomTrainer(Trainer):
    """Custom Trainer to track gradient norms and handle metrics computation."""
    
    def __init__(self, y_train_original=None, y_val_original=None, scale=100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_train_original = y_train_original
        self.y_val_original = y_val_original
        self.scale = scale
        self.tb_writer = None
        
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Override log to add gradient norms and custom metrics."""
        super().log(logs, *args, **kwargs)
        
        # Log to TensorBoard if available
        if self.tb_writer is not None and hasattr(self.state, 'global_step'):
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.state.global_step)
    
    def training_step(self, model, inputs, *args, **kwargs):
        """Override to track gradient norms."""
        loss = super().training_step(model, inputs, *args, **kwargs)
        
        # Track gradient norms (after backward pass)
        if self.tb_writer is not None and hasattr(self.state, 'global_step') and self.state.global_step % self.args.logging_steps == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.tb_writer.add_scalar("train/gradient_norm", total_norm, self.state.global_step)
        
        return loss


def main():
    parser = argparse.ArgumentParser(description="Train RoBERTa regression model")
    
    # Data arguments
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--text_col", type=str, default="Review Text", help="Text column name")
    parser.add_argument("--target_col", type=str, default="y_norm", help="Target column name")
    parser.add_argument("--scale", type=float, default=100.0, help="Scale factor for log1p transform")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Pre-trained model name")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate (0.1-0.2)")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./roberta_regression_output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (default 10%%)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Evaluation arguments
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch"], help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps (if strategy=steps)")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["steps", "epoch"], help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps (if strategy=steps)")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    
    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--early_stopping_metric", type=str, default="eval_rmse", help="Metric for early stopping")
    
    # Loss function
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "huber"], help="Loss function type")
    parser.add_argument("--huber_delta", type=float, default=1.0, help="Huber loss delta parameter")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    tb_log_dir = output_dir / "runs" / "train"
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
    
    print("=" * 80)
    print("Loading and preprocessing data...")
    print("=" * 80)
    
    # Load data
    train_dataset, val_dataset, y_train_orig, y_val_orig = load_and_preprocess_data(
        csv_path=args.csv_path,
        text_col=args.text_col,
        target_col=args.target_col,
        scale=args.scale,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Initialize model
    print("Initializing model...")
    model = RoBERTaRegressionModel(
        model_name=args.model_name,
        dropout_rate=args.dropout_rate,
    )
    
    # Setup loss function - modify model to use Huber loss if needed
    if args.loss_type == "huber":
        loss_fn = nn.HuberLoss(delta=args.huber_delta)
        # Store original forward
        original_forward = model.forward
        
        def forward_with_huber(input_ids, attention_mask=None, labels=None):
            outputs = original_forward(input_ids, attention_mask, labels=None)
            loss = None
            if labels is not None:
                loss = loss_fn(outputs["logits"], labels.float())
            outputs["loss"] = loss
            return outputs
        
        model.forward = forward_with_huber
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        logging_steps=args.logging_steps,
        save_total_limit=3,  # Keep best + last 2
        load_best_model_at_end=True,
        metric_for_best_model=args.early_stopping_metric,
        greater_is_better=False,  # Lower is better for RMSE/MAE
        report_to="tensorboard",
        logging_dir=str(tb_log_dir),
    )
    
    # Compute metrics function
    compute_metrics_fn = compute_metrics_factory(
        y_original=y_val_orig,
        scale=args.scale,
    )
    
    # Initialize trainer
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
            )
        )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        y_train_original=y_train_orig,
        y_val_original=y_val_orig,
        scale=args.scale,
        callbacks=callbacks,
    )
    trainer.tb_writer = tb_writer
    
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    # Train
    train_result = trainer.train()
    
    print("=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Final evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    print("\nFinal Evaluation Results:")
    print("-" * 80)
    for key, value in eval_results.items():
        print(f"{key}: {value:.6f}")
    
    # Save final model and tokenizer
    print("\nSaving model and tokenizer...")
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(exist_ok=True)
    
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # Save best model (already saved by Trainer if load_best_model_at_end=True)
    # The best model is automatically saved to the output_dir by Trainer
    print(f"Best model saved at: {output_dir}")
    
    # Save regression head separately
    regression_head_dir = output_dir / "regression_head"
    regression_head_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "regressor.state_dict": model.regressor.state_dict(),
            "dropout.state_dict": model.dropout.state_dict(),
            "hidden_size": model.encoder.config.hidden_size,
            "dropout_rate": args.dropout_rate,
        },
        regression_head_dir / "pytorch_model.bin",
    )
    
    # Save training config
    config = {
        "model_name": args.model_name,
        "dropout_rate": args.dropout_rate,
        "scale": args.scale,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "loss_type": args.loss_type,
        "final_metrics": eval_results,
    }
    
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete! Output saved to: {output_dir}")
    print(f"TensorBoard logs: {tb_log_dir}")
    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir {tb_log_dir}")
    
    tb_writer.close()


if __name__ == "__main__":
    main()
