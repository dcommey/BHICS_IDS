# scripts/noise_experiment.py

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from datetime import datetime
import seaborn as sns
from dataclasses import asdict
from functools import wraps
import xgboost as xgb

# Import the metrics tracker
from src.metrics.experiment_metrics import ExperimentTracker, ModelMetrics

# Import model creation functions
from src.ids.ids_model import (
    create_cnn_model, create_lstm_model, create_random_forest_model, 
    create_xgboost_model, compile_and_fit
)
from src.data.cicddos_loader import CICDDoSLoader

# Configuration
config = {
    'data_path': os.path.join('data', 'prepared_data', 'prepared_data_cicddos2019_tftp_challenging.npz'),
    'scaler_path': os.path.join('data', 'prepared_data', 'feature_scaler_cicddos2019_tftp_challenging.joblib'),
    'random_seed': 42
}

def measure_training_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        training_time = time.time() - start_time
        return (*result, training_time)
    return wrapper

def measure_inference_time(model, X_test, model_type):
    """Measure average inference time per sample in nanoseconds"""
    # Warmup prediction
    if model_type in ['cnn', 'lstm']:
        _ = model.predict(X_test[:1])
    else:
        _ = model.predict(X_test[:1])
    
    # Actual timing
    start_time = time.perf_counter_ns()
    if model_type in ['cnn', 'lstm']:
        _ = model.predict(X_test)
    else:
        _ = model.predict(X_test)
    total_time = time.perf_counter_ns() - start_time
    return total_time / len(X_test)  # Average nanoseconds per sample


def add_noise(X, noise_level):
    """Add Gaussian noise to the input data"""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

class TrainingHistory:
    def __init__(self):
        self.histories = {}
    
    def add_history(self, model_type, noise_level, history):
        key = f"{model_type}_{noise_level}"
        self.histories[key] = history
    
    def plot_training_curves(self, output_dir):
        """Plot and save training curves for each model and noise level"""
        metrics = ['accuracy', 'loss']
        
        for metric in metrics:
            plt.figure(figsize=(15, 10))
            for key, history in self.histories.items():
                model_type, noise_level = key.split('_')
                if f'val_{metric}' in history:  # Only for neural network models
                    plt.plot(history[metric], label=f'{model_type} (noise={noise_level}) - train')
                    plt.plot(history[f'val_{metric}'], label=f'{model_type} (noise={noise_level}) - val')
            
            plt.title(f'Training {metric.capitalize()} Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'training_{metric}_curves.png'))
            plt.close()
    
    def save_history_data(self, output_dir):
        """Save raw training history data"""
        history_data = {}
        for key, history in self.histories.items():
            history_data[key] = {k: v for k, v in history.items()}
        
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(history_data, f, indent=2)

@measure_training_time
def train_model(model_type, X_train, y_train, X_val, y_val, input_shape=None):
    """Modified to include validation data and return history"""
    if model_type in ['cnn', 'lstm']:
        if model_type == 'cnn':
            model = create_cnn_model(input_shape)
        else:
            model = create_lstm_model(input_shape)
            
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)
        
        history = compile_and_fit(model, X_train, y_train_cat, X_val, y_val_cat, None)
        return model, history.history
    
    elif model_type == 'rf':
        model = create_random_forest_model()
        model.fit(X_train, y_train)
        return model, {'training_score': [model.score(X_train, y_train)]}
    
    elif model_type == 'xgb':
        model = create_xgboost_model()
        eval_set = [(X_val, y_val)]
        
        # Train the model
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        return model, {
            'training_score': [model.score(X_train, y_train)],
            'validation_score': [model.score(X_val, y_val)],
            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None
        }

def experiment_with_noise(X, y, model_types, noise_levels, output_dir):
    tracker = ExperimentTracker(os.path.join(output_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    training_history = TrainingHistory()
    
    # Convert DataFrame to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    for noise_level in noise_levels:
        print(f"\nExperimenting with noise level: {noise_level}")
        X_train_noisy = add_noise(X_train, noise_level)
        X_val_noisy = add_noise(X_val, noise_level)
        X_test_noisy = add_noise(X_test, noise_level)
        
        for model_type in model_types:
            print(f"Training and evaluating {model_type.upper()} model...")
            
            # Prepare input shape for neural networks
            input_shape = None
            if model_type in ['cnn', 'lstm']:
                X_train_reshaped = X_train_noisy.reshape((X_train_noisy.shape[0], X_train_noisy.shape[1], 1))
                X_val_reshaped = X_val_noisy.reshape((X_val_noisy.shape[0], X_val_noisy.shape[1], 1))
                X_test_reshaped = X_test_noisy.reshape((X_test_noisy.shape[0], X_test_noisy.shape[1], 1))
                input_shape = (X_train_reshaped.shape[1], 1)
            else:
                X_train_reshaped = X_train_noisy
                X_val_reshaped = X_val_noisy
                X_test_reshaped = X_test_noisy
            
            # Train model and measure time
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            try:
                model, history, training_time = train_model(
                    model_type, X_train_reshaped, y_train, 
                    X_val_reshaped, y_val, input_shape
                )
                
                # Save training history
                training_history.add_history(model_type, noise_level, history)
                
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_usage = final_memory - initial_memory
                
                # Measure inference time
                inference_time = measure_inference_time(model, X_test_reshaped, model_type)
                
                # Evaluate model
                if model_type in ['cnn', 'lstm']:
                    y_pred = (model.predict(X_test_reshaped)[:, 1] > 0.5).astype(int)
                else:
                    y_pred = model.predict(X_test_reshaped)
                
                # Calculate metrics
                metrics = ModelMetrics(
                    model_type=model_type,
                    noise_level=noise_level,
                    accuracy=accuracy_score(y_test, y_pred),
                    precision=precision_score(y_test, y_pred),
                    recall=recall_score(y_test, y_pred),
                    f1_score=f1_score(y_test, y_pred),
                    training_time=training_time,
                    inference_time=inference_time,
                    epoch_times=history.get('epoch_times', []),
                    memory_usage=memory_usage
                )
                
                tracker.add_metrics(metrics)
                
                print(f"Results for {model_type} at noise level {noise_level}:")
                print(f"Accuracy: {metrics.accuracy:.4f}")
                print(f"Training time: {metrics.training_time:.2f}s")
                print(f"Inference time: {metrics.inference_time:.2f} ns/sample")
                
            except Exception as e:
                print(f"Error training {model_type} model with noise level {noise_level}: {str(e)}")
                continue
    
    # Save all metrics and training curves
    tracker.save_metrics()
    training_history.plot_training_curves(output_dir)
    training_history.save_history_data(output_dir)
    
    return tracker, training_history

def plot_noise_impact_summary(tracker, output_dir):
    """Create summary plots for noise impact across models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    df = pd.DataFrame([asdict(m) for m in tracker.metrics])
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            plt.plot(model_data['noise_level'], model_data[metric], 
                    marker='o', label=model_type.upper())
        
        plt.title(f'{metric.capitalize()} vs Noise Level')
        plt.xlabel('Noise Level')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'noise_impact_{metric}.png'))
        plt.close()

def main():
    # Configuration
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_loader = CICDDoSLoader(config['data_path'], config['scaler_path'])
    data_loader.load_data()
    X, y = data_loader.get_train_data()
    
    model_types = ['rf', 'xgb', 'cnn', 'lstm']
    noise_levels = [0, 0.1, 0.5, 1, 2, 4]
    
    # Run experiment
    tracker, training_history = experiment_with_noise(X, y, model_types, noise_levels, output_dir)
    
    # Create summary plots
    plot_noise_impact_summary(tracker, output_dir)
    
    print(f"\nExperiment results saved in {output_dir}")

if __name__ == "__main__":
    main()