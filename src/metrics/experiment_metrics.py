import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import pandas as pd

@dataclass
class ModelMetrics:
    model_type: str
    noise_level: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    inference_time: float
    epoch_times: List[float]
    memory_usage: float
    
class ExperimentTracker:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metrics: List[ModelMetrics] = []
        
    def add_metrics(self, metrics: ModelMetrics):
        self.metrics.append(metrics)
        
    def save_metrics(self):
        # Convert to DataFrame for easier analysis
        metrics_dict = [asdict(m) for m in self.metrics]
        df = pd.DataFrame(metrics_dict)
        
        # Save as CSV
        df.to_csv(f"{self.output_path}_metrics.csv", index=False)
        
        # Save as JSON for complete data preservation
        with open(f"{self.output_path}_metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Generate summary statistics
        summary = self._generate_summary(df)
        with open(f"{self.output_path}_summary.txt", 'w') as f:
            f.write(summary)
            
    def _generate_summary(self, df: pd.DataFrame) -> str:
        summary = []
        summary.append("=== Experiment Summary ===\n")
        
        # Per model type analysis
        for model_type in df['model_type'].unique():
            model_df = df[df['model_type'] == model_type]
            summary.append(f"\nModel: {model_type.upper()}")
            summary.append("-" * 40)
            
            # Performance metrics
            summary.append("\nPerformance Metrics (mean ± std):")
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in metrics:
                mean = model_df[metric].mean()
                std = model_df[metric].std()
                summary.append(f"{metric}: {mean:.4f} ± {std:.4f}")
            
            # Timing metrics
            summary.append("\nTiming Metrics:")
            summary.append(f"Avg Training Time: {model_df['training_time'].mean():.2f}s")
            summary.append(f"Avg Inference Time: {model_df['inference_time'].mean()*1000:.2f}ms")
            
            # Impact of noise
            summary.append("\nNoise Impact Analysis:")
            noise_impact = model_df.groupby('noise_level')['accuracy'].mean()
            for noise, acc in noise_impact.items():
                summary.append(f"Noise {noise}: {acc:.4f} accuracy")
                
        return "\n".join(summary)