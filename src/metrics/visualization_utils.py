import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ResultsVisualizer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.plot_dir = self.output_dir / 'plots'
        self.tables_dir = self.output_dir / 'tables'
        self.plot_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_theme()
        
        # Define distinct color scheme and styles for models
        self.model_styles = {
            'rf': {'color': '#2ecc71', 'linestyle': '-', 'marker': 'o', 'linewidth': 2.5, 'markersize': 8},
            'xgb': {'color': '#e74c3c', 'linestyle': '--', 'marker': 's', 'linewidth': 2.5, 'markersize': 8},
            'cnn': {'color': '#3498db', 'linestyle': '-.', 'marker': '^', 'linewidth': 2.5, 'markersize': 8},
            'lstm': {'color': '#9b59b6', 'linestyle': ':', 'marker': 'D', 'linewidth': 2.5, 'markersize': 8}
        }
        
    def load_results(self, metrics_file):
        """Load results from CSV file"""
        df = pd.read_csv(metrics_file)
        print("\nLoaded results shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nModel types:", df['model_type'].unique())
        return df
    
    def create_noise_impact_plots(self, df):
        """Create plots showing impact of noise on different metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Impact of Noise on Model Performance', fontsize=16, y=1.02)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Create error plot for each model
            for model in df['model_type'].unique():
                model_data = df[df['model_type'] == model]
                style = self.model_styles[model]
                
                ax.plot(model_data['noise_level'], model_data[metric],
                       label=model.upper(),
                       **style)
            
            ax.set_xlabel('Noise Level', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10, framealpha=0.9)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, pad=10)
            
            # Add minor gridlines
            ax.grid(True, which='minor', alpha=0.15)
            ax.minorticks_on()
            
            # Set background color
            ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        save_path = self.plot_dir / 'noise_impact_all_metrics.pdf'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved noise impact plot to {save_path}")
        
    def create_timing_comparison(self, df):
        """Create plots comparing training and inference times"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Custom color palette
        palette = [self.model_styles[model]['color'] for model in df['model_type'].unique()]
        
        # Training time comparison
        sns.boxplot(data=df, x='model_type', y='training_time', 
                   ax=ax1, palette=palette)
        ax1.set_title('Training Time Distribution', fontsize=12, pad=10)
        ax1.set_ylabel('Time (seconds)', fontsize=10)
        ax1.set_xlabel('Model', fontsize=10)
        
        # Inference time comparison (log scale)
        sns.boxplot(data=df, x='model_type', y='inference_time', 
                   ax=ax2, palette=palette)
        ax2.set_title('Inference Time Distribution', fontsize=12, pad=10)
        ax2.set_ylabel('Time (nanoseconds/sample)', fontsize=10)
        ax2.set_xlabel('Model', fontsize=10)
        ax2.set_yscale('log')
        
        # Style both plots
        for ax in [ax1, ax2]:
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            # Rotate x-labels if needed
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self.plot_dir / 'timing_comparison.pdf'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved timing comparison to {save_path}")
        
    def create_memory_usage_plot(self, df):
        """Create plot showing memory usage across models and noise levels"""
        plt.figure(figsize=(10, 6))
        
        for model in df['model_type'].unique():
            model_data = df[df['model_type'] == model]
            style = self.model_styles[model]
            
            plt.plot(model_data['noise_level'], model_data['memory_usage'],
                    label=model.upper(),
                    **style)
        
        plt.xlabel('Noise Level', fontsize=10)
        plt.ylabel('Memory Usage (MB)', fontsize=10)
        plt.title('Memory Usage vs Noise Level', fontsize=12, pad=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=10, framealpha=0.9)
        
        # Add minor gridlines
        plt.grid(True, which='minor', alpha=0.15)
        plt.minorticks_on()
        
        # Set background color
        plt.gca().set_facecolor('#f8f9fa')
        
        save_path = self.plot_dir / 'memory_usage.pdf'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved memory usage plot to {save_path}")
    
    def generate_latex_tables(self, df):
        """Generate LaTeX tables for the paper"""
        # Overall performance table
        performance_table = self._create_performance_table(df)
        
        # Noise impact table
        noise_impact_table = self._create_noise_impact_table(df)
        
        # Resource usage table
        resource_table = self._create_resource_table(df)
        
        # Save tables
        self._save_latex_table(performance_table, 'performance_table.tex')
        self._save_latex_table(noise_impact_table, 'noise_impact_table.tex')
        self._save_latex_table(resource_table, 'resource_table.tex')
    
    def _create_performance_table(self, df):
        """Create overall performance comparison table"""
        # Get baseline performance (noise_level = 0)
        baseline_df = df[df['noise_level'] == 0].groupby('model_type').agg({
            'accuracy': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1_score': ['mean', 'std']
        }).round(4)
        
        # Create LaTeX table
        latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Baseline Model Performance (No Noise)}
\\begin{tabular}{lcccc}
\\hline
Model & Accuracy & Precision & Recall & F1-Score \\\\
\\hline
"""
        
        for model in baseline_df.index:
            row = [
                model.upper(),
                f"{baseline_df.loc[model, ('accuracy', 'mean')]:0.4f} $\\pm$ {baseline_df.loc[model, ('accuracy', 'std')]:0.4f}",
                f"{baseline_df.loc[model, ('precision', 'mean')]:0.4f} $\\pm$ {baseline_df.loc[model, ('precision', 'std')]:0.4f}",
                f"{baseline_df.loc[model, ('recall', 'mean')]:0.4f} $\\pm$ {baseline_df.loc[model, ('recall', 'std')]:0.4f}",
                f"{baseline_df.loc[model, ('f1_score', 'mean')]:0.4f} $\\pm$ {baseline_df.loc[model, ('f1_score', 'std')]:0.4f}"
            ]
            latex_table += " & ".join(row) + " \\\\\n"
        
        latex_table += """\\hline
\\end{tabular}
\\label{tab:baseline_performance}
\\end{table}
"""
        return latex_table
    
    def _create_noise_impact_table(self, df):
        """Create table showing accuracy degradation with noise"""
        pivoted_df = df.pivot_table(
            index='noise_level',
            columns='model_type',
            values='accuracy',
            aggfunc='mean'
        ).round(4)
        
        latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Model Accuracy Under Different Noise Levels}
\\begin{tabular}{l""" + "c" * len(pivoted_df.columns) + "}\n\\hline\n"
        
        # Header
        latex_table += "Noise Level & " + " & ".join(col.upper() for col in pivoted_df.columns) + " \\\\\n\\hline\n"
        
        # Data rows
        for idx, row in pivoted_df.iterrows():
            latex_table += f"{idx:0.1f} & " + " & ".join(f"{val:0.4f}" for val in row) + " \\\\\n"
        
        latex_table += """\\hline
\\end{tabular}
\\label{tab:noise_impact}
\\end{table}
"""
        return latex_table
    
    def _create_resource_table(self, df):
        """Create table comparing computational resources"""
        resource_df = df.groupby('model_type').agg({
            'training_time': ['mean', 'std'],
            'inference_time': ['mean', 'std'],
            'memory_usage': ['mean', 'std']
        }).round(4)
        
        latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Computational Resource Usage by Model}
\\begin{tabular}{lccc}
\\hline
Model & Training Time (s) & Inference Time (ns/sample) & Memory Usage (MB) \\\\
\\hline
"""
        
        for model in resource_df.index:
            row = [
                model.upper(),
                f"{resource_df.loc[model, ('training_time', 'mean')]:0.2f} $\\pm$ {resource_df.loc[model, ('training_time', 'std')]:0.2f}",
                f"{resource_df.loc[model, ('inference_time', 'mean')]:0.2f} $\\pm$ {resource_df.loc[model, ('inference_time', 'std')]:0.2f}",
                f"{resource_df.loc[model, ('memory_usage', 'mean')]:0.2f} $\\pm$ {resource_df.loc[model, ('memory_usage', 'std')]:0.2f}"
            ]
            latex_table += " & ".join(row) + " \\\\\n"
        
        latex_table += """\\hline
\\end{tabular}
\\label{tab:resource_usage}
\\end{table}
"""
        return latex_table
    
    def _save_latex_table(self, table_content, filename):
        """Save LaTeX table to file"""
        save_path = self.tables_dir / filename
        with open(save_path, 'w') as f:
            f.write(table_content)
        print(f"Saved LaTeX table to {save_path}")