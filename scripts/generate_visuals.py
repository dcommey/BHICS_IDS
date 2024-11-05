# scripts/generate_visuals.py

import os
from pathlib import Path
from src.metrics.visualization_utils import ResultsVisualizer

def main():
    # Find the most recent experiment results
    output_dir = Path("experiment_results")
    metrics_files = list(output_dir.glob("experiment_*_metrics.csv"))
    
    if not metrics_files:
        print("No experiment results found!")
        return
    
    # Get the most recent file
    latest_metrics_file = max(metrics_files, key=lambda x: os.path.getctime(x))
    print(f"Processing results from: {latest_metrics_file}")
    
    # Create visualizer
    visualizer = ResultsVisualizer(output_dir)
    
    # Load and process results
    results_df = visualizer.load_results(latest_metrics_file)
    
    # Create visualizations
    print("Creating plots...")
    visualizer.create_noise_impact_plots(results_df)
    visualizer.create_timing_comparison(results_df)
    visualizer.create_memory_usage_plot(results_df)
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    visualizer.generate_latex_tables(results_df)
    
    print(f"\nVisualization results saved in {output_dir}/plots")
    print(f"LaTeX tables saved in {output_dir}/tables")

if __name__ == "__main__":
    main()