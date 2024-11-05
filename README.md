# BHICS_IDS: Model Selection for BHICS Project

## Overview
This repository contains the machine learning model selection and evaluation experiments for the BHICS (Blockchain-enabled Honeypot IoT Conversion System) project. We evaluate and compare different ML models to identify the most suitable intrusion detection model for resource-constrained IoT environments.

## Models Evaluated
- XGBoost (XGB)
- Random Forest (RF)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

## Key Experiments
- Baseline performance comparison
- Noise resilience testing (Gaussian noise levels σ² from 0.1 to 4.0)
- Resource utilization analysis (memory, training time, inference speed)

## Project Structure
```
BHICS_IDS/
├── data/
│   └── prepared_data/        # Preprocessed datasets
├── experiment_results/
│   ├── plots/               # Performance visualization plots
│   └── tables/              # LaTeX tables for paper
├── scripts/
│   ├── noise_experiment.py  # Main experiment script
│   └── generate_visuals.py  # Plot and table generation
├── src/
│   ├── data/               # Data loading utilities
│   ├── ids/               # Model implementations
│   └── metrics/           # Performance tracking
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation
```bash
git clone https://github.com/dcommey/BHICS_IDS.git
cd BHICS_IDS
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
1. Run model evaluation experiments:
```bash
python scripts/noise_experiment.py
```

2. Generate result visualizations:
```bash
python scripts/generate_visuals.py
```

## Key Findings
Our experiments demonstrate XGBoost's superior performance as the optimal model for BHICS:

| Model   | Baseline Accuracy | Training Time (s) | Inference Time (ms) |
|---------|------------------|-------------------|-------------------|
| XGBoost | 99.59%           | 0.18              | 0.246             |
| RF      | 99.57%           | 7.40              | 6.135             |
| CNN     | 99.55%           | 92.72             | 42.329            |
| LSTM    | 99.55%           | 150.14            | 87.125            |

## License
MIT License

## Contact
Daniel Commey - dcommey@tamu.edu

Project Link: https://github.com/dcommey/BHICS_IDS

## Related Projects
- [BHICS](https://github.com/dcommey/BHICS) - Main project repository implementing the complete Blockchain-enabled Honeypot IoT Conversion System