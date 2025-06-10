# MovieLens Ising Model Analysis

## Project Overview
Movie recommendation using Ising models to learn rating dependencies.

## Key Results
- **76.9% prediction accuracy** (vs 66.0% baseline)
- **16.6% relative improvement** over independent model
- **89.6% precision** in recommendations
- Discovered meaningful movie dependencies (Star Wars franchise, genre clustering)

## Project Structure
```
CS179FinalProject/
├── data/ml-latest-small/     # MovieLens dataset
├── notebooks/
│   ├── ising_movielens_analysis.ipynb    # Main analysis (complete)
│   └── clean_analysis_example.ipynb      # Clean modular version
├── src/
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── ising_model.py         # Ising model implementation
│   ├── evaluation.py          # Prediction evaluation functions
│   └── visualization.py       # Plotting and visualization
└── requirements.txt
```

## Usage

### Main Analysis
Run `notebooks/ising_movielens_analysis.ipynb` for complete analysis.

### Modular Version
Run `notebooks/clean_analysis_example.ipynb` to see organized code structure.

### Dependencies
```bash
pip install -r requirements.txt
pip install pyGMs  # For exact likelihood computation
```

## Key Findings
1. **Structure Learning**: Dense connectivity (40.2 connections/movie) captures user behavior
2. **Meaningful Dependencies**: Star Wars franchise effects, genre clustering, audience segmentation  
3. **Prediction Performance**: Significant improvement over collaborative filtering baselines
4. **Interpretable Results**: Learned patterns match intuitive movie relationships

## Technical Approach
- **Model**: Ising model (pairwise MRF) over binary movie preferences
- **Structure Learning**: L1-regularized logistic regression
- **Evaluation**: Train/test split with multiple baseline comparisons
- **Metrics**: Accuracy, precision, recall, F1-score, statistical significance

## Authors
CS179 Final Project - Graphical Models
