# Jane Street Real-Time Market Data Forecasting

![Competition](https://img.shields.io/badge/Competition-Kaggle-20BEFF?style=flat-square&logo=kaggle)
![Status](https://img.shields.io/badge/Status-Working-brightgreen?style=flat-square)
![ML](https://img.shields.io/badge/ML-Ensemble-orange?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=flat-square&logo=pytorch)
![Data](https://img.shields.io/badge/Data-Financial-blue?style=flat-square)
![Models](https://img.shields.io/badge/Models-4-purple?style=flat-square)

This project is a solution for the **Kaggle 2024 Jane Street Real-Time Market Data Forecasting competition**. It uses anonymized financial market data from Jane Street's real production systems to build machine learning models for predicting market responses.Link here: [https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)

## Project Structure

```
├── datasets/
│   ├── data_descripton.md     # Detailed description of dataset fields and files
│   └── download_data.sh       # Script to download competition data
├── data(creat the dir and save the data loaded from `datasets/download_data.sh` to run the scripts)/
│   ├── raw/                          # Original competition data
│   │   └── jane-street-real-time-market-data-forecasting/
│   │       ├── features.csv          # Feature descriptions
│   │       ├── responders.csv        # Target variable descriptions
│   │       ├── train.parquet/        # Training data (partitioned)
│   │       ├── test.parquet/         # Test data
│   │       └── lags.parquet/         # Lag data
│   └── derived/                      # Processed data
│       ├── sampled/                  # Sampled datasets for development
│       └── with_lags/               # Data with lagged features
├── source/
│   ├── config.json                   # Project configuration file
│   ├── EDA/
│   │   ├── EDA_Features.ipynb        # Feature correlation analysis
│   │   └── EDA_Traning.ipynb         # Training data exploration
│   ├── training/                     # Model training modules
│   │   ├── nn_train.ipynb            # Neural network training
│   │   ├── ridge_train.ipynb         # Ridge regression training
│   │   ├── xgb_train.ipynb           # XGBoost training with GPU support
│   │   ├── tabm_train.ipynb          # TabNet training
│   │   ├── ensemble.ipynb            # Model ensemble and final submission
│   │   ├── tanm_reference.py         # TabNet model reference implementation
│   │   └── README.md                 # Training module documentation
│   └── utils/
│       ├── sampling.py               # Data sampling utilities
│       ├── create_lag.py             # Create lagged responder features
│       ├── reduce_memory_usage.py    # Memory optimization utilities
│       └── README.md                 # Utility usage instructions
├── models/                           # Saved model files
├── outputs/                          # Training outputs and results
├── graphs/                           # Visualization outputs
├── temp/                             # Temporary files
├── LICENSE                           # License file
└── README.md                         # Project overview
```

## Workflow

### 1. Data Exploration and Preprocessing
- Download competition data from Kaggle and place in `data/raw/`
- Use `EDA/EDA_Features.ipynb` to analyze feature correlations and patterns
- Use `EDA/EDA_Traning.ipynb` to explore training data distributions and responder variables
- Perform data sampling through `utils/sampling.py` for rapid development on limited hardware
- Create lagged features using `utils/create_lag.py` to enhance temporal information
- Apply memory optimization with `utils/reduce_memory_usage.py` for large datasets

### 2. Multi-Model Training
- **Neural Network**: Multi-layer perceptron with cross-validation (`nn_train.ipynb`)
- **Ridge Regression**: Linear baseline model with fast training (`ridge_train.ipynb`)
- **XGBoost**: Gradient boosting trees with GPU acceleration (`xgb_train.ipynb`)
- **TabNet**: Attention-based neural network for tabular data (`tabm_train.ipynb`)

### 3. Model Ensemble
Use `ensemble.ipynb` to perform weighted fusion of all models:
- NN+XGBoost ensemble (70%)
- Ridge regression (10%)  
- TabNet (40%)

## Hardware Requirements

- **Sampled**: 16GB RAM, LGBM training
- **Recommended**: 64GB+ RAM, at least 1 NVIDIA GPU for full data processing and model training

---
