# Jane Street Real-Time Market Data Forecasting

![Competition](https://img.shields.io/badge/Competition-Kaggle-20BEFF?style=flat-square&logo=kaggle)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)
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
├── source/
│   ├── config.json            # Project configuration file
│   ├── EDA/
│   │   └── test.ipynb         # Exploratory data analysis
│   ├── training/              # Model training modules
│   │   ├── nn_train.ipynb     # Neural network training
│   │   ├── ridge_train.ipynb  # Ridge regression training
│   │   ├── xgb_train.ipynb    # XGBoost training
│   │   ├── tabm_train.ipynb   # TabNet training
│   │   ├── ensemble.ipynb     # Model ensemble and final submission
│   │   └── tanm_reference.py  # TabNet model reference implementation
│   └── utils/
│       ├── sampling.py        # Data sampling utilities
│       └── README.md          # Utility usage instructions
├── LICENSE                    # License file
└── README.md                  # Project overview
```

## Workflow

### 1. Data Exploration and Preprocessing
- Review `datasets/data_descripton.md` to understand the dataset structure and field definitions
- Use `datasets/download_data.sh` to download the competition data from Kaggle
- Use `EDA/test.ipynb` to explore the patterns of data including 79 anonymized features and target variables
- Perform data sampling through `utils/sampling.py` if you don't have enough RAM with your local device (require around 64GB RAM and at least 1 GPU for the original task) for rapid development training of light models
- Create lagged features to enhance temporal information

### 2. Multi-Model Training
- **Neural Network**: Multi-layer perceptron with 5-fold cross-validation
- **Ridge Regression**: Linear baseline model with fast training
- **XGBoost**: Gradient boosting trees with GPU acceleration
- **TabNet**: Attention-based neural network specialized for tabular data

### 3. Model Ensemble
Use `ensemble.ipynb` to perform weighted fusion of all models:
- NN+XGBoost ensemble (70%)
- Ridge regression (10%)  
- TabNet (40%)

### 4. Real-time Inference
Use Kaggle's official inference server for streaming predictions (as required by the host), supporting real-time market data processing.

## Model Evaluation

Uses weighted R² as the primary evaluation metric, employing time series cross-validation to ensure stable model performance on financial time series data. Models complement each other through ensemble strategies to improve overall prediction performance.

---
