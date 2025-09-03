# Training Instructions for Jane Street Real-Time Market Data Forecasting

## 1. Download Competition Data
Download the dataset from Kaggle:
[Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data)

## 2. Train Individual Models
Run the following notebooks to train models and obtain their weights:
- `nn_train.ipynb`
- `ridge_train.ipynb`
- `xgb_train.ipynb`
- `tabm_train.ipynb`

## 3. Ensemble Models
Run the `ensemble.ipynb` notebook to combine the trained models.

## Hardware Requirements
- At least one GPU
- Minimum 64GB RAM
