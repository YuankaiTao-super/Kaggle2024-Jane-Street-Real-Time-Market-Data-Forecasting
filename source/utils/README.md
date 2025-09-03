# Utilities for Jane Street Data Processing

## Files

- `sampling.py` - Data sampling script for creating smaller datasets
- `create_lag.py` - Create lagged responder features for training
- `reduce_memory_usage.py` - Memory optimization for pandas DataFrames

### Data Sampling

- Sample 10% of data from all partitions (0-9)
- Save individual partition samples to `data/derived/sampled/`
- Merge all samples into a single file
- **Note**: Suggest to use it to release the pressure on RAM when your device cannot meet the requirements mentioned (64GB + 1GPU)

### Create Lagged Features

- Process all training data partitions
- Create lagged responder features (`responder_0_lag_1` to `responder_8_lag_1`)
- Split data into training and validation sets
- Save processed data to `data/derived/with_lags/`

### Memory Optimization

- Analyze each column's data range
- Convert integers to smallest suitable type (int8, int16, int32, int64)
- Convert floats to optimal precision (float16/float32, float64)
- Display memory usage before and after optimization
- **Reference**: Based on Yuanzhe Zhou's Kaggle notebook: [Jane Street Baseline LGB, XGB and CatBoost](https://www.kaggle.com/code/yuanzhezhou/jane-street-baseline-lgb-xgb-and-catboost/notebook)

## Notes

- The memory optimization script is particularly useful for large datasets
- Memory reduction typically achieves 30-50% savings depending on data characteristics
- All scripts use configurable paths via `config.json`