# Data Sampling Tool

## Files

- `sampling.py` - Main sampling script

## Quick Start

```bash
cd source/utils
python sampling.py
```

This will:
- Sample 10% of data from all partitions (0-9)
- Save individual partition samples to `data/derived/sampled/`
- Merge all samples into a single file

## Functions

### sample_data(sample_ratio=0.1, random_seed=42)
- Processes partitions 0-9 from train.parquet
- Samples specified ratio from each partition
- Saves to individual files: `partition_X_sample.parquet`

### merge_samples()
- Reads all partition sample files
- Combines into single file: `train_sample_merged.parquet`

## Output

All files saved to: `data/derived/sampled/`