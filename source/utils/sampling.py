import polars as pl
import json
from pathlib import Path


def sample_data(sample_ratio=0.1, random_seed=42):
    # Load config
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set paths
    global_path = Path(config["path"]["global_path"])
    raw_data_path = global_path / "data" / "raw" / "jane-street-real-time-market-data-forecasting"
    output_path = global_path / "data" / "derived" / "sampled"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process partitions 0-9
    for partition_id in range(10):
        input_path = raw_data_path / "train.parquet" / f"partition_id={partition_id}"
        
        # Load data first, then sample
        df = pl.read_parquet(input_path)  # Changed from scan_parquet to read_parquet
        total_rows = len(df)
        sample_size = int(total_rows * sample_ratio)
        
        sampled_df = df.sample(n=sample_size, seed=random_seed)
        
        # Save
        output_file = output_path / f"partition_{partition_id}_sample.parquet"
        sampled_df.write_parquet(output_file)
        
        print(f"Partition {partition_id}: {total_rows:,} -> {len(sampled_df):,} rows")


def merge_samples():
    # Load config
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    global_path = Path(config["path"]["global_path"])
    output_path = global_path / "data" / "derived" / "sampled"
    
    # Read all sample files
    dfs = []
    for partition_id in range(10):
        file_path = output_path / f"partition_{partition_id}_sample.parquet"
        df = pl.read_parquet(file_path)
        dfs.append(df)
    
    # Merge and save
    merged_df = pl.concat(dfs)
    merged_file = output_path / "train_sample_merged.parquet"
    merged_df.write_parquet(merged_file)
    
    print(f"Merge completed: {len(merged_df):,} rows")


if __name__ == "__main__":
    # Sample 10% of data
    sample_data(sample_ratio=0.1)
    
    # Merge samples
    merge_samples()