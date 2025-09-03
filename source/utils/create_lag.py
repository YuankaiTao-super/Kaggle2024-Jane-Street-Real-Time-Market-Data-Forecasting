import polars as pl
from pathlib import Path
import json

# Global constants
responders = [f"responder_{idx}" for idx in range(9)]


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def create_lagged_features(df: pl.DataFrame) -> pl.DataFrame:
    df_sorted = df.sort(["date_id", "symbol_id", "time_id"])
    lag_expressions = []
    
    for col in responders:
        lag_col_name = f"{col}_lag_1"
        lag_expr = pl.col(col).shift(1).over(["date_id", "symbol_id"]).alias(lag_col_name)
        lag_expressions.append(lag_expr)
    
    return df_sorted.with_columns(lag_expressions)


def process_training_data(input_path: str = None, output_path: str = None) -> None:
    if input_path is None or output_path is None:
        config = load_config()
        global_path = Path(config["path"]["global_path"])

        input_path = global_path / "data" / "raw" / "jane-street-real-time-market-data-forecasting" / "train.parquet"
        output_path = global_path / "data" / "derived" / "with_lags"
    
    all_partitions = []
    
    # Process each partition (0-9)
    for partition_id in range(10):
        partition_path = Path(input_path) / f"partition_id={partition_id}"
            
        print(f"Processing partition {partition_id}...")
        df = pl.scan_parquet(partition_path).collect()
        df_with_lags = create_lagged_features(df)
        
        # Remove rows with NaN lag values
        lag_col_names = [f"{col}_lag_1" for col in responders]
        df_clean = df_with_lags.filter(
            pl.all_horizontal([pl.col(col).is_not_null() for col in lag_col_names])
        )
        
        all_partitions.append(df_clean)
        print(f"Partition {partition_id}: {len(df)} -> {len(df_clean)} rows")
    
    if all_partitions:
        combined_df = pl.concat(all_partitions)
        unique_dates = sorted(combined_df["date_id"].unique().to_list())
        split_idx = int(len(unique_dates) * 0.8)  # 80/20 train/validation split
        
        train_dates = unique_dates[:split_idx]
        valid_dates = unique_dates[split_idx:]
        
        train_df = combined_df.filter(pl.col("date_id").is_in(train_dates))
        valid_df = combined_df.filter(pl.col("date_id").is_in(valid_dates))
        
        output_path = Path(output_path)
        train_df.write_parquet(output_path / "training.parquet")
        valid_df.write_parquet(output_path / "validation.parquet")
        
        print("DONE")

def create_validation_lags(test_df: pl.DataFrame, lag_data: pl.DataFrame = None) -> pl.DataFrame:
    combined_df = pl.concat([lag_data, test_df])
    df_with_lags = create_lagged_features(combined_df)
    return df_with_lags.tail(len(test_df))

if __name__ == "__main__":
    print("Creating lagged features for Jane Street data...")
    process_training_data()
    print("Completed!")
