# File and Field Information

• **train.parquet** - The training set, contains historical data and returns. For convenience, the training set has been partitioned into ten parts.
  ° `date_id` and `time_id` - Integer values that are ordinally sorted, providing a chronological structure to the data, although the actual time intervals between `time_id` values may vary.
  ° `symbol_id` - Identifies a unique financial instrument.
  ° `weight` - The weighting used for calculating the scoring function.
  ° `feature_{00...78}` - Anonymized market data.
  ° `responder_{0...8}` - Anonymized responders clipped between -5 and 5. The `responder_6` field is what you are trying to predict.

• **test.parquet** - A mock test set which represents the structure of the unseen test set. This example set demonstrates a single batch served by the evaluation API, that is, data from a single `date_id`, `time_id` pair. The test set contains columns including `date_id`, `time_id`, `symbol_id`, `weight`, `is_scored`, and `feature_{00...78}`. *You will not be directly using the test set or sample submission in this competition, as the evaluation API will get/set the test set and predictions.*
  ° `is_scored` - Indicates whether this row is included in the evaluation metric calculation.

• **lags.parquet** - Values of `responder_{0...8}` lagged by one `date_id`. The evaluation API serves the entirety of the lagged responders for a `date_id` on that `date_id`'s first `time_id`. In other words, all of the previous date's responders will be served at the first time step of the succeeding date.

• **sample_submission.csv** - This file illustrates the format of the predictions your model should make.

• **features.csv** - metadata pertaining to the anonymized features

• **responders.csv** - metadata pertaining to the anonymized responders