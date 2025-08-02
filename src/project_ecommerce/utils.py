import pandas as pd
import numpy as np
from rec_engine import log
from rec_engine.utils import create_feature_sequence

from .extract import load_data


def process_clicks(
        clicks_df: pd.DataFrame,
        products_df: pd.DataFrame,
        seq_features: list[str],
        add_to_cart_score: float= 0.5,
        conversion_score: float = 1.0
    ) -> pd.DataFrame:
    """
    Process Clicks Dataset
    In order to do a retrieval task, I need to separate positive interactions from negative ones. **Clicks are assumed as positive interactions**, so those interactions without any click are considered as negative.
    In the other hand, I need a **score feature** to rank products by the feedback received by the user. I will use the following weights to create this feature:

    |Score|Add to Cart|Conversion|
    |----|----|----|
    |`0.0`|No|No|
    |`0.5`|Yes|No|
    |`1.0`|No/Yes|Yes|

    Other steps:
    - Cast `time` to string (demanded by TensorFlow)
    - Join `products_df` with `clicks_df` in one dataset
    - Create sequential features defined in `seq_features`

    Parameters
    ---
    clicks_df : pd.DataFrame
        Clicks dataset
    products_df : pd.DataFrame
        Products dataset
    seq_features : list[str]
        List of sequential features to create
    add_to_cart_score : float, optional
        Add to cart score, by default 0.5
    conversion_score : float, optional
        Conversion score, by default 1.0

    Returns
    ---
    pd.DataFrame
    """
    # Convert `time` to string given that TensorFlow does not support `datetime64` data type.
    clicks_df["time"] = clicks_df["time"].astype(str)
    # Create score feature
    clicks_df["score"] = np.where(clicks_df["conversion"] == True, conversion_score, np.where(clicks_df["add_to_cart"] == True, add_to_cart_score, 0.0))

    # Add candidate related features to the query
    clicks_df = clicks_df.merge(products_df, on= ["product_id", "merchant_id"], suffixes= (None, "_y"))
    clicks_df.drop(clicks_df.columns[clicks_df.columns.str.contains("_y$")], axis= 1, inplace= True)

    # Create sequential features
    for feature in seq_features:
        clicks_df = create_feature_sequence(
            data_df=clicks_df,
            feature=feature,
            agent_id="user_id",
            time_feature="time"
        )

    return clicks_df

def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Pipeline for data preparation. Steps:
    - Data loading
    - Clicks processing:
        - Cast `time` to string
        - Create `score` target variable
        - Create sequential features
        - Merge with products dataset

    Returns
    ---
    tuple[pd.DataFrame, pd.DataFrame, dict]
    """

    # Load data
    clicks_df, products_df, params = load_data()

    log.info("Data preprocessing")
    # Create `score` target variable, sequential features and merge with products dataset
    seq_features = [f.replace("seq-seq_", "") for f in params["tower"]["query"] if f.startswith("seq-")]
    clicks_df = process_clicks(
        clicks_df = clicks_df,
        products_df = products_df,
        seq_features = seq_features
    )

    # Combine `product_id` and `merchant_name` as a single key for identification
    products_df["id"] = products_df.apply(lambda x: f"{x['product_id']}|{x['merchant_name']}", axis=1)
    return clicks_df, products_df, params
