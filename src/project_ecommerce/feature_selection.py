import argparse
from datetime import datetime

import tensorflow as tf
import pandas as pd
from rec_engine import log
from rec_engine.utils import split_and_preprocess
from rec_engine.feature_selection import FeatureSelection

from . import LOGS_PATH
from .utils import prepare_data


FS_PATH = LOGS_PATH / "feature_selection"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run feature selection process for the recommendation system."
    )

    # Load data
    clicks_df, products_df, params = prepare_data()

    # Set seed for reproducibility
    tf.random.set_seed(params["seed"])

    # Preprocess data
    prep_clicks_train, prep_clicks_val, prep_products, feature_dim = (
        split_and_preprocess(clicks_df, products_df, params, shuffle=False)
    )
    log.info("Data preprocessing completed")

    params["logs_path"] = str(FS_PATH)
    results = FeatureSelection(
        train = prep_clicks_train,
        val = prep_clicks_val,
        candidates = prep_products,
        feature_dim = feature_dim,
        params = params,
        baseline_features = {
            "query": ["time", "score", "cat-user_id"],
            "candidate": ["cat-product_id"]
        }
    ).run()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = FS_PATH / f"fs-{timestamp}.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")
