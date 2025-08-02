import argparse
import json
from datetime import datetime

import tensorflow as tf
from rec_engine import log
from rec_engine.utils import create_dataset
from rec_engine.experiments import deep_layers_exp

from . import LOGS_PATH
from .utils import prepare_data


EXP_DL_PATH = LOGS_PATH / "exp_deep_layers"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run deep layers experiment for the recommendation system."
    )
    parser.add_argument(
        "deep_layers",
        type=json.loads,
        help="JSON string with parameters for deep layers experiment. "
             "Example: '[[64, 32], [128, 64]]' "
    )
    args = parser.parse_args()

    # Load data
    clicks_df, products_df, params = prepare_data()

    # Set seed for reproducibility
    tf.random.set_seed(params["seed"])

    # Parse products dataframe to dataset
    products_ds = create_dataset(products_df, params["tower"]["candidate"])
    log.info("Data preprocessing completed")

    params["logs_path"] = str(EXP_DL_PATH)
    # I won't use sequential features for this analysis in order to apply CV
    params["tower"]["query"] = [f for f in params["tower"]["query"] if not f.startswith("seq-")]
    # Because I am applying cross-validation, I can't use sequential features
    results = deep_layers_exp(
        candidates = products_ds,
        data_df = clicks_df,
        params = params,
        deep_layers = args.deep_layers
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = EXP_DL_PATH / f"exp_dl-{timestamp}.csv"
    results.to_csv(results_path, index="run_number")
    log.info(f"Results saved to {results_path}")
