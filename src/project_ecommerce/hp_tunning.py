import argparse
import json
from rec_engine import log
from rec_engine.utils import split_and_preprocess
from rec_engine.objective import optimize

from . import OPTUNA_PATH
from .utils import prepare_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for recommendation engine."
    )
    parser.add_argument(
        "-n", "--n_trials", type=int, default=100, help="Number of trials."
    )
    args = parser.parse_args()

    # Load data
    clicks_df, products_df, params = prepare_data()

    # Load optuna config
    with open(OPTUNA_PATH / "opt_config.json") as f:
        opt_config = json.load(f)
    log.info(f"Loaded optuna config from {OPTUNA_PATH / 'opt_config.json'}")

    # Preprocess data
    prep_clicks_train, prep_clicks_val, prep_products, feature_dim = (
        split_and_preprocess(clicks_df, products_df, params, shuffle=True, seed=params["seed"])
    )

    # Hyperparemeter tunning
    optimize(
        n_trials = args.n_trials,
        opt_config = opt_config,
        params = params,
        train = prep_clicks_train,
        val = prep_clicks_val,
        candidates = prep_products,
        feature_dim = feature_dim
    )
