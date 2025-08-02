from datetime import datetime

import tensorflow as tf
import pandas as pd
from rec_engine import log
from rec_engine.utils import split_and_preprocess
from rec_engine.model import RecommenderEngineModel
from rec_engine.train_functions import train_model

from . import LOGS_PATH
from .utils import prepare_data


BASELINE_PATH = LOGS_PATH / "baseline"

if __name__ == "__main__":
    # Load data
    clicks_df, products_df, params = prepare_data()

    # Set seed for reproducibility
    tf.random.set_seed(params["seed"])

    # Preprocess data
    prep_clicks_train, prep_clicks_val, prep_products, feature_dim = (
        split_and_preprocess(clicks_df, products_df, params, shuffle=False, seed=params["seed"])
    )
    log.info("Data preprocessing completed")

    params["logs_path"] = str(BASELINE_PATH)
    # Create model instance for baseline
    model = RecommenderEngineModel(
        params=params,
        candidates=prep_products,
        preprocessing= False, # Disable preprocessing
        feature_dim= feature_dim, # Feature dimension is a must, if preprocessing is disabled
        train_metrics= True # Enable training metrics
    )
    log.info("Training baseline")
    fitted_model = train_model(
        model= model,
        train= prep_clicks_train,
        val= prep_clicks_val,
        params= params,
        profile= (20, 25), # enable profiling
    )
    results = fitted_model.history.history

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = BASELINE_PATH / f"baseline-{timestamp}.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")
