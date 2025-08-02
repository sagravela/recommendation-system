from datetime import datetime

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from rec_engine import log
from rec_engine.utils import create_dataset
from rec_engine.experiments import resample_exp

from . import LOGS_PATH
from .utils import prepare_data


EXP_RS_PATH = LOGS_PATH / "exp_resample"

if __name__ == "__main__":
    # Load data
    clicks_df, products_df, params = prepare_data()

    # Set seed for reproducibility
    tf.random.set_seed(params["seed"])

    # Parse products dataframe to dataset
    products_ds = create_dataset(products_df, params["tower"]["candidate"])

    # Split data
    # Drop sequential features from both, the data and params, because resampling methods doesn't accept their data type
    params["tower"]["query"] = [f for f in params["tower"]["query"] if not f.startswith("seq-")]
    clicks_train_df_sh, clicks_val_df_sh = train_test_split(
        clicks_df.loc[:,~clicks_df.columns.str.startswith("seq_")],
        test_size=0.2,
        shuffle=True,
        stratify= clicks_df["score"],
        random_state=params["seed"]
    )
    log.info("Data preprocessing completed")

    params["logs_path"] = str(EXP_RS_PATH)
    results = resample_exp(
        train_df = clicks_train_df_sh,
        val_df = clicks_val_df_sh,
        candidates= products_ds,
        params = params,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = EXP_RS_PATH / f"exp_res-{timestamp}.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")
