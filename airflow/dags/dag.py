import logging
import json
from datetime import datetime
import pendulum

from airflow.decorators import dag, task

from project_ecommerce import MODEL_PATH, PARAMS_PATH, PROCESSED_PATH, RAW_PATH

logger = logging.getLogger("airflow.task")

# Tasks
@task()
def extract() -> dict:
    from project_ecommerce.extract import extract_db


    try:
        clicks_df, products_df = extract_db(conn_id = "aiven_rec_db")

        # Save data
        clicks_df.to_parquet(RAW_PATH / "clicks.parquet")
        products_df.to_parquet(RAW_PATH / "products.parquet")
        logger.info(f"Data saved to {RAW_PATH}.")
    except Exception as e:
        logger.error(e)
        logger.warning("Data extraction failed. Continue with existing data.")
    return

@task()
def transform() -> None:
    from rec_engine.utils import create_dataset
    from project_ecommerce.utils import prepare_data


    try:
        logger.info("Preparing data")
        clicks_df, products_df, _ = prepare_data()

        # Create TensorFlow dataset
        logger.info("Creating TF datasets.")
        clicks_ds = create_dataset(clicks_df, clicks_df.columns)
        products_ds = create_dataset(products_df, products_df.columns)

        # Save datasets
        logger.info(f"Datasets saved to {PROCESSED_PATH}.")
        clicks_ds.save(str(PROCESSED_PATH / "clicks"))
        products_ds.save(str(PROCESSED_PATH / "products"))
    except Exception as e:
        logger.error(e)
        logger.warning("Data transformation failed. Continue with existing processed data.")
    return

@task()
def training() -> None:
    import tensorflow as tf
    import tensorflow_recommenders as tfrs

    from rec_engine.model import RecommenderEngineModel
    from rec_engine.inference import RecommendationEngine
    from rec_engine.train_functions import train_model


    try:
        # Load datasets and parameters
        logger.info(f"Loading datasets from {PROCESSED_PATH}.")
        clicks_ds = tf.data.Dataset.load(str(PROCESSED_PATH / "clicks"))
        products_ds = tf.data.Dataset.load(str(PROCESSED_PATH / "products"))
        with open(PARAMS_PATH) as f:
            parameters = json.load(f)

        # Filter dataset by selected features to train the model
        query_features = [f.split("-")[1] if "-" in f else f for f in parameters["tower"]["query"]]
        candidate_features = [f.split("-")[1] if "-" in f else f for f in parameters["tower"]["candidate"]]
        query_ds = clicks_ds.map(lambda x: {k: v for k, v in x.items() if k in query_features + candidate_features})
        # Add `id` column for retrieval index
        candidate_ds = products_ds.map(lambda x: {k: v for k, v in x.items() if k in candidate_features + ["id"]})

        model = RecommenderEngineModel(
            params = parameters,
            candidates = candidate_ds,
            queries = query_ds,
            preprocessing = True # Encapsulate preprocessing steps with the model
        )

        # Train model
        logger.info("Fitting model.")
        learning_rates = [1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-3]
        # Set the number of epochs
        parameters["model"]["max_epochs"] = len(learning_rates)
        model = train_model(
            model = model,
            train = query_ds,
            params = parameters,
            learning_rates= learning_rates,
            logging = False
        )

        ## Retrieval Index
        # I use the *brute-force* index because it is the most accurate, although the slowest.
        # In my use case, it performs efficiently due to the small size of the dataset (~10,000 candidates).
        # However, as the number of candidates scales up, *ScaNN* is a recommended alternative for its balance between speed and accuracy.
        query_model = tf.keras.Sequential([model.query_prep_layer, model.query_embedding_layer, model.query_model_layer])
        candidate_model = tf.keras.Sequential([model.candidate_prep_layer, model.candidate_embedding_layer, model.candidate_model_layer])
        index = tfrs.layers.factorized_top_k.BruteForce(query_model, k= 1000) # retrieve 1000 products
        index.index_from_dataset(
            tf.data.Dataset.zip((
                candidate_ds.map(lambda x: x["id"]).batch(1024),
                candidate_ds.batch(1024).map(candidate_model)
            ))
        )
        logger.info("Index created.")
        # Sample query input
        query_input = {
            "user_id": "new_user",
            "channel": "Organic",
            "device_type": "Desktop",
            "query_text": "pie",
            "seq_category_name": ["0"] * 5,
            "time":datetime.strptime("2023-11-02 08:40:19", "%Y-%m-%d %H:%M:%S")
        }

        # Build the model through inference
        logger.info("Building model through inference.")
        recommender = RecommendationEngine(
            index = index,
            model = model,
            candidates = candidate_ds,
            candidate_id= "id"
        )
        _ = recommender(query_input)

        # Save model
        # Save retrieval index
        index.save(MODEL_PATH / 'retrieval_index')
        # Save model for ranking
        model.retrieval_task = tfrs.tasks.Retrieval() # Remove metrics to avoid error
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=parameters['model']['initial_learning_rate'])
        model.compile(optimizer=optimizer)
        model.save(MODEL_PATH / 'ranking_model')
        logger.info(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        logger.error(e)
        logger.error("Model training failed. Continue with existing model.")
    return


@dag(
    dag_id="recommendation_engine_training",
    schedule="@daily",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["recommendation", "etl", "model", "train"],
)
def recommendation_engine_training() -> None:
    extract() >> transform() >> training()

recommendation_engine_training()
