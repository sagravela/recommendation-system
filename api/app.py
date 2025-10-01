from pathlib import Path
from datetime import datetime

import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from rec_engine.inference import RecommendationEngine


app = FastAPI()

# Load the model
index = tf.keras.models.load_model(str(Path("model") / "retrieval_index"), compile = False)
model = tf.keras.models.load_model(str(Path("model") / "ranking_model"), compile = False)
products = tf.data.Dataset.load("products")

# Combine `product_id` and `merchant_name` as the id
recommender = RecommendationEngine(
    index=index,
    model=model,
    candidates=products,
    candidate_id="id"
)

class UserQuery(BaseModel):
    user_id: str
    channel: str
    device_type: str
    query_text: str
    seq_category_name: list = ["0"] * 5 # Placeholder for new users
    time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@app.post("/search/")
async def get_recommendations(query: UserQuery):
    query_dict = query.model_dump()
    recommendations = recommender(query_dict)
    return {"recommendations": recommendations}
