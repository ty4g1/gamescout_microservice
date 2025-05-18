from fastapi import FastAPI
from app.models.data import Data
from app.services.vectorize import feature_vector

app = FastAPI()

@app.get("/vectorize")
async def create_item(data: Data):
    return {"vector": feature_vector(data).tolist()}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}