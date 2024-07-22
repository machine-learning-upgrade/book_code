from fastapi import Depends, FastAPI
import httpx
import pickle # For loading the scikit-learn model
import comet_ml
from functools import lru_cache
from typing_extensions import Annotated

from . import config

app = FastAPI()
model = None # Placeholder for the loaded model

# See earlier chapters for guidance around defining environment variables
@lru_cache
def get_settings():
    return config.Settings()

@app.on_event("startup")
async def startup_event(settings: Annotated[config.Settings, Depends(get_settings)]):
    global model
    comet_ml.init(api_key=settings.comet_api_key)
    experiment = comet_ml.APIExperiment()
    model_artifact = experiment.get_artifact('baseline-housing-model')
    model_artifact.download('./')
    with open('./baseline.pkl', 'rb') as f:
        model = pickle.load(f)

from fastapi import HTTPException
from pydantic import BaseModel
import numpy as np
class PredictionInput(BaseModel): 
    features: list # Assuming a simple list of features

@app.post("/prediction/")
async def make_prediction(input_data: PredictionInput):
    try:
        prediction = model.predict([input_data.features])
        return {"prediction": prediction.tolist()} # Convert numpy array to list
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))