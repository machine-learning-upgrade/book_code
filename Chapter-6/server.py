from fastapi import HTTPException, BackgroundTasks 
from pydantic import BaseModel 
import numpy as np 
from csv import DictWriter 

async def log_data(data: dict): 
    with open('./production_data.csv', "a+") as f: 
        d = DictWriter(f, fieldnames=list(data.keys())) 
        d.writerow(data) 

class PredictionInput(BaseModel): 
    features: list  # Assuming a simple list of features 

@app.post("/prediction/") 
async def make_prediction(background_tasks: BackgroundTasks, input_data: PredictionInput): 
    try: 
        prediction = model.predict([input_data.features]) 
        result = {"prediction": prediction.tolist()}  # Convert numpy array to list 
        logged_data =  input_data.dict() 
        logged_data['prediction'] = result['prediction'] 

        # Add the log_data function to background tasks 
        background_tasks.add_task(log_data, data=logged_data) 
    except Exception as e: 
        raise HTTPException(status_code=400, detail=str(e)) 