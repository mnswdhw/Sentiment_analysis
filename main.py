from fastapi import FastAPI
from inference import *


app = FastAPI()

@app.get("/")
async def root(review):

    infer = Inference("/home/manas/Desktop/Job Assessments/truefoundry/sa.pth")
    sentiment_pred = infer.infer(review)

    return {"Predicted sentiment": sentiment_pred}


