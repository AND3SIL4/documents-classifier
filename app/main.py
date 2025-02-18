from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = FastAPI()


@app.get("/")
def root():
    time = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
    return JSONResponse(
        status_code=200,
        content={
            "status": "working",
            "time": time,
            "version": "0.0.1",
            "author": "Felipe Silva"
        }
    )
