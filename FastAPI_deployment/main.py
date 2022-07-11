from model_files.randomforest_smote import rf_predict
from model_files.large import large_predict
from model_files.medium import medium_predict
from model_files.small import small_predict
from pydantic import BaseModel
from fastapi import FastAPI

tags_metadata = [
    {
        "name": "Random Forest Based on SMOTE",
        "description": "Using random forest model based on SMOTE to predict 19 codes which have more than 100 records.These codes are 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 14, 15, 16, 17, 18, 24, 28, 34.",
    },
    {
        "name": "Large Forest For Codes with Large Samples",
        "description": "Random Forest predicting codes with more than 15k recordes which are 1, 5, 8, 10."
    },
    {
        "name": "SVM For Codes with Medium Samples",
        "description": "Support vector machine predicting codes with more than 1k recordes which are 2, 5, 17, 18, 24, 28."
    },
    {
        "name": "SVM For Codes with Small Samples",
        "description": "Support vector machine predicting codes with more than 100 recordes which are 3, 4, 6, 7, 9, 12, 14, 16, 34."
    },
      
]

app = FastAPI(openapi_tags=tags_metadata)



# app = FastAPI()


class request_body(BaseModel):
    collateral_text:str


@app.post('/predict_randomforest_19_codes',tags=['Random Forest Based on SMOTE'])
def randomforest_predict(data : request_body):
    # only deal with one input
    test_data = data.dict()
    return(rf_predict(test_data))

@app.post('/predict_largemodel', tags=['Large Forest For Codes with Large Samples'])
def largemodel_predict(data : request_body):
    # only deal with one input
    test_data = data.dict()
    return(large_predict(test_data))

@app.post('/predict_mediummodel', tags = ['SVM For Codes with Medium Samples'])
def mediummodel_predict(data : request_body):
    # only deal with one input
    test_data = data.dict()
    return(medium_predict(test_data))

@app.post('/predict_smallmodel', tags=['SVM For Codes with Small Samples'])
def smallmodel_predict(data : request_body):
    # only deal with one input
    test_data = data.dict()
    return(small_predict(test_data))

    