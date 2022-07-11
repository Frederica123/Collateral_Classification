from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import joblib
import json
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
nlp = model.encode
 

#Read the code_description_json file
with open("config.json","r") as f:
    config = json.load(f)
    
config = config[0]

loaded_rf = joblib.load("model_joblibs/" + config['large_joblib'])
# original_df = pd.read_csv('before_smote_standard_token_from_6.csv',index_col = 0)
description = pd.read_json(config['description_large'], encoding = 'utf-8-sig')

def large_predict(test_data):
    # only deal with one input
    input_ = nlp(test_data['collateral_text']).reshape(1,-1)
    code = loaded_rf.predict(input_)[0]
    des = description[description['Collateral Code']==code].reset_index()['Collateral Code Description'][0]
    probability = max(loaded_rf.predict_proba(input_)[0])
    p_dict = {}
    loaded_rf.predict(input_)[0]
    classes = []
    for i, j in zip(description['Collateral Code Description'].values, loaded_rf.predict_proba(input_)[0]):
        p_dict[i] = j
    #return({'response':test_data})
    return  ({'collateral_code':str(code),
            'collateral_code_description':des,
            'probability_prediction':str(probability),
             'probability_dict': p_dict})
