import os
from joblib import load
import logging.config
from typing import Dict, Any

from fastapi import FastAPI, status, Response
from fastapi.responses import JSONResponse

from utils.helpers import convert_input_to_df

from layouts.input_layout import Input
from scripts.training import ML_PIPELINES_PATH



logging.config.fileConfig('logging.conf', disable_existing_loggers = False)
logger = logging.getLogger(__name__)
logger.propagate = False

CACHE: Dict[str, Any] = {}

app = FastAPI()

@app.get('/health')
async def health_check():

	content = {'ML pipeline' : None}
	
	#Check the presence of a trained estimator
	try:		
		if os.path.isfile(os.path.join(ML_PIPELINES_PATH, f'ml_pipeline.pickle')):
			content['ML pipeline'] = 'Ok'
		else:
			content['ML pipeline'] = 'ML pipeline unavailable'
	except Exception:
		content['ML pipeline'] = 'ML pipeline unavailable'
	
	return JSONResponse(content = content)

@app.post('/predict')
async def predict(input_data: Input, response: Response):
	
	#Use a trained estimator for predictions if it exists, otherwise load it and cache it.
	if CACHE.get('ml_pipeline', None) is None:
		
		try:
			
			ml_pipeline = load(os.path.join(ML_PIPELINES_PATH, 'ml_pipeline.pickle'))
			CACHE['ml_pipeline'] = ml_pipeline

			
		except FileNotFoundError:
			
			logger.error(f'The ML pipeline does not exists')
			
			response.status_code = status.HTTP_404_NOT_FOUND
			return JSONResponse(content = {'Status' : 'The model was not found.'},
								status_code = status.HTTP_404_NOT_FOUND)
								
	#compute prediction
	df = convert_input_to_df(input_data)
	prediction = CACHE['ml_pipeline'].predict(df)[0]

	return JSONResponse(content = {'prediction' : prediction})
