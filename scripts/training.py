import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

from os import path, makedirs, getcwd
import logging.config
from datetime import datetime
from joblib import dump


logging.config.fileConfig('logging.conf', disable_existing_loggers = False)
logger = logging.getLogger(__name__)

SEED = 7
np.random.seed(SEED)

#Path where the trained estimator are stored.
ML_PIPELINES_PATH = 'ml_pipelines/'

def get_data():
	#Load the diabetes dataset from scikit-learn
	diabetes = load_diabetes(scaled = False)
	X, y, features = diabetes.data, diabetes.target, diabetes.feature_names
	df = pd.DataFrame(data = X, columns = features)
	
	return df, y

def build_pipeline(num_features: list, cat_features: list) -> Pipeline:
	
	num_pipeline = Pipeline([
		('scaler', StandardScaler())
	]) 
	cat_pipeline = Pipeline([
		('ohe', OneHotEncoder(handle_unknown = 'ignore'))
	])
	
	preprocessor_pipeline = ColumnTransformer([
		('num', num_pipeline, num_features),
		('cat', cat_pipeline, cat_features)
	])
	
	full_pipeline = Pipeline([
		('preprocessor', preprocessor_pipeline),
		('model', Ridge(random_state = SEED))
	])
	
	return full_pipeline
	
def train() -> Pipeline:
	#Train and return a ML estimator
	df, y = get_data()
	
	num_features = ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
	cat_features = ['sex']
	logger.info(f'Training data shape: {df.shape}')
	
	estimator_pipeline = build_pipeline(num_features = num_features, cat_features = cat_features)
	logger.info('Machine Learning pipeline built.')
	
	estimator_pipeline.fit(df, y)
	logger.info('Machine Learning pipeline trained.')
	
	return estimator_pipeline
	
if __name__ == '__main__':
	
	estimator_pipeline = train()
	new_version = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
	#Save the trained estimator to ML_PIPELINES_PATH
	if not path.exists(ML_PIPELINES_PATH) : makedirs(ML_PIPELINES_PATH)
	dump(estimator_pipeline, path.join(ML_PIPELINES_PATH, f'ml_pipeline.pickle'))
	logger.info('New pipeline saved successfully.')
