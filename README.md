# Real Time Machine Learning Prediction

A basic implementation of a REST API for on demand prediction.  
The project include: 

 - REST service build with the FastAPI framework.
 - ML model chaching to improve response time.  
 - Health check endpoint.
 
## Project Structure

> |--main.py  
> |--scripts  
>      |--training.py  
> |--utils  
>      |--helpers.py  
> |--layouts  
>      |--input_layout.py  

 - main.py is the entry point of the REST API app.
 - training.py is called to train the ML pipeline.
 - input_layout.py defines a model in Pydantic.  
 - helpers.py is 

## How to use

To make prediction you need to first train a ML estimator running the training.py script;
then you can run the server with FastAPI and make predictions at 127.0.0.1/docs.
