# Real Time Machine Learning Prediction

A basic implementation of a REST API for on demand prediction.  
The project include: 

 - REST service build with the FastAPI framework.
 - ML model chaching to improve response time.  
 - Health check endpoint.
 
## Project Structure
```
├── README.md
│ 
├── layouts
│   ├── __init__.py
│   └── input_layout.py
├── logging.conf
├── main.py
├── scripts
│   ├── __init__.py
│   └── training.py
└── utils
    ├── __init__.py
    └── helpers.py
```

 - main.py is the entry point of the REST API app.
 - training.py is called to train the ML pipeline.
 - input_layout.py defines a model in Pydantic.  
 - helpers.py is 

## How to use

To make prediction you need to first train a ML estimator running the training.py script;
then you can run the server with FastAPI and make predictions at 127.0.0.1/docs.
