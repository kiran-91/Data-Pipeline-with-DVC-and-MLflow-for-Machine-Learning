import pandas as pd 
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os 
import mlflow 
from urllib.parse import urlparse


os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/kiran-91/MLpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="kiran-91"
os.environ["MLFLOW_TRACKING_PASSWORD"]="0565d1006266530a1a0b90bfa4691b157194d022"

#load the parameters from the yaml file 
params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data=pd.read_csv(data_path)
    x=data.drop(columns=["Outcome"])
    y=data["Outcome"]
    
    mlflow.set_tracking_uri("https://dagshub.com/kiran-91/MLpipeline.mlflow")
    
    # loading the model from the pickle file
    model=pickle.load(open(model_path, 'rb'))
    prediction=model.predict(x)
    accuracy=accuracy_score(y,prediction)
    
    # log the metrics 
    mlflow.log_metric("Accuracy", accuracy)
    print("Model accuracy is ", accuracy)
    
if __name__ == "__main__":
    evaluate(params["data"], params["model"])
    
    