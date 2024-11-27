import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import yaml
from mlflow.models import infer_signature
import os 
from urllib.parse import urlparse
import mlflow

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/kiran-91/MLpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="kiran-91"
os.environ["MLFLOW_TRACKING_PASSWORD"]="0565d1006266530a1a0b90bfa4691b157194d022"

def hyperparameter_tuning(x_train,y_train,param_grid):
    rf=RandomForestClassifier()
    grid_search=GridSearchCV(estimator=rf, param_grid=param_grid,cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(x_train,y_train)
    return grid_search

# Load all the parameters from the params.yaml file 
print("Loading parameters from the yaml file...")
params=yaml.safe_load(open("params.yaml"))["train"]
print("params file loaded")
def train(data_path,model_path,random_state,n_estimators,max_depth):
    data=pd.read_csv(data_path)
    print("Data loaded successfully")
    x=data.drop(columns=["Outcome"])
    y=data["Outcome"]    
    print("MLflow Tracking URI:", mlflow.get_tracking_uri())
    
    
    mlflow.set_tracking_uri("https://dagshub.com/kiran-91/MLpipeline.mlflow")
    
    #start mlflow run 
    with mlflow.start_run():
        # split the data into training and testing
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
        signature=infer_signature(x_train,y_train)
        
        # defining the hyperparameter grid
        
        param_grid={
            "n_estimators": [100,200],
            "max_depth": [5,10,None],
            "min_samples_split": [2,5],
            "min_samples_leaf": [1,2],
        }
        
        #performing hyperparameter tuning
        grid_search=hyperparameter_tuning(x_train,y_train,param_grid)
        
        #getting the best model
        best_model=grid_search.best_estimator_
        
        # predicting and evaluting the model 
        y_pred=best_model.predict(x_test)
        accuracy=accuracy_score(y_test,y_pred)
        print("Accuracy is ", accuracy)
        
        # log additional metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_param("best_max_depth", grid_search.best_params_["max_depth"])
        mlflow.log_param("best_sample_split", grid_search.best_params_["min_samples_split"])
        mlflow.log_param("best_sample_leaf", grid_search.best_params_["min_samples_leaf"])
        
        # log confusion matrix and classification report 
    
        cm=confusion_matrix(y_test,y_pred)
        cr=classification_report(y_test,y_pred)
        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(str(cr),"classification_report.txt")
        
        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model,"model", registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(best_model,"model", signature=signature)
            
        # create directory to save the model
        
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        
        filename=model_path
        pickle.dump(best_model, open(filename, 'wb'))
        
        print(f"Model saved to {model_path}")
        
if __name__ == "__main__":
    train(params['data'], params['model'], params['random_state'],params['n_estimators'], params["max_depth"])
    