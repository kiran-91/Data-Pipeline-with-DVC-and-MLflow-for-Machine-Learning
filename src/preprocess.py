import pandas as pd 
import sys 
import yaml
import os 

## Loading the parameters from params.yaml file 
params=yaml.safe_load(open("params.yaml"))["preprocess"]

def preprocess(input_path,output_path):
    data=pd.read_csv(input_path)
    # the dataset dows not have any null values and is very clean and numeric
    # so we can directly save it to the output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)
    print(f"Preprocessed data saved to {output_path}")
    
if __name__ == "__main__":
    preprocess(params["input"],params["output"])