
# Diabetes Risk Assessment: An end-to-end Machine Learning Pipeline with DVC and MLflow
### Project Overview
This project is an end-to-end machine learning pipeline designed to predict the likelihood of diabetes in patients using the Pima Indians Diabetes Dataset. The pipeline includes data preprocessing, model training, evaluation, and logging of experiments using MLflow. The project is built with DVC (Data Version Control) for managing data and pipeline stages, ensuring reproducibility and scalability.

The goal of this project is to demonstrate how to build a robust and reproducible machine learning workflow using modern tools like DVC, MLflow, and scikit-learn. The pipeline is modular, allowing for easy updates to the dataset, model, or evaluation metrics.

---

## 📂Project Structure
``` bash
MLpipeline/
├── data/
│   ├── raw/                  # Raw dataset files
│   │   └── data.csv          # Original dataset 
│   └── processed/            # Processed dataset files
│       └── data.csv          # Cleaned and preprocessed dataset
│
├── models/                   # Trained model files
│   └── model.pkl             # Trained model saved as pickle file
│
├── src/                      # Source code for the pipeline
│   ├── __init__.py           # Python package initialization
│   ├── preprocess.py         # Script for data preprocessing
│   ├── train.py              # Script for model training
│   └── evaluate.py           # Script for model evaluation
│
├── params.yaml               # Configuration file for pipeline parameters
├── requirements.txt          # Python dependencies for the project
├── dvc.yaml                  # DVC pipeline configuration file
├── README.md                 # Project documentation
├── .gitignore                # Files and directories to ignore in Git
└── .dvcignore                # Files and directories which DVC ignores 
```
---
## Key Features
1. Data Preprocessing:
   - Handling missing values, renaming columns and prepare the dataset for `training`
   - Saves the cleaned dataset in the `data/preprocesses/` directory
2. Model Training:
   - Trains a Random Forest Classifier using hyperparameter tuning
   - Logs training metrics (e.g., accuracy, precision,recall) using MLFlow
3. Model Evaluation:
   - Evaluates the trained model on the test set and logs performance metrics
   - Saves the trained model as serialized file `model.pkl` in the `models/` directory
4. Reproducibility
   - Uses **DVC** to manage data and pipeline stages, ensuring reproducibility and scalability
   - Ensures that the pipeline can be rerun with the same results
5. Experiment tracking:
   - Uses **MLFlow** to track experiments, logs, parameters and visualize metrics.

---

## Pipeline Stages
The pipeline consists of the following stages, defined in the `dvc.yaml` file:
1. Preprocessing:
   - Cleans the raw dataset and saves it to the `data/preprocessed/` directory
   - Handles missing values and renames columns for consistencey 

2. Training:
   - Trains a Random Forest Classifier using preprocessed dataset
   - Logs the hyperparameters and metrics using MLFlow

3. Evaluation:
   - Evaluates the trained model on the test set
   - Logs the performance metrics (e.g., accuracy, F1Score) using MLFlow

4. Pipeline Execution:
   - The pipeline is executed using **DVC**, which ensures reproducibility and tracks dependencies between stages

---

## Installation 
1. Clone the repository 
``` bash 
   git clone https://dagshub.com/kiran-91/MLpipeline.git
   cd MLpipeline
   ```
2. Install the required packages using pip
``` bash
   pip install -r requirements.txt
```
3. Run the Pipeline
``` bash 
   dvc repro
```
4. View the results in MLFLow
``` bash
   mlflow ui
```
---
### For Adding Stages
```bash
   dvc stage add -n preprocess \
   -p preprocess.input,preprocess.output \
   -d src/preprocess.py -d data/raw/data.csv \
   -o data/preprocessed/data.csv \
   python src/preprocess.py
```
	
``` bash	
   dvc stage add -n train \
   -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
   -d src/train.py -d data/raw/data.csv \
   -o models/model.pkl \
   python src/train.py
```
	
 ``` bash  
   dvc stage add -n evaluate \
   -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
   python src/evaluate.py
```
---
## Results
For real-time project tracking, please refer to this DagsHub link, where DVC and MLFlow are monitoring progress. You can view live updates and changes in the MLFlow dashboard

👉 **[MLFlow tracking](https://dagshub.com/kiran-91/MLpipeline)**

- Checkout on the above `link`
- Click on the `Experiments` tab
- Click on the `Go to MLFlow UI` button to view the MLFlow dashboard
- You can view the experiment logs, model artifacts, and performance metrics in the MLFlow dashboard.

---

### Key Topics

#### Data Version Control (DVC)
- **What is DVC?**
  DVC is an open-source tool designed for versioning datasets, machine learning models, and pipeline stages. It works seamlessly with Git to ensure that data and models are versioned just like code.

- **How DVC Works in This Project:**
  - **Reproducibility:** The dataset, models, and pipeline stages are tracked using DVC, making it easy to reproduce results in different environments.
  - **Pipeline Stages:** The project is broken into stages (e.g., preprocessing, training, evaluation), which are automatically updated if dependencies like data or code are modified.
  - **Remote Storage:** DVC supports remote data storage solutions such as **DagsHub**, **AWS S3**, and **Google Drive**, making it suitable for large datasets and models that can’t be stored in Git.

#### Experiment Tracking with MLflow
- **What is MLflow?**
  MLflow is an open-source platform for managing the lifecycle of machine learning projects. It helps track experiment parameters, metrics, and artifacts.

- **How MLflow Works in This Project:**
  - **Experiment Logs:** Tracks hyperparameters like `n_estimators` and `max_depth` and performance metrics like accuracy.
  - **Comparison:** Allows comparing different experiments to identify the best model.
  - **Artifacts:** Logs model files (e.g., `random_forest.pkl`) and evaluation results.

#### Stages of the Pipeline
1. **Preprocessing:**
   - Script: `preprocess.py`
   - Task: Reads the raw dataset from `data/raw/data.csv`, performs basic preprocessing (e.g., renaming columns), and outputs cleaned data to `data/processed/data.csv`.
   - Benefit: Ensures consistent data preparation across runs.

2. **Training:**
   - Script: `train.py`
   - Task: Trains a **Random Forest Classifier** on the preprocessed data. The trained model is saved as `models/model.pkl`.
   - Benefit: Hyperparameters and model artifacts are logged to MLflow for easy tracking and reproducibility.

3. **Evaluation:**
   - Script: `evaluate.py`
   - Task: Loads the trained model, evaluates its performance on the dataset, and logs metrics (e.g., accuracy) to MLflow.
   - Benefit: Facilitates easy comparison of different models and experiments.

---


### Additional Concepts

#### What is Git?
Git is a distributed version control system that helps developers track changes in their codebase. It’s widely used for collaborative development, ensuring all changes are logged and easily reversible.

#### What is DagsHub, and How is it Different from GitHub?
- **DagsHub** is a platform built specifically for machine learning projects. It integrates tools like DVC, MLflow, and Git to provide a complete solution for data and model management.
- **GitHub**, on the other hand, is a general-purpose platform for hosting code repositories. While it’s great for managing code, it doesn’t natively support large file versioning or experiment tracking.
- **Key Differences:**
  - **Data Management:** DagsHub allows versioning of datasets and models, while GitHub primarily handles code.
  - **Integration:** DagsHub integrates with DVC and MLflow for managing pipelines and experiments, which GitHub lacks natively.

---

### Goals of This Project
1. **Reproducibility:**
   - Using DVC ensures that results can be reproduced by tracking data, code, and model versions.

2. **Experimentation:**
   - MLflow allows easy tracking and comparison of different experiments to optimize the machine learning process.

3. **Collaboration:**
   - Tools like DVC and DagsHub enable teams to collaborate efficiently, ensuring smooth version control for both code and data.

---

### Use Cases
1. **Data Science Teams:** Organize and track datasets, models, and experiments in a reproducible manner.
2. **Machine Learning Researchers:** Iteratively experiment with models and easily manage data versions and performance metrics.
3. **Large-Scale Projects:** Handle big datasets and ensure seamless collaboration among multiple contributors.

---

### Technology Stack
- **Python:** Core programming language for the project.
- **DVC:** Tracks datasets, models, and pipelines.
- **MLflow:** Logs experiments, metrics, and artifacts.
- **Scikit-learn:** Builds and trains the Random Forest Classifier.
- **DagsHub:** Combines Git, DVC, and MLflow for a unified machine learning workflow.

---

This project demonstrates how to build and manage an end-to-end machine learning pipeline with industry-standard tools. By integrating DVC, MLflow, and DagsHub, it ensures reproducibility, scalability, and effective collaboration—essential for any modern data science team.