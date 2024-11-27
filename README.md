
# End-to-End Machine Learning Pipeline with DVC, MLflow, and DagsHub: A Random Forest Case Study
### Project: Data Pipeline with DVC and MLflow for Machine Learning

This project demonstrates how to build a complete machine learning pipeline using **DVC (Data Version Control)** for managing data and model versions, and **MLflow** for tracking experiments. The pipeline involves training a **Random Forest Classifier** on the **Pima Indians Diabetes Dataset** and is designed to ensure reproducibility, organization, and efficient collaboration.

---

### Key Features

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
   - Task: Trains a **Random Forest Classifier** on the preprocessed data. The trained model is saved as `models/random_forest.pkl`.
   - Benefit: Hyperparameters and model artifacts are logged to MLflow for easy tracking and reproducibility.

3. **Evaluation:**
   - Script: `evaluate.py`
   - Task: Loads the trained model, evaluates its performance on the dataset, and logs metrics (e.g., accuracy) to MLflow.
   - Benefit: Facilitates easy comparison of different models and experiments.

### For Adding Stages

dvc stage add -n preprocess \
-    -p preprocess.input,preprocess.output \
-    -d src/preprocess.py -d data/raw/data.csv \
-    -o data/preprocessed/data.csv \
-    python src/preprocess.py
	
	
dvc stage add -n train \
-    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
-    -d src/train.py -d data/raw/data.csv \
-    -o models/model.pkl \
-    python src/train.py
	
dvc stage add -n evaluate \
-    -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
-    python src/evaluate.py

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