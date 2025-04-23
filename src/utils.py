from typing import Dict, Any, Union
import os
import sys  
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj: Any) -> None:
    """
    Save any object to a file using dill serialization
    
    Args:
        file_path: Path where object will be saved
        obj: Any Python object to be serialized
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict,
    param: Dict
) -> Dict[str, float]:
    """
    Evaluate multiple models using GridSearchCV
    
    Returns:
        Dict containing model names and their R² scores
    """
    try:
        report = {}
        for model_name, model in models.items():
            para = param[model_name]
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_) # Update model parameters
            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report
    except Exception as e:
        logging.info('Exception occurred during model training')
        raise CustomException(e, sys)
    
def load_object(file_path: str) -> Any:
    """
    Load an object from a file using dill serialization
    
    Args:
        file_path: Path to the file containing the serialized object
        
    Returns:
        The deserialized object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)