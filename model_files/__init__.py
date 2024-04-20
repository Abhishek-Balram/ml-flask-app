import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def preprocess_origin_cols(df):
    """
    Preprocesses the 'Origin' column in the given DataFrame.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the 'Origin' column.
        
    Returns:
        pandas.DataFrame: The DataFrame with the preprocessed 'Origin' column.
    """
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
    return df

acc_ix, hpower_ix, cyl_ix = 3, 5, 1

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    """
    Custom transformer that adds new attributes to the data.
    """
    def __init__(self, acc_on_power=True):
        """
        Initializes the CustomAttrAdder.
        
        Args:
            acc_on_power (bool): Whether to include the 'acc_on_power' attribute (default: True).
        """
        self.acc_on_power = acc_on_power
        
    def fit(self, X, y=None):
        """
        Fits the transformer to the data.
        
        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray, optional): The target values (default: None).
            
        Returns:
            CustomAttrAdder: The fitted transformer.
        """
        return self
    
    def transform(self, X):
        """
        Transforms the input data by adding new attributes.
        
        Args:
            X (numpy.ndarray): The input data.
            
        Returns:
            numpy.ndarray: The transformed data with new attributes.
        """
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        return np.c_[X, acc_on_cyl]

def num_pipeline_transformer(data):
    """
    Creates a pipeline transformer for numerical attributes.
    
    Args:
        data (pandas.DataFrame): The input DataFrame.
        
    Returns:
        tuple: A tuple containing the numerical attributes and the numerical pipeline transformer.
    """
    numerics = ['float64', 'int64']
    num_attrs = data.select_dtypes(include=numerics)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
    ])
    return num_attrs, num_pipeline

def pipeline_transformer(data):
    """
    Creates a full pipeline transformer for the input data.
    
    Args:
        data (pandas.DataFrame): The input DataFrame.
        
    Returns:
        ColumnTransformer: The full pipeline transformer.
    """
    cat_attrs = ["Origin"]
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(), cat_attrs),
    ])
    full_pipeline.fit_transform(data)
    return full_pipeline

def predict_mpg(config, model):
    """
    Predicts the miles per gallon (MPG) for a given configuration using a trained model.
    
    Args:
        config (dict or pandas.DataFrame): The configuration data.
        model (object): The trained model.
        
    Returns:
        numpy.ndarray: The predicted MPG values.
    """
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
        
    preproc_df = preprocess_origin_cols(df)
    print(preproc_df)
    
    pipeline = pipeline_transformer(preproc_df)
    prepared_df = pipeline.transform(preproc_df)
    print(len(prepared_df[0]))
    
    y_pred = model.predict(prepared_df)
    return y_pred