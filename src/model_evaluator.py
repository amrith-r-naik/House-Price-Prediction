# Import Libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_metrics(model_path, test_data_path, residual_plot=True):
  '''
  Evaluates given model on the test data and prints the metrics.

  Parameters
  ----------
  model_path : str
    The path to where the model is stored.
  test_data_path : str
    The path to where the test data is stored. The test data must be a csv file.
  residual_plot : bool, default=True
    If "False", the function does not display a residual plot.
    Else, the function displays a residual plot.

  Examples
  ----------
  >>> evaluate_metrics('models/randomforest_model.pkl','data/test.csv')
  >>> evaluate_metrics('models/xgboost_model.pkl','data/test.csv')
  '''

  # Load the test data and saved model
  test_data = pd.read_csv(test_data_path)
  model = joblib.load(model_path)

  # Get features columns and target variable
  X_test = test_data[1:-1]
  y_true = test_data[-1]

  # Test the model
  y_pred = model.predict(X_test)

  # Calculate the metrics and store in a dictionary
  metrics = {}
  metrics['Accuracy'] = accuracy_score(y_true,y_pred)
  metrics['Precision'] = precision_score(y_true,y_pred)
  metrics['F1 Score'] = f1_score(y_true,y_pred)
  metrics['Mean Squared Error'] = mean_squared_error(y_true,y_pred)
  metrics['Mean Absolute Error'] = mean_absolute_error(y_true,y_pred)

  # Print each of the metric
  for metric, value in metrics.items():
    print(f'{metric}: {(value*100):.2f}%')

  # Print the residual plot yet to be implemented
