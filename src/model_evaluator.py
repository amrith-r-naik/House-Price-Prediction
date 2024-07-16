# Import Libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix

# Function that takes in model_path & test_data.csv as input then performs metric calculation, displaying the confusion matrix and what not
def evaluate(model_path, test_data_path):
  test_data = pd.read_csv(test_data_path)   # Load the test data
  model = joblib.load(model_path)           # Load the saved model

  X_test = test_data[1:-1]                  # Get features columns
  y_true = test_data[-1]                    # Get target variable

  y_pred = model.predict(X_test)            # Test the model

  # Calculate the metrics and store in a dictionary
  metrics = {}
  metrics['Accuracy'] = accuracy_score(y_true,y_pred)
  metrics['Precision'] = precision_score(y_true,y_pred)
  metrics['F1 Score'] = f1_score(y_true,y_pred)
  metrics['Sensitivity'] = recall_score(y_true,y_pred)
  cm = confusion_matrix(y_true,y_pred)
  tn = cm[0, 0]
  fp = cm[0, 1]
  metrics['Specificity'] = tn / (tn + fp) if (tn + fp) != 0 else 0

  # Print each of the metric
  for metric, value in metrics.items():
    print(f'{metric}: {(value*100):.2f}%')
