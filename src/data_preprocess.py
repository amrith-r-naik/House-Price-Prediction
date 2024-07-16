# Importing dependencies & libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Preprocessing
def preprocess(data):
    # Load data
    #print(data.head(5))

    # Getting to know data
    #print(f'Number of rows: {data.shape[0]}')
    #print(f'Number of columns: {data.shape[1]}')
    #print(data.info()) # There are no null values

    #for col in data.columns:
    #    print(data[col].value_counts())
    # Dont think any feature selection and further preprocessing necessary.

    # Data Split
    X = data.drop('Price', axis=1)
    y = data['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv('data/training_data.csv', index=False)
    test.to_csv('data/testing_data.csv', index=False)

if __name__ == '__main__':
    dataset = pd.read_csv('data/csvdata.csv')
    preprocess(dataset)