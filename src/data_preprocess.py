# Importing dependencies & libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Preprocessing
def preprocess(data):

    # Data Split
    X = data.drop('Price', axis=1)
    y = data['Price']

    enc = LabelEncoder()
    X['City'] = enc.fit_transform(X['City'])
    X['Location'] = enc.fit_transform(X['Location'])

    std_enc = StandardScaler()
    X['Area'] = std_enc.fit_transform(X[["Area"]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv('data/training_data.csv', index=False)
    test.to_csv('data/testing_data.csv', index=False)

if __name__ == '__main__':
    dataset = pd.read_csv('data/csvdata.csv')
    preprocess(dataset)