# Importing dependencies & libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import TargetEncoder

# Preprocessing
def preprocess(data):

    # Sorting the database on 'Price' feature
    data = data.sort_values(by='Price')  # save the sorted dataset as the primary data.
    
    # Dropping off columns with unique count < 5

    location = data['Location'].value_counts()
    single_occurence_loc = location[location==4].index # The number goes from 1 to 4 as minimum splits must be 5 for Taegret Encoding.
    data = data[~data['Location'].isin(single_occurence_loc)] # Save the dataset as primary dataset.

    # Preprocessing

    label_enc = LabelEncoder()      # Label Encoding of 'City'
    data['City'] = label_enc.fit_transform(data['City'])

    target_enc = TargetEncoder()    # Target Encoding of 'Location' column
    data['Location'] = target_enc.fit_transform(data[['Location']], data[['Price']])
    
    data.to_csv('data/preprocessed_data.csv', index=False)
    
    # Data Split