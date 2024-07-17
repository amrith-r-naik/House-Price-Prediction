import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def model_trainer(data):

    # Declaring dictionary of models
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000,solver='saga',verbose=1),
        'random_forest': RandomForestRegressor(verbose=1),
        'svr': SVR(verbose=1),
        'xgboost': XGBRegressor(),
        'knn': KNeighborsRegressor(),
    }

    # Model training 
    for name, model in models.items():
        print(f"{name} Model Running...")
        predictor = model.fit(data.drop('Price', axis=1), data['Price'])
        
        # Saving the trained model
        joblib.dump(predictor, f"models/{name}_model.pkl")
        print(f"Model {name} saved into 'models' directory.")

if __name__ == '__main__':
    dataset = pd.read_csv('data/training_data.csv')
    model_trainer(dataset)
