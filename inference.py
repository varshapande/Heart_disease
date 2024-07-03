import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(data):
    data = data.drop_duplicates()
    
    # Encode categorical features
    c = data.select_dtypes(include='object').columns.to_list()
    encoder = LabelEncoder()
    for col in c:
        data[col] = encoder.fit_transform(data[col])
    
    # Handle outliers in 'RestingBP'
    UPPER_LIMIT = data['RestingBP'].mean() + 3*data['RestingBP'].std()
    LOWER_LIMIT = data['RestingBP'].mean() - 3*data['RestingBP'].std()
    data['RestingBP'] = np.where(data['RestingBP'] > UPPER_LIMIT, UPPER_LIMIT,
                                 np.where(data['RestingBP'] < LOWER_LIMIT, LOWER_LIMIT, data['RestingBP']))
    
    return data

def inference(model, data, feature_columns):
    # Ensure the columns are in the same order as during training
    X_test = data[feature_columns]
    return model.predict(X_test)

if __name__ == "__main__":
    # Load the saved model
    best_rf_model = joblib.load('best_rf_model.joblib')
    print("Model loaded from 'best_rf_model.joblib'")
    
    # Load and preprocess new data
    new_data = load_data('/home/varshap/Downloads/heartdisease 1/Heart_Disease Prediction.csv')
    new_data = preprocess_data(new_data)
    
    feature_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                       'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                       'Oldpeak', 'ST_Slope']
    
    # Perform inference
    predictions = inference(best_rf_model, new_data, feature_columns)
    
    print("Inference results:", predictions)
