import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
import joblib

warnings.filterwarnings('ignore')

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

def train_test_split_data(data):
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=50)
    param_dist = {
        "max_depth": [3, None],
        "max_features": sp_randint(5, 11),
        "min_samples_split": sp_randint(2, 11),
        "min_samples_leaf": sp_randint(1, 11),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }
    samples = 8  # number of random samples
    randomCV = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=samples, cv=3)
    randomCV.fit(X_train, y_train)
    
    print("Best Parameters:", randomCV.best_params_)
    return randomCV.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

if __name__ == "__main__":
    data = load_data('/home/varshap/Downloads/heartdisease 1/Heart_Disease Prediction.csv')
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split_data(data)
    
    best_rf_model = train_random_forest(X_train, y_train)
    evaluate_model(best_rf_model, X_test, y_test)
    
    # Save the model to a file
    joblib.dump(best_rf_model, 'best_rf_model.joblib')
    print("Model saved as 'best_rf_model.joblib'")
