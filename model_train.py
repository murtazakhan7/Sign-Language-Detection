import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the .npz file
def train_and_save_model():
    data = pd.read_csv('./landmarks_data_combined.csv')
    print(data.head())

    y= data['label']
    X = data.drop('label',axis='columns')
    # print(X.head())


    # print(data.info())

    # Split into features and target


    # # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # # Train the model
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # print(X_train[0])
    # # print(y_train[0:100])
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot as plt
    # print(X[120])

    # cm = confusion_matrix(y_test, y_pred)

    # import seaborn as sb
    # plt.figure(figsize=(10,7))
    # sb.heatmap(cm,annot=True)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')


    joblib.dump(model, 'logistic_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == "__main__":
    train_and_save_model()