import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras

# Load data
train_data = pd.read_csv('t.csv')
test_data = pd.read_csv('test.csv')

# Drop unnecessary columns
train_data.drop(['Name', 'Ticket'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket'], axis=1, inplace=True)

# One-hot encode categorical variables
sex_embark_train = pd.get_dummies(train_data[['Sex', 'Embarked']], drop_first=True)
train_data = pd.concat([train_data, sex_embark_train], axis=1)
train_data.drop(['Sex', 'Embarked', 'Cabin'], axis=1, inplace=True)

# Select features and target variable
X_train = train_data[['Pclass', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']].values
Y_train = train_data['Survived'].values

# Handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
X_train[:, [1, 3]] = imputer.fit_transform(X_train[:, [1, 3]])

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.4, random_state=101)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)
predictions_logistic = logistic_model.predict(x_test)

# Naive Bayes
naive_bayes_model = GaussianNB()
predictions_naive_bayes = naive_bayes_model.predict(x_test)

# Random Forest
random_forest_model = RandomForestClassifier(n_estimators=100)
random_forest_model.fit(x_train, y_train)
predictions_random_forest = random_forest_model.predict(x_test)

# Neural Network with TensorFlow/Keras
model = keras.Sequential([
    keras.layers.Input(shape=(7,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, verbose=0)

# Predictions with Neural Network
predictions_nn = np.round(model.predict(x_test)).flatten().astype(int)

# Print classification reports
print("Logistic Regression:")
print(classification_report(y_test, predictions_logistic))

print("Naive Bayes:")
print(classification_report(y_test, predictions_naive_bayes))

print("Random Forest:")
print(classification_report(y_test, predictions_random_forest))

print("Neural Network:")
print(classification_report(y_test, predictions_nn))

# Confusion matrices
print("Confusion Matrix - Logistic Regression:")
print(confusion_matrix(y_test, predictions_logistic))

print("Confusion Matrix - Naive Bayes:")
print(confusion_matrix(y_test, predictions_naive_bayes))

print("Confusion Matrix - Random Forest:")
print(confusion_matrix(y_test, predictions_random_forest))

print("Confusion Matrix - Neural Network:")
print(confusion_matrix(y_test, predictions_nn))

# Prepare submission for Kaggle
X_test = test_data[['Pclass', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']].values
X_test[:, [1, 3]] = imputer.transform(X_test[:, [1, 3]])
X_test = scaler.transform(X_test)

submission_nn = pd.DataFrame({"PassengerId": test_data['PassengerId'], "Survived": np.round(model.predict(X_test)).flatten().astype(int)})
submission_nn.to_csv("Kaggle_NeuralNetwork.csv", index=False)
