import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
heart_data = pd.read_csv('/content/heart_disease_data.csv')

# Display the first few rows of the dataset
print(heart_data.head())

# Display the shape of the dataset
print(heart_data.shape)

# Display information about the dataset
print(heart_data.info())

# Check for missing values in the dataset
print(heart_data.isnull().sum())

# Display summary statistics of the dataset
print(heart_data.describe())

# Display the distribution of the target variable
print(heart_data['target'].value_counts())

# Separate features (X) and target variable (Y)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Display features and target variable
print(X)
print(Y)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model on the training set
model.fit(X_train, Y_train)

# Make predictions on the training set
X_train_prediction = model.predict(X_train)

# Evaluate the accuracy on the training set
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# Input data for prediction
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

# Convert the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make predictions on the input data
prediction = model.predict(input_data_reshaped)
print(prediction)

# Display the prediction result
if prediction[0] == 0:
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')
