#Python3.7.4
#This file takes the previously imputed data and tests the accuracy of predicting 20% of its 'Attended' values by training the model with Random Forest using the other 80% of the data. It returns the Accuracy Score, F1 Score, and Matrix values.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

#Imports file of choice to test accuracy.
data = pd.read_csv('D19&D21.csv', header=0)

data = data[data['Attended'].isin(['Yes', 'No'])]

# Convert 'Attended' column from 'Yes'/'No' to 1/0
data['Attended'] = data['Attended'].map({'Yes': 1, 'No': 0})

#Sets Independent and Dependent Variable
X = data[['days_before_event']]
y = data['Attended'] 

# Split data into training and testing sets, split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Defines Random Forest Function
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

train_accuracy_rf = random_forest.score(X_train, y_train)
test_accuracy_rf = random_forest.score(X_test, y_test)

predictions_rf = random_forest.predict(X_test)

conf_matrix = confusion_matrix(y_test, predictions_rf)

true_negative = conf_matrix[0][0]
false_positive = conf_matrix[0][1]
false_negative = conf_matrix[1][0]
true_positive = conf_matrix[1][1]

#Prints Accuracy, F1 Score and Matrix values.
print("Random Forest - Accuracy:", accuracy_score(y_test, predictions_rf))
print("Random Forest - F1 Score:", f1_score(y_test, predictions_rf))
print(true_negative, false_positive, false_negative, true_positive)