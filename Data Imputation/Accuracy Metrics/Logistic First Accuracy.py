#Python3.7.4
#This file takes the raw data and tests the accuracy of predicting 20% of its 'Attended' values by training the model with Logistic Imputation using the other 80% of the filled in data. It returns the Accuracy Score, F1 Score, and Matrix values. It also uses smote to deal with the oversampling of the 'yes' value.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# Load the first CSV file
data1 = pd.read_csv('D21.csv', header=0)
data1['Date of Event'] = pd.to_datetime(data1['Date of Event'], dayfirst=True)
data1['Created Date'] = pd.to_datetime(data1['Created Date'], dayfirst=True)
data1['days_before_event'] = (data1['Date of Event'].iloc[0] - data1['Created Date']).dt.days

# Load the second CSV file
data2 = pd.read_csv('D19.csv', header=0)
data2['Date of Event'] = pd.to_datetime(data2['Date of Event'], dayfirst=True)
data2['Created Date'] = pd.to_datetime(data2['Created Date'], dayfirst=True)
data2['days_before_event'] = (data2['Date of Event'].iloc[0] - data2['Created Date']).dt.days 

# Load the third CSV file
data3 = pd.read_csv('GP21.csv', header=0)
data3['Date of Event'] = pd.to_datetime(data3['Date of Event'], dayfirst=True)
data3['Created Date'] = pd.to_datetime(data3['Created Date'], dayfirst=True)
data3['days_before_event'] = (data3['Date of Event'].iloc[0] - data2['Created Date']).dt.days 

# Load the fourth CSV file
data4 = pd.read_csv('MSE21.csv', header=0)
data4['Date of Event'] = pd.to_datetime(data4['Date of Event'], dayfirst=True)
data4['Created Date'] = pd.to_datetime(data4['Created Date'], dayfirst=True)
data4['days_before_event'] = (data4['Date of Event'].iloc[0] - data2['Created Date']).dt.days 

# Load the fifth CSV file
data5 = pd.read_csv('NP21.csv', header=0)
data5['Date of Event'] = pd.to_datetime(data5['Date of Event'], dayfirst=True)
data5['Created Date'] = pd.to_datetime(data5['Created Date'], dayfirst=True)
data5['days_before_event'] = (data5['Date of Event'].iloc[0] - data2['Created Date']).dt.days 

# Load the sixth CSV file
data6 = pd.read_csv('SRM22.csv', header=0)
data6['Date of Event'] = pd.to_datetime(data6['Date of Event'], dayfirst=True)
data6['Created Date'] = pd.to_datetime(data6['Created Date'], dayfirst=True)
data6['days_before_event'] = (data6['Date of Event'].iloc[0] - data2['Created Date']).dt.days 

# Load the seventh CSV file
data7 = pd.read_csv('SRM23.csv', header=0)
data7['Date of Event'] = pd.to_datetime(data7['Date of Event'], dayfirst=True)
data7['Created Date'] = pd.to_datetime(data7['Created Date'], dayfirst=True)
data7['days_before_event'] = (data7['Date of Event'].iloc[0] - data2['Created Date']).dt.days 

All_Data=[data1, data2, data3, data4, data5, data6, data7]

# Combine the datasets
combined_data = pd.concat([data7], ignore_index=True)

#Fill in blanks wiht 'Unknown'
combined_data['Attended'] = combined_data['Attended'].fillna('Unknown')

# Exclude rows with 'Unknown' placeholders or any other non-target value
combined_data = combined_data[combined_data['Attended'].isin(['Yes', 'No'])]

# Convert 'Attended' column from 'Yes'/'No' to 1/0
combined_data['Attended'] = combined_data['Attended'].map({'Yes': 1, 'No': 0})

# Convert 'Attendee Status' column into categorical variables
combined_data['Attendee Status'] = combined_data['Attendee Status'].map({'Attending': 1, 'Cancelled': 0, 'Booker not attending': 0})

# Features to include in the test
features = ['days_before_event'] 

# Split data into features (X) and target (y)
X = combined_data[features]
y = combined_data['Attended'] 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_smote, y_train_smote)

# Evaluate model on training data
train_accuracy = logistic_regression.score(X_train_smote, y_train_smote)

# Evaluate model on testing data
test_accuracy = logistic_regression.score(X_test, y_test)

predictions = logistic_regression.predict(X_test)

f1 = f1_score(y_test, predictions)

conf_matrix = confusion_matrix(y_test, predictions)

class_report = classification_report(y_test, predictions)

#SMOTE bring Down yes to undersample, would need the code tweaking to add SMOTE

print("Accuracy:", accuracy_score(y_test, predictions))

print("F1 Score:", f1)

print("Confusion Matrix:\n", conf_matrix)



