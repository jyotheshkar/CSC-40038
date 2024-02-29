#Python3.7.4
#This code uses logistic imputation to predict the missing values in a given raw data set, by combining datasets by event type. It then returns the data as a new file.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#Preprocesses Data File
def preprocess_data(file_path, event_name):
    data = pd.read_csv(file_path, header=0)
    #Date of Event was added into file to read
    event_date = pd.to_datetime(data.loc[0, 'Date of Event'], dayfirst=True)
    data['Created Date'] = pd.to_datetime(data['Created Date'], dayfirst=True)
    # Calculate 'days_before_event' for all rows using the singular event date
    data['days_before_event'] = (event_date - data['Created Date']).dt.days
    data['Event'] = event_name
    return data

# Use the function to load and preprocess each CSV file
data1 = preprocess_data('D21.csv', 'Data 1')
data2 = preprocess_data('D19.csv', 'Data 2')
data3 = preprocess_data('GP21.csv', 'Data 3')
data4 = preprocess_data('MSE21.csv', 'Data 4')
data5 = preprocess_data('NP21.csv', 'Data 5')
data6 = preprocess_data('SRM22.csv', 'Data 6')
data7 = preprocess_data('SRM23.csv', 'Data 7')

# Combine the datasets, can choose 1 or multiple.
combined_data = pd.concat([data1, data2, data3, data4, data5, data6, data7], ignore_index=True)

#Defines missing data
missing_attended = combined_data['Attended'].isnull().sum()

#Assigns numerical values to yes and no
combined_data['Attended'] = combined_data['Attended'].map({'Yes': 1, 'No': 0})

#Defines the data that exists
train_data = combined_data[combined_data['Attended'].notnull()]
#Defines blank cells
predict_data = combined_data[combined_data['Attended'].isnull()]

#Trains training data
X_train = train_data[['days_before_event']]
y_train = train_data['Attended']

#Defines Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

#Applies trained model to blank cells
X_predict = predict_data[['days_before_event']]
predicted_attended = model.predict(X_predict)

#Combines all yes values
combined_data.loc[combined_data['Attended'].isnull(), 'Attended'] = predicted_attended

#Converts to percentage 
percentage_attended_yes = (combined_data['Attended'].sum() / len(combined_data)) * 100

print(f"Percentage of 'Attended' as 'Yes': {percentage_attended_yes}%")

#Saves combined new data to csv
def save_combined_data_to_csv(data, file_path, include_index=False):
    data['Attended'] = data['Attended'].map({1: 'Yes', 0: 'No'})
    data.to_csv(file_path, index=include_index)

file_path = '~/Desktop/ALL.csv' 
save_combined_data_to_csv(combined_data, file_path)
print(f"Data saved to {file_path}.")