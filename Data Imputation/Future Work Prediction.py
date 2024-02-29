#Python3.7.4
#This code uses logistic imputation to predict the missing values in a new csv file sent across from the previous prediction of bookings per day. It leverages a chosen combined dataset to train the model and then uses it to impute the percentage chance of a booking to attend on a specific day. It then returns the overall predicted attendance in raw and percentage form.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and preprocess Trained Data
Train_Data = pd.read_csv('D19&D21.csv')
Train_Data['Attended'] = Train_Data['Attended'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

# Prepare the feature matrix and target vector
X = Train_Data[['days_before_event']]
y = Train_Data['Attended']

# Train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Load "Target file, already had booking number predicted per day
Test_Data = pd.read_csv('IT Managers.csv')

# Implement training data to new dataset
attendance_probabilities = model.predict_proba(Test_Data[['Days to Event (T-minus)']].rename(columns={'Days to Event (T-minus)': 'days_before_event'}))[:, 1]

# Estimate 'Actual Attendees'
Test_Data['Attended'] = attendance_probabilities  # Stores the probabilities in 'Attended' column
Test_Data['Actual Attendees'] = Test_Data['Forecasted Registrations'] * Test_Data['Attended']

# Calculate the total predicted attendance and express it as a percentage of total forecasted registrations
total_actual_attendees = Test_Data['Actual Attendees'].sum()
total_forecasted_registrations = Test_Data['Forecasted Registrations'].sum()
attendance_percentage = (total_actual_attendees / total_forecasted_registrations) * 100
print(f"Total Predicted Attendence: {total_actual_attendees:.0f} ")
print(f"Total Predicted Attendance Percentage of Forecasted Registrations: {attendance_percentage:.2f}%")

#This percentage and actual number would have been printed on to the frontend
