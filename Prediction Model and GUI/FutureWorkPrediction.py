import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Read the CSV file into a pandas DataFrame
d19 = pd.read_csv('D19_imputed.csv', header=1)
d21 = pd.read_csv('D21_imputed.csv', header=1)
gp21 = pd.read_csv('GP21_imputed.csv', header=1)
mse21 = pd.read_csv('MSE21_imputed.csv', header=1)
np21 = pd.read_csv('NP21_imputed.csv', header=1)
srm22 = pd.read_csv('SRM22_imputed.csv', header=1)
srm23 = pd.read_csv('SRM23_imputed.csv', header=1)

"""Add Event Dates to a List"""
event_dates = ["19/11/2019", "09/12/2021", "22/04/2021", "24/03/2021", "09/11/2021", "15/06/2022", "08/06/2023"]
event_dates = pd.to_datetime(event_dates, dayfirst = True) # Convert the event dates to datetime format

# Format the data
d19["Created Date"] = pd.to_datetime(d19["Created Date"], dayfirst = True)
d19['days_before_event'] = (event_dates[0] - d19['Created Date']).dt.days
d21["Created Date"] = pd.to_datetime(d21["Created Date"], dayfirst = True)
d21['days_before_event'] = (event_dates[1] - d21['Created Date']).dt.days
gp21["Created Date"] = pd.to_datetime(gp21["Created Date"], dayfirst = True)
gp21['days_before_event'] = (event_dates[2] - gp21['Created Date']).dt.days
mse21["Created Date"] = pd.to_datetime(mse21["Created Date"], dayfirst = True)
mse21['days_before_event'] = (event_dates[3] - mse21['Created Date']).dt.days
np21["Created Date"] = pd.to_datetime(np21["Created Date"], dayfirst = True)
np21['days_before_event'] = (event_dates[4] - np21['Created Date']).dt.days
srm22["Created Date"] = pd.to_datetime(srm22["Created Date"], dayfirst = True)
srm22['days_before_event'] = (event_dates[5] - srm22['Created Date']).dt.days
srm23["Created Date"] = pd.to_datetime(srm23["Created Date"], dayfirst = True)
srm23['days_before_event'] = (event_dates[6] - srm23['Created Date']).dt.days

# Combine the target audience into one pandas DataFrame
ITM = pd.concat([d19, d21], ignore_index=True).dropna()
PM = pd.concat([gp21, np21], ignore_index=True).dropna()
EPM = pd.concat([mse21], ignore_index=True).dropna()
EM = pd.concat([srm22, srm23], ignore_index=True).dropna()
Other = pd.concat([d19, d21, gp21, mse21, np21, srm22, srm23], ignore_index=True).dropna()

audience_dataset = {1:ITM, 2:PM, 3:EPM, 4:EM, 5:Other}

def audience_pred(targetAudience):
    # Load and preprocess Trained Data
    Train_Data = audience_dataset[targetAudience]
    Train_Data['Attended'] = Train_Data['Attended'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

    # Prepare the feature matrix and target vector
    X = Train_Data[['days_before_event']]
    y = Train_Data['Attended']

    '''# Train the logistic regression model
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
    attendance_percentage = (total_actual_attendees / total_forecasted_registrations) * 100'''
    total_actual_attendees = Train_Data['Attended'].value_counts()[1]
    total_forecasted_registrations = len(Train_Data['Attended'])
    attendance_percentage = total_actual_attendees / total_forecasted_registrations
    return attendance_percentage
    # print(f"Total Predicted Attendence: {total_actual_attendees:.0f} ")
    # print(f"Total Predicted Attendance Percentage of Forecasted Registrations: {attendance_percentage:.2f}%")

    #This percentage and actual number would have been printed on to the frontend
