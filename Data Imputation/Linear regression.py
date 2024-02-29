# This code uses the Linear Regression model to impute the missing values in the Attended column.
# It switches the Attended column to a percentage of attending and uses it as a target.
# It calculates the number of days left till the event for each row and uses that as a feature.
# Metrics like R-squared, Mean Absolute Error, and Relative Mean Absolute Error (Percentage) are used to evaluate the performance of the model.


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Reading from CSV file
file_path = 'C:\\Users\\Admin\\Music\\IT.csv'
Linear_df = pd.read_csv(file_path)

# Converting the columns 'Created Date' and Event start date to day, month, year format using to_datetime
Linear_df['Created Date'] = pd.to_datetime(Linear_df['Created Date'], format='%d/%m/%Y')
Linear_df['Event start date'] = pd.to_datetime(Linear_df['Event start date'], format='%d/%m/%Y')

# Choosing the first row of Event start date column
event_start_date = Linear_df.loc[0, 'Event start date']

# Calculating the number of days before the event for each registration
Linear_df['DaysBeforeEvent'] = (event_start_date - Linear_df['Created Date']).dt.days

# Locating the index of the first value of zero in the 'DaysBeforeEvent' column
first_zero_index = Linear_df[Linear_df['DaysBeforeEvent'] == 0].index.min()

# Slicing the Linear_df up to the index of the first zero
Linear_df = Linear_df.loc[:first_zero_index]

# Dropping the rows with missing values to prepare the modeling
Linear_df.dropna(subset=['Created Date','Attended', 'Attendee Status', 'Reference', 'DaysBeforeEvent'], inplace=True)

# Converting the categorical variables of Attended and Attendee Status columns to numerical using map 
Linear_df['Attended'] = Linear_df['Attended'].map({'Yes': 1, 'No': 0})
Linear_df['Attendee Status'] = Linear_df['Attendee Status'].map({'Attending': 1, 'Cancelled': 0, 'Booker not attending': 0})

# Calculating the percentage of attendees for each date using mean
percentage_attended = Linear_df.groupby('Created Date')['Attended'].mean() * 100

# Creating a DataFrame with the percentage_attended
percentage_attended_df = pd.DataFrame({'Percentage Attended': percentage_attended})

# Merging the percentage_attended_df with the Linear_df using merge
Linear_df = pd.merge(Linear_df, percentage_attended_df, on='Created Date', how='left')

# Setting the threshold for considering 'Attended' as 'Yes'
threshold_percentage = 50

# Reverting the 'Attended' column values based on the threshold
Linear_df['Attended'] = np.where(Linear_df['Percentage Attended'] >= threshold_percentage, 'Yes', 'No')

# Adding the 'Attended' column to Linear_df
Linear_df['Attended'] = Linear_df['Attended'].map({'Yes': 1, 'No': 0})

# Splitting the data into train and test sets 80% for training and 20% testing  
X = Linear_df[['DaysBeforeEvent']]
y = Linear_df[['Percentage Attended']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# After training the model, we test the model on the remaining 20%
y_pred = model.predict(X_test)

# Calculating the R squared value
r2 = r2_score(y_test, y_pred)

# Calculating the mean absolute error value
mae = mean_absolute_error(y_test, y_pred)

# Calculating the range of the test data 
range_y = np.ravel(y_test).max() - np.ravel(y_test).min()

# Calculating the relative mean absolute error
relative_mae = mae / range_y

# Converting the relative MAE to percentage by Multiplying by 100
relative_mae_percentage = relative_mae * 100

# Print the results for the R Squared, Mean Absolute Error, and Relative Mean Absolute Error (Percentage)
print("R^2 Score:", r2)
print("Mean Absolute Error:", mae)
print("Relative Mean Absolute Error (Percentage): {:.2f}%".format(relative_mae_percentage))

# Imputing the values for missing data if there are missing values in the attended column
if Linear_df['Attended'].isnull().any():
    X_missing = Linear_df.loc[Linear_df['Attended'].isnull(), ['DaysBeforeEvent']]
    y_missing_pred = model.predict(X_missing)
    Linear_df.loc[Linear_df['Attended'].isnull(), 'Percentage Attended'] = np.ravel(y_missing_pred)
    Linear_df['Attended'] = np.where(Linear_df['Percentage Attended'] >= threshold_percentage, 'Yes', 'No')
    print("Missing values predicted and filled in.")


# Dropping the 'Event start date' column before printing Linear_df
Linear_df.drop(columns=['Event start date'], inplace=True)

# Print the updated DataFrame with missing values filled in
print(Linear_df)
# Save the linear_df with the imputed data in a CSV file called output_linear_regression
output_file_path = 'C:\\Users\\Admin\\Music\\output_linear_regression.csv'
Linear_df.to_csv(output_file_path, index=False)
