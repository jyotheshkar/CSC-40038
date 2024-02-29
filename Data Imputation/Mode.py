"""This script conduct the mode imputation."""

# Import the necessary library
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# Read the CSV file into a pandas DataFrame
d19 = pd.read_csv('D19.csv', header=1)
d21 = pd.read_csv('D21.csv', header=1)
gp21 = pd.read_csv('GP21.csv', header=1)
mse21 = pd.read_csv('MSE21.csv', header=1)
np21 = pd.read_csv('NP21.csv', header=1)
srm22 = pd.read_csv('SRM22.csv', header=1)
srm23 = pd.read_csv('SRM23.csv', header=1)

# Formatting the dataframe
d19['File'] = "0"
d19['Row'] = d19.reset_index().index
d21['File'] = "1"
d21['Row'] = d21.reset_index().index
gp21['File'] = "2"
gp21['Row'] = gp21.reset_index().index
mse21['File'] = "3"
mse21['Row'] = mse21.reset_index().index
np21['File'] = "4"
np21['Row'] = np21.reset_index().index
srm22['File'] = "5"
srm22['Row'] = srm22.reset_index().index
srm23['File'] = "6"
srm23['Row'] = srm23.reset_index().index

# Combine the target audience into one pandas DataFrame
df = pd.concat([d19, d21, gp21, mse21, np21, srm22, srm23], ignore_index=True)

# Format the data
df["Created Date"] = pd.to_datetime(df["Created Date"], format='%d/%m/%Y')
df['Attendee Status'] = df['Attendee Status'].map({'Attending': 1, "Cancelled": 0, "Booker not attending": 0})
df['Attended'] = df['Attended'].map({'Yes': 1, 'No': 0})
# print(df)

# Split the DataFrame into training and testing set
df_train, df_test = train_test_split(df.dropna(), test_size=0.2, random_state=42) # Remove the missing values
# print(df_train)
# print(df_test)


# Group the feature attributes from the training set for counting the most common value
subset_df = df_train[['Attendee Status', 'Attended']]
grouped_df = subset_df.groupby('Attendee Status')['Attended'].agg(lambda x: x.value_counts().index[0]).reset_index()

# Impute the most common value
imputed_df = df_test.merge(grouped_df, on='Attendee Status', suffixes=('', '_imputed'))
test = imputed_df['Attended']  # Recorded values in the testing set
pred = imputed_df['Attended_imputed']  # Imputed values in the testing set


# Calculate the accuracy of the imputation
accuracy = accuracy_score(test, pred)
f1 = f1_score(test, pred)
cm = confusion_matrix(test, pred)
print("Before resampling:")
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'CM Score: {cm}')

# Resample the dataset to balance the target variables
rus = RandomUnderSampler()
X = imputed_df[['Attendee Status','Attended_imputed']]  # Features in the training set
y = imputed_df['Attended']  # Target variable in the training set
X_resampled, y_resampled = rus.fit_resample(X, y)
y_test = y_resampled
y_pred = X_resampled['Attended_imputed']
#print(X_resampled)
#print(y_resampled)
#print(y_test)
#print(y_pred)

# Calculate the accuracy of the imputation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("\nAfter resampling:")
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'CM Score: {cm}\n')

# Group the feature attributes from the training set for counting the most common value
subset_df = df[['Attendee Status', 'Attended']]
grouped_df = subset_df.dropna().groupby('Attendee Status')['Attended'].agg(lambda x: x.value_counts().index[0]).reset_index()
grouped_df.columns = ['Attendee Status', 'Attended_imputed']

# Impute the most common value
df['Attended'] = df['Attended'].fillna(df['Attendee Status'].map(grouped_df.set_index('Attendee Status')['Attended_imputed']))
df['Attended'] = df['Attended'].map({1 : 'Yes', 0 : 'No'})
df = df.sort_values(by=["File", "Row"])

# Specify the path and filename for the CSV file
csv_path = 'Imputed Result.csv'

# Write the DataFrame to a CSV file
df.to_csv(csv_path, index=False)

# print(df)