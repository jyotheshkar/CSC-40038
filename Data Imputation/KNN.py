"""This script conduct the KNN imputation."""

# Import the necessary library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Set the n neighbour to use in the KNN method
n = 47

# Read the CSV file into a pandas DataFrame
d19 = pd.read_csv('D19.csv', header=1)
d21 = pd.read_csv('D21.csv', header=1)
gp21 = pd.read_csv('GP21.csv', header=1)
mse21 = pd.read_csv('MSE21.csv', header=1)
np21 = pd.read_csv('NP21.csv', header=1)
srm22 = pd.read_csv('SRM22.csv', header=1)
srm23 = pd.read_csv('SRM23.csv', header=1)
dates = pd.read_csv('Events.csv', header=0)

# Format the dataframe
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

# Convert "Event start date" column to datetime format
dates["Event start date"] = pd.to_datetime(dates["Event start date"], format='%d/%m/%Y')

# Extract the date from the first row
d19_date = dates.loc[0, "Event start date"]
d21_date = dates.loc[1, "Event start date"]
gp21_date = dates.loc[2, "Event start date"]
mse21_date = dates.loc[3, "Event start date"]
np21_date = dates.loc[4, "Event start date"]
srm22_date = dates.loc[5, "Event start date"]
srm23_date = dates.loc[6, "Event start date"]

# Format the data
d19["Created Date"] = pd.to_datetime(d19["Created Date"], format='%d/%m/%Y')
d19['DaysBeforeEvent'] = (d19_date - d19['Created Date']).dt.days
d21["Created Date"] = pd.to_datetime(d21["Created Date"], format='%d/%m/%Y')
d21['DaysBeforeEvent'] = (d21_date - d21['Created Date']).dt.days
gp21["Created Date"] = pd.to_datetime(gp21["Created Date"], format='%d/%m/%Y')
gp21['DaysBeforeEvent'] = (gp21_date - gp21['Created Date']).dt.days
mse21["Created Date"] = pd.to_datetime(mse21["Created Date"], format='%d/%m/%Y')
mse21['DaysBeforeEvent'] = (mse21_date - mse21['Created Date']).dt.days
np21["Created Date"] = pd.to_datetime(np21["Created Date"], format='%d/%m/%Y')
np21['DaysBeforeEvent'] = (np21_date - np21['Created Date']).dt.days
srm22["Created Date"] = pd.to_datetime(srm22["Created Date"], format='%d/%m/%Y')
srm22['DaysBeforeEvent'] = (srm22_date - srm22['Created Date']).dt.days
srm23["Created Date"] = pd.to_datetime(srm23["Created Date"], format='%d/%m/%Y')
srm23['DaysBeforeEvent'] = (srm23_date - srm23['Created Date']).dt.days

# Combine the target audience into one pandas DataFrame
df = pd.concat([d19, d21, gp21, mse21, np21, srm22, srm23], ignore_index=True)
df = df[df['DaysBeforeEvent'] >= 0]

df['Encoded'] = df['Attendee Status'].astype('category').cat.codes

not_missing = df.dropna(subset= ['Attended']) # drops all the missing values in that column
missing = df[df['Attended'].isna()] # checks for the missing values in the column
not_missing['AttendedEncoded'] = not_missing['Attended'].astype('category').cat.codes # changes yes/no values to numerical values

# Split into X and y variables

X = not_missing[['DaysBeforeEvent']]
y = not_missing['AttendedEncoded']

# Split into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X_train, y_train)

# Predict and check the accuracy score

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Before resampling:")
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'CM Score: {cm}')

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X, y)

# Predict the missing values
missing_values = missing[['DaysBeforeEvent']]
predicted_values = knn.predict(missing_values)

# Map the predicted endoded data back to the Attended Status
encode_map = dict(enumerate(not_missing['Attended'].astype('category').cat.categories))
predicted_values_map = [encode_map[pred] for pred in predicted_values]

# Update datset with predicted values
missing_indices = missing.index
df.loc[missing_indices, 'Attended'] = predicted_values_map
df = df.sort_values(by=["File", "Row"])

# Specify the path and filename for the CSV file
csv_path = 'Imputed Result.csv'

# Write the DataFrame to a CSV file
df.to_csv(csv_path, index=False)

# Resample the dataset to balance the target variables
smote = SMOTE()
# X_resampled, y_resampled = smote.fit_resample(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X_train, y_train)

# Predict and check the accuracy score

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("\nAfter resampling:")
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'CM Score: {cm}')