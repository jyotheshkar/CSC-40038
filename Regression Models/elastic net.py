"""This code performs elastic net regression on the dataframes imported from the data_prep.py file. A function is made for the elastic net model and
when a dataframe is passed to the function, the model is fit to the data and the results are printed. The results are also saved to a csv file in a subfolder of the
code directory. The function also plots the actual and predicted booking counts for the test set and highlights the weeks where the advertisement was active. Cross-validation
is performed inside the function to ensure the model is not overfitting the data and the hyperparameters are optimized. 

The python libraries used in this code and their corresponding versions are:
matplotlib                   3.8.0
numpy                        1.26.0
scikit-learn                 1.4.0"""

from data_prep import * 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


"""function to perform elastic net regression on the day-wise dataframes"""
df_elasticnet_results = pd.DataFrame() # Create a new dataframe to store the results of the elastic net model
def elastic_net (dataframe, audience_name):
    global df_elasticnet_results

    max_days = dataframe["Days to Event"].max()
    dataframe["Day #"] = (max_days + 1) - dataframe["Days to Event"] # Reverse the order of the days to event column to start from 0 to enable fit_intercept = False

    X = dataframe[["Day #", "Advertisement Status"]] # Create a feature matrix
    y = dataframe["Booking Count"] # Create a target vector for booking count

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify = X["Advertisement Status"]) # Split the data into training and testing sets. Stratify ensures that the train and test sets have similar proportions of days with and without advertisements.

    X_test_df = pd.DataFrame(X_test, columns = ["Day #", "Advertisement Status"]) # Create a new dataframe for the test set
    y_test_series = pd.Series(y_test, name = "Booking Count") # Create a new series for the test set
    test_df = pd.concat([X_test_df.reset_index(drop = True), y_test_series.reset_index(drop = True)], axis = 1) # Combine the testing data into a single dataframe
    test_df = test_df.sort_values(by = "Day #", ascending = True) # Sort the testing data by days to event in ascending order 

    X_test_sorted = test_df[["Day #", "Advertisement Status"]] # Sorted feature matrix 
    y_test_sorted = test_df["Booking Count"] # Sorted target vector
    X_ad_active = X_test_sorted[X_test_sorted["Advertisement Status"] == 1] # Feature matrix for days with advertisements
    X_ad_inactive = X_test_sorted[X_test_sorted["Advertisement Status"] == 0] # Feature matrix for days without advertisements

    regr = ElasticNetCV(l1_ratio = [.1, .3, .5, .7, .9, .95, .99], cv = 10, random_state = 1, fit_intercept = False) # Create an elastic net model with cross-validation

    regr.fit(X_train, y_train) # Fit the model to the training data    
    pred = regr.predict(X_test_sorted) # Make predictions using the testing dataset
    pred_ad_inactive = regr.predict(X_ad_inactive) # Make predictions for days without advertisements
    if X_ad_active.empty == False:
        pred_ad_active = regr.predict(X_ad_active)
    else:
        pred_ad_active = pred_ad_inactive 

    predplot_df = X_test_sorted.copy() # Create a new dataframe for the predictions
    predplot_df["Predicted Count"] = pred # Add a new column for the predicted booking count
    ads_active_df = predplot_df[predplot_df["Advertisement Status"] == 1] # Create a new dataframe for days with advertisements

    mse = mean_squared_error(y_test_sorted, pred) # Calculate the mean squared error
    r2 = r2_score(y_test_sorted, pred) # Calculate the coefficient of determination (r-squared)
    alpha = regr.alpha_ # Get the optimal alpha value
    l1_ratio = regr.l1_ratio_ # Get the optimal l1 ratio
    variance = np.var(y_test_sorted) # Calculate the variance of the test data
    relative_mse = mse / variance # Calculate the relative mean squared error
    mae = mean_absolute_error(y_test_sorted, pred) # Calculate the mean absolute error
    range_y = y_test_sorted.max() - y_test_sorted.min() # Calculate the range of the test data
    relative_mae = mae / range_y # Calculate the relative mean absolute error
    ad_effect = (pred_ad_active.mean() - pred_ad_inactive.mean()) / pred_ad_inactive.mean() # Calculate the effect of advertisements on the predicted booking count

    results = {
            "Target Audience": audience_name,
            "Alpha": round(alpha, 2),
            "L1 Ratio": l1_ratio,
            "Mean Squared Error": round(mse, 2),
            "R-squared": round(r2, 2),
            "% R-squared": round(r2 * 100, 2),
            "% Relative MSE": round(relative_mse * 100, 2), 
            "Mean Absolute Error": round(mae, 2),
            "Relative MAE": round(relative_mae, 2),
            "% Accuracy": round((1 - relative_mae) * 100, 2),
            "% Booking Increase per Day in Advertisement Weeks": round(ad_effect * 100, 2) 
        } # Create a dictionary with important metrics of the elastic net model

    new_row = pd.DataFrame(results, index = [0]) # setting index to 0 is important when assigning a single row to a dataframe
    df_elasticnet_results = pd.concat([df_elasticnet_results, new_row], ignore_index = True) # Add the results to the dataframe

    plt.figure(figsize = (10, 6)) # Create a new figure    
    plt.scatter(X_test_sorted["Day #"], y_test_sorted, color = "black", label = "Actual") # Create a scatter plot of the actual booking count
    plt.plot(X_test_sorted["Day #"], pred, color = "blue", linewidth = 1, label = "Predicted") # Create a line plot of the predicted booking count
    plt.scatter(ads_active_df["Day #"], ads_active_df["Predicted Count"], color = "red", label = "Advertisement Active") # Create a scatter plot of days with advertisements
    plt.title("Elastic Net Regression for " + audience_name) # Set the title of the plot
    plt.xlabel("Day #") # Set the x-axis label
    plt.ylabel("Booking Count") # Set the y-axis label
    plt.legend() # Add a legend to the plot
    plt.show() # Display the plot



elastic_net(df_ITM_days, "IT Managers")
elastic_net(df_PM_days, "Property Managers")
elastic_net(df_EPM_days, "Education Property Managers")
elastic_net(df_EM_days, "Education Managers")
elastic_net(df_other_days, "Other Target Audiences")

print(df_elasticnet_results)


"""Save the results to a csv file in subfolder of code directory"""
subfolder = "Prediction Model Results" # Name of the subfolder to contain the results
full_path = os.path.join(path, subfolder) # Create the full path to the subfolder
if not os.path.exists(full_path):
    os.makedirs(full_path) # Create a subfolder in the code directory for results if it does not exist
try:
    df_elasticnet_results.to_csv(os.path.join(full_path, "elastic_net_results.csv")) # Save the results to a csv file in the subfolder
except PermissionError:
    print("\nUnable to write to file. Please make sure the file is not open in another program.\n")