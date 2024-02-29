"""This code performs ordinary least squares (OLS) linear regression on the dataframes imported from the data_prep.py file. A function is made for the OLS model and
when a dataframe is passed to the function, the model is fit to the data and the results are printed. The results are also saved to a csv file in a subfolder of the
code directory. The function also plots the actual and predicted booking counts for the test set and highlights the weeks where the advertisement was active. Cross-validation
is performed inside the function to ensure the model is not overfitting the data and the results are averaged over the 10 folds. 

The python libraries used in this code and their corresponding versions are:
matplotlib                   3.8.0
numpy                        1.26.0
scikit-learn                 1.4.0
"""

from data_prep import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


"""function to perform OLS linear regression on the day-wise dataframes and plot the results"""
df_regression_results = pd.DataFrame() # Create a new dataframe to store the results of the linear regression
def ols_linear_regression_days (dataframe, audience_name):
    global df_regression_results

    max_days = dataframe["Days to Event"].max()
    dataframe["Days to Event"] = max_days - dataframe["Days to Event"] # Reverse the order of the days to event to start from 0 to enable fit_intercept = False

    X = dataframe[["Days to Event", "Advertisement Status"]] # Create a feature matrix
    y = dataframe["Booking Count"] # Create a target vector

    regr = linear_model.LinearRegression(fit_intercept = False) # Create a linear regression model

    kf = KFold(n_splits = 10, shuffle = True, random_state = 1) # Create a 10-fold cross-validation object

    mse_scores = []
    r2_scores = []
    variance_scores = []
    relative_mse_scores = []
    mae_scores = []
    relative_mae_scores = []
    ad_effects = []

    for train_index, test_index in kf.split(X): # for loop to perform cross-validation
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_test_df = pd.DataFrame(X_test, columns = ["Days to Event", "Advertisement Status"]) # Create a new dataframe for the test set
        y_test_series = pd.Series(y_test, name = "Booking Count") # Create a new series for the test set
        test_df = pd.concat([X_test_df.reset_index(drop = True), y_test_series.reset_index(drop = True)], axis = 1) # Combine the testing data into a single dataframe
        test_df = test_df.sort_values(by = "Days to Event", ascending = True) # Sort the testing data by days to event in ascending order 

        X_test_sorted = test_df[["Days to Event", "Advertisement Status"]] # Sorted feature matrix 
        y_test_sorted = test_df["Booking Count"] # Sorted target vector
        X_ad_active = X_test_sorted[X_test_sorted["Advertisement Status"] == 1] # Create a new dataframe with ad status = 1
        X_ad_inactive = X_test_sorted[X_test_sorted["Advertisement Status"] == 0] # Create a new dataframe with ad status = 0

        regr.fit(X_train, y_train) # Fit the model to the training data
        pred = regr.predict(X_test_sorted) # Make predictions using the test set
        pred_ad_inactive = regr.predict(X_ad_inactive) # Make predictions using the test set with ad status = 0
        if X_ad_active.empty == False:
            pred_ad_active = regr.predict(X_ad_active) # Make predictions using the test set with ad status = 1
        else:
            pred_ad_active = pred_ad_inactive

        mse = mean_squared_error(y_test_sorted, pred) # Calculate the mean squared error
        r2 = r2_score(y_test_sorted, pred) # Calculate the r-squared score
        variance = np.var(y_test_sorted) # Calculate the variance of the test data
        relative_mse = mse / variance # Calculate the relative mean squared error
        mae = mean_absolute_error(y_test_sorted, pred) # Calculate the mean absolute error
        range_y = y_test_sorted.max() - y_test_sorted.min() # Calculate the range of the test data
        relative_mae = mae / range_y # Calculate the relative mean absolute error
        ad_effect = (pred_ad_active.mean() - pred_ad_inactive.mean()) / pred_ad_inactive.mean() # Calculate the effect of advertisements on the predicted booking count

        # Append the metrics to the lists
        mse_scores.append(mse) 
        r2_scores.append(r2)
        variance_scores.append(variance)
        relative_mse_scores.append(relative_mse)
        mae_scores.append(mae)
        relative_mae_scores.append(relative_mae)
        ad_effects.append(ad_effect)
    
    # Find the mean value of the metrics
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    avg_relative_mse = np.mean(relative_mse_scores)
    avg_mae = np.mean(mae_scores)
    avg_relative_mae = np.mean(relative_mae_scores)
    avg_ad_effect = np.mean(ad_effects)

    results = {
            "Target Audience": audience_name,
            "Mean Squared Error": round(avg_mse, 2),
            "R-squared": round(avg_r2, 2),
            "% R-squared": round(avg_r2 * 100, 2),
            "% Relative MSE": round(avg_relative_mse * 100, 2),
            "Mean Absolute Error": round(avg_mae, 2),
            "Relative MAE": round(avg_relative_mae, 2),
            "% Accuracy": round((1 - avg_relative_mae) * 100, 2),
            "% Booking Increase due to Advertisement": round(avg_ad_effect * 100, 2) 
        } # Create a dictionary with important metrics of the regression model
    
    X_trainplot, X_testplot, y_trainplot, y_testplot = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify = X["Advertisement Status"]) # Split the data into training and testing sets for plotting
    X_testplot_df = pd.DataFrame(X_testplot, columns = ["Days to Event", "Advertisement Status"]) # Create a new dataframe for the test feature matrix
    y_testplot_series = pd.Series(y_testplot, name = "Booking Count") # Create a new series for the test target variable
    testplot_df = pd.concat([X_testplot_df.reset_index(drop = True), y_testplot_series.reset_index(drop = True)], axis = 1) # Combine the testing data into a single dataframe
    testplot_df = testplot_df.sort_values(by = "Days to Event", ascending = True) # Sort the testing data by days to event in ascending order 

    X_testplot_sorted = testplot_df[["Days to Event", "Advertisement Status"]] # Sorted feature matrix 
    y_testplot_sorted = testplot_df["Booking Count"] # Sorted target vector

    regr.fit(X_trainplot, y_trainplot) # Fit the model to the training data
    predplot = regr.predict(X_testplot_sorted) # Make predictions using the test set

    predplot_df = X_testplot_sorted.copy() # Create a new dataframe for the predicted data
    predplot_df["Predicted Count"] = predplot # Add a new column to the dataframe with the predicted data
    ads_active_df = predplot_df[predplot_df["Advertisement Status"] == 1] # Create a new dataframe with only the advertisement weeks 

    plt.figure(figsize = (10, 6)) # Create a new figure
    plt.scatter(X_testplot_sorted["Days to Event"], y_testplot_sorted, color = "black", label = "Actual") # Create a scatter plot of the actual data
    plt.plot(X_testplot_sorted["Days to Event"], predplot, color = "blue", linewidth = 1, label = "Predicted") # Create a line plot of the predicted data
    plt.scatter(ads_active_df["Days to Event"], ads_active_df["Predicted Count"], color = "red", label = "Advertisement Active") # Create a scatter plot of the advertisement weeks
    plt.title("OLS Linear Regression for " + audience_name) # Set the title of the plot
    plt.xlabel("Day #") # Set the x-axis label
    plt.ylabel("Booking Count") # Set the y-axis label
    plt.legend() # Add a legend to the plot
    plt.show() # Display the plot

    new_row = pd.DataFrame(results, index = [0]) # setting index to 0 is important when assigning a single row to a dataframe
    df_regression_results = pd.concat([df_regression_results, new_row], ignore_index = True) # Add the results to the dataframe



ols_linear_regression_days(df_ITM_days, "IT Managers")
ols_linear_regression_days(df_PM_days, "Property Managers")
ols_linear_regression_days(df_EPM_days, "Education Property Managers")
ols_linear_regression_days(df_EM_days, "Education Managers")
ols_linear_regression_days(df_other_days, "Other Target Audiences")

print(df_regression_results)


"""Save the results to a csv file in subfolder of code directory"""
subfolder = "Prediction Model Results" # Name of the subfolder to contain the results
full_path = os.path.join(path, subfolder) # Create the full path to the subfolder
if not os.path.exists(full_path):
    os.makedirs(full_path) # Create a subfolder in the code directory for results if it does not exist
try:
    df_regression_results.to_csv(os.path.join(full_path, "linear_regression_results_multivariate.csv")) # Save the results to a csv file in the subfolder
except PermissionError:
    print("\nUnable to write to file. Please make sure the file is not open in another program.\n")