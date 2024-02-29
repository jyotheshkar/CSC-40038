from DataPrep import *
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV, ElasticNet

"""function to perform elastic net regression on the day-wise dataframes"""
def elastic_net_data (dataframe):
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
    X_ad_active = X_test_sorted[X_test_sorted["Advertisement Status"] == 1] # Feature matrix for days with advertisements
    X_ad_inactive = X_test_sorted[X_test_sorted["Advertisement Status"] == 0] # Feature matrix for days without advertisements

    regr = ElasticNetCV(l1_ratio = [.1, .3, .5, .7, .9, .95, .99], cv = 10, random_state = 1, fit_intercept = False) # Create an elastic net model

    regr.fit(X_train, y_train) # Fit the model to the training data
    pred_ad_inactive = regr.predict(X_ad_inactive) # Make predictions for days without advertisements
    if X_ad_active.empty == False:
        pred_ad_active = regr.predict(X_ad_active)
    else:
        pred_ad_active = pred_ad_inactive

    alpha = regr.alpha_ # Get the optimal alpha value
    l1_ratio = regr.l1_ratio_ # Get the optimal l1 ratio
    ad_effect = (pred_ad_active.mean() - pred_ad_inactive.mean()) / pred_ad_inactive.mean() # Calculate the effect of advertisements on the predicted booking count

    return X, y, l1_ratio, alpha, ad_effect


# add selected audience to invoke particular elastic_net function. df_other_days, df_ITM_days, df_PM_days, df_EPM_days, df_EM_days are the dataframes to be used depending on the audience option
def prediction(list):
    audience_dataset = {1: df_ITM_days, 2: df_PM_days, 3: df_EPM_days, 4: df_EM_days, 5: df_other_days}
    start_date, event_date, current_date, current_booking_count, target_audience = list
    start_date = datetime.strptime(start_date, "%d/%m/%Y")
    current_date = datetime.strptime(current_date,
                                     "%d/%m/%Y")  # Ask the user for the current date
    event_date = datetime.strptime(event_date, "%d/%m/%Y") - timedelta(
        days=1)  # Ask the user for the event date
    date_range = pd.date_range(start=start_date,
                               end=event_date)  # Create a date range from the start date to the event date
    input_df = pd.DataFrame(date_range, columns=["Date"])  # Create a new dataframe with the date range
    input_df["Days to Event"] = (pd.to_datetime(event_date, dayfirst=True) - input_df[
        "Date"]).dt.days  # Add a new column to the dataframe showing the days left before the event
    input_df["Day #"] = (input_df["Days to Event"].max() + 1) - input_df[
        "Days to Event"]  # Reverse the order of the days to event column to start from 1 for the model to work properly
    input_df["Advertisement Status"] = input_df["Days to Event"].apply(lambda
                                                                           x: 0)  # Add a new column to the dataframe to indicate advertisement status. Assuming that no ads were launched before current date
    audience = audience_dataset[target_audience]

    train_x_final, train_y_final, l1_ratio, alpha, ad_effect = elastic_net_data(audience)
    target_days_df = input_df[input_df[
                                  "Date"] >= current_date].copy()  # Create a new dataframe with the days from the current date to the event date
    target_days_df.reset_index(drop=True, inplace=True)  # Reset the index of the dataframe
    regr_final = ElasticNet(l1_ratio=l1_ratio, alpha=alpha, fit_intercept=False)  # Create an elastic net model
    regr_final.fit(train_x_final, train_y_final)  # Fit the model to the training data
    pred_final = regr_final.predict(
        target_days_df[["Day #", "Advertisement Status"]])  # Make predictions using the testing dataset
    target_days_df.loc[:, "Predicted Count"] = pred_final  # Add a new column for the predicted booking count
    target_days_df.loc[:, "Predicted Count with Ad"] = pred_final  # Add a new column for the predicted booking count
    target_days_df.loc[target_days_df.index[:7], "Predicted Count with Ad"] *= (
                1 + ad_effect)  # Update the ad status to 1 for the next 7 days
    total_event_bookings = int(round(current_booking_count + target_days_df["Predicted Count"].sum(),
                                     0))  # Calculate the total predicted bookings for the event
    total_event_bookings_with_ad = int(round(current_booking_count + target_days_df["Predicted Count with Ad"].sum(),
                                             0))  # Calculate the total predicted bookings for the event
    return total_event_bookings, total_event_bookings_with_ad
    # print("The total predicted bookings for the event are:", total_event_bookings) # Print the total predicted bookings for the event