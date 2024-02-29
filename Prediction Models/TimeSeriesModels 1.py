"""
The code below performs a Time Series Analysis on the events data.
The aim is to predict the number of registrations at a specific time
The data is preprocessed and three Time series models are used for the analysis, specifically:

 ** ARIMA
 ** SARIMAX
 ** Exponential Smoothing.

 Python version 3.11
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

from pmdarima import auto_arima

#Read a CSV file, preprocess it, and filter data based on the target date.
def read_and_preprocess(file_path, event_code, target_date_str):

    # Read CSV file skipping the first row (header=1)
    df = pd.read_csv(file_path, header=1)
    
    # Convert 'Created Date' column to datetime format
    df['Created Date'] = pd.to_datetime(df['Created Date'], dayfirst=True)

    # Add a new column 'Event' with the provided event code
    df['Event'] = event_code

    # Convert target date string to datetime
    target_date = pd.to_datetime(target_date_str, dayfirst=True)

    # Calculate T_minus (days until the event), excluding the target date
    df['T_minus'] = (target_date - df['Created Date']).dt.days - 1 

    # Filter out rows where T_minus is negative (i.e., created date is after the target date)
    df = df[df['T_minus'] >= 0]

    return df

# Combine multiple DataFrames into a single DataFrame.
def combine_datasets(*datasets):
    return pd.concat(datasets, ignore_index=True)

'''def plot_registrations_countdown(combined_data,tile):
    t_minus_registrations = combined_data.groupby('T_minus').size()
    plt.figure(figsize=(20, 6))
    plt.plot(t_minus_registrations.index, t_minus_registrations.values, marker='o', linestyle='-', color='b')
    plt.title(tile)
    plt.xlabel('Days Until Event (T-minus)')
    plt.ylabel('Number of Registrations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()'''

# Calculate daily registrations, Z-score, and advertisement flag.
def calculate_daily_registrations(combined_data):
    # Calculate total registrations
    agg_operations = {
        'Total Registrations': ('Created Date', 'size'),
        'T_minus': ('T_minus', 'first')  # Assumes each date has a unique, correct T_minus value
    }
    daily_registrations = combined_data.groupby('Created Date').agg(**agg_operations).reset_index()
    
    # Calculate Z-Score
    daily_registrations['Z-Score'] = ((daily_registrations['Total Registrations'] - daily_registrations['Total Registrations'].mean()) 
                                      / daily_registrations['Total Registrations'].std())
    
    # Determine if advertisement was done based on Z-Score
    daily_registrations['Advertisement'] = (daily_registrations['Z-Score'] > 2).astype(int)

    return daily_registrations
    
#  Prepare data for forecasting by filling missing T_minus days and filling NaN values.
def prepare_for_forecasting_with_t_minus(daily_registrations):
    # Sort 'daily_registrations' by 'T_minus'
    daily_registrations_sorted = daily_registrations.sort_values(by='T_minus')
    
    # Find the min and max 'T_minus' values to create a complete range
    min_t_minus = daily_registrations_sorted['T_minus'].min()
    max_t_minus = daily_registrations_sorted['T_minus'].max()
    
    # Create a DataFrame representing every 'T_minus' day in the inclusive range
    complete_t_minus_range = pd.DataFrame(range(min_t_minus, max_t_minus + 1), columns=['T_minus'])
    
    # Merge with the actual registration data, ensuring every 'T_minus' day is represented
    complete_reg = pd.merge(complete_t_minus_range, daily_registrations_sorted, on='T_minus', how='left', sort=True)
    
    # Fill missing values for days without registrations with zeros
    complete_reg['Total Registrations'].fillna(0, inplace=True)
    complete_reg['Z-Score'].fillna(0, inplace=True)
    complete_reg['Advertisement'].fillna(0, inplace=True)
    
    # Set 'T_minus' as the DataFrame index
    complete_reg = complete_reg.set_index('T_minus')
    
    return complete_reg


def split_data_for_forecasting(complete_reg, split_ratio=0.8):

    # Ensure that the lastest days ie days closer to the event are used for testing and those further away are used for training.
    complete_reg_sorted = complete_reg.sort_values(by='T_minus', ascending=True)
    
    # Determine the split point
    total_days = len(complete_reg_sorted)
    test_size = int(total_days * (1 - split_ratio))
    #train_size = total_days - test_size

    train = complete_reg_sorted.iloc[test_size:]
    test = complete_reg_sorted.iloc[:test_size]

    # Extract the endogenous and exogenous variables for the training and testing sets
    train_endog = train['Total Registrations']
    train_exog = train[['Advertisement']]
    test_endog = test['Total Registrations']
    test_exog = test[['Advertisement']]

    return train_endog, train_exog, test_endog, test_exog

# Auto-arima to get the best parameters for the model
def auto_arima_forecasting(endog, exog):
    auto_model = auto_arima(endog, exogenous=exog, seasonal=True, m=7, stepwise=True, trace=True, error_action='ignore', suppress_warnings=True)
    #print(auto_model.summary())
    return auto_model.order, auto_model.seasonal_order

# Performs SARIMAX forecasting
def sarimax_forecasting(train_endog, train_exog, test_exog, order, seasonal_order):
    model = SARIMAX(train_endog, order=order, seasonal_order=seasonal_order, exog=train_exog)
    results = model.fit()
    #print(results.summary())
    forecast_results = results.get_forecast(steps=len(test_exog), exog=test_exog)
    return forecast_results.predicted_mean, forecast_results.conf_int()


def plot_forecast_vs_actuals(train, test, forecast_values, conf_int):
    # Calculate the number of days for the x-axis in reverse
    day_count_train = range(len(train))
    day_count_test = range(len(train) + len(test) - 1, len(train) - 1, -1)  # Reverse the range for test dataset

    # Reverse the 'Total Registrations' series
    train_registrations_reversed = train['Total Registrations'].iloc[::-1].values
    test_registrations_reversed = test['Total Registrations'].iloc[::-1].values
    forecast_values_reversed = forecast_values[::-1]
    conf_int_reversed_lower = conf_int.iloc[::-1, 0]
    conf_int_reversed_upper = conf_int.iloc[::-1, 1]

    # Plotting
    plt.figure(figsize=(30, 6))

    # Plot the observed values with reversed day count as x-axis
    plt.plot(list(day_count_train), train_registrations_reversed, label='Train', color='blue')
    plt.plot(list(day_count_test), test_registrations_reversed, label='Test', color='green')

    # Plot the forecasted values
    plt.plot(list(day_count_test), forecast_values_reversed, label='Forecast', color='red')

    # Plot the confidence interval
    plt.fill_between(list(day_count_test), conf_int_reversed_lower, conf_int_reversed_upper, color='pink', alpha=0.3)

    # Reverse the x-axis tick labels and remove '.0'
    #x_ticks = plt.gca().get_xticks()
    #plt.gca().set_xticklabels([int(x) for x in reversed(x_ticks)])

    # Set labels and title
    plt.xlabel('Day Count')
    plt.ylabel('Total Registrations')
    plt.title('SARIMAX Model Forecast vs Actuals')

    plt.legend()
    plt.show()

# Using ARIMA

# Function to split the dataset into training and testing sets
def split_time_series_data(complete_reg, split_ratio=0.8):

    # Ensure that the lastest days ie days closer to the event are used for testing and those further away are used for training.
    complete_reg_sorted = complete_reg.sort_values(by='T_minus', ascending=True)
    
    # Determine the split point
    total_days = len(complete_reg_sorted)
    test_size = int(total_days * (1 - split_ratio))
    #train_size = total_days - test_size

    train = complete_reg_sorted.iloc[test_size:]
    test = complete_reg_sorted.iloc[:test_size]

    return train, test


# Function to find the best ARIMA model order
def find_best_arima_order(endog):
    auto_model = auto_arima(endog, seasonal=False, stepwise=True, trace=True, error_action='ignore', suppress_warnings=True, start_p=1, start_q=1, max_p=5, max_q=5, d=None, test='adf', start_P=0, D=0)
    print("Best ARIMA order:", auto_model.order)
    return auto_model.order

# Function to forecast using ARIMA
def arima_forecasting(endog, order):
    model = ARIMA(endog, order=order)
    fitted_model = model.fit()
    #print(fitted_model.summary())
    
    forecast_results = fitted_model.get_forecast(steps=len(test_endog_ar))
    forecast_values_arima = forecast_results.predicted_mean
    conf_int = forecast_results.conf_int()
    return forecast_values_arima, conf_int

# Using Exponential Smoothing
def exp_smoothing(endog, seasonal_periods = 7):
    model = ExponentialSmoothing(endog, trend = 'add', seasonal = 'add', seasonal_periods = seasonal_periods)
    fitted = model.fit()
    forecast_values_exp = fitted.forecast(steps = len(test_endog_exp))
    return forecast_values_exp


# IT Managers SARIMAX

# D19 & D21
d19 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\D19.csv", 'D19', '19/11/2019')
d21 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\D21.csv", 'D21', '09/12/2021')

# Combine datasets
combined_it_managers = combine_datasets(d19, d21)

# Proceed with analysis and plotting
#plot_registrations_countdown(combined_it_managers, 'Registrations Countdown to IT Managers Conference')
daily_registrations_it = calculate_daily_registrations(combined_it_managers)
complete_reg_it = prepare_for_forecasting_with_t_minus(daily_registrations_it)

# Split the data into training and testing sets
train_endog, train_exog, test_endog, test_exog = split_data_for_forecasting(complete_reg_it)

# Use Auto ARIMA to find the optimal SARIMAX parameters
order, seasonal_order = auto_arima_forecasting(train_endog, train_exog)

# Fit the SARIMAX model using the optimal parameters
forecast_values, conf_int = sarimax_forecasting(train_endog, train_exog, test_exog, order, seasonal_order)

test_df = test_exog.copy()
test_df['Total Registrations'] = test_endog.values

# Plot the forecast against actual values
plot_forecast_vs_actuals(complete_reg_it.iloc[:len(train_endog)], test_df, forecast_values, conf_int)

# IT Managers ARIMA 
train, test = split_time_series_data(complete_reg_it, split_ratio=0.8)

# For ARIMA, focusing on the endogenous variable only
train_endog_ar = train['Total Registrations']
test_endog_ar = test['Total Registrations']

# Finding the best ARIMA order
order = find_best_arima_order(train_endog_ar)

# Forecasting with ARIMA
forecast_values_arima, conf_int = arima_forecasting(train_endog_ar, order)

# Plotting the results
plt.figure(figsize=(30, 6))
plt.plot(train.index, train_endog_ar, label='Training Data')
plt.plot(test.index, test_endog_ar, label='Actual Test Data')
plt.plot(test.index, forecast_values_arima, label='ARIMA Forecast')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
#Reverse the x-axis tick labels and remove '.0'
x_ticks = plt.gca().get_xticks()
plt.gca().set_xticklabels([int(x) for x in reversed(x_ticks)])

plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Registrations')
plt.title('ARIMA Forecast vs Actuals')
plt.show()

# IT Managers EXP
train, test = split_time_series_data(complete_reg_it, split_ratio=0.8)

train_endog_exp= train['Total Registrations']
test_endog_exp = test['Total Registrations']

forecast_values_exp = exp_smoothing(train_endog_exp)

# Plotting the results
plt.figure(figsize=(30, 6))
plt.plot(train.index, train_endog_exp, label='Training Data')
plt.plot(test.index, test_endog_exp, label='Actual Test Data')
plt.plot(test.index, forecast_values_exp, label='Exponential Smoothing Forecast', color='red')
#Reverse the x-axis tick labels and remove '.0'
x_ticks = plt.gca().get_xticks()
plt.gca().set_xticklabels([int(x) for x in reversed(x_ticks)])
plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Registrations')
plt.title('Exponential Smoothing Forecast vs Actuals')
plt.show()

# Calculate RMSE and MAE for ARIMA
rmse_arima = np.sqrt(mean_squared_error(test_endog_ar, forecast_values_arima))
mae_arima = mean_absolute_error(test_endog_ar, forecast_values_arima)

# Calculate RMSE and MAE for SARIMAX
rmse_sarimax = np.sqrt(mean_squared_error(test_endog, forecast_values))
mae_sarimax = mean_absolute_error(test_endog, forecast_values)

#Calculate RMSE and MAE for Exponential Smoothing
rmse_exp = np.sqrt(mean_squared_error(test_endog_exp, forecast_values_exp))
mae_exp = mean_squared_error(test_endog_exp, forecast_values_exp)

# Find the maximum value across all test datasets
max_value = max(test_endog_ar.max(), test_endog.max(), test_endog_exp.max())

# Find the minimum value across all test datasets
min_value = min(test_endog_ar.min(), test_endog.min(), test_endog_exp.min())

# Calculate the range
value_range = max_value - min_value

# Calculate RMSE as a percentage of the value range
rmse_pct_arima = (rmse_arima / value_range) * 100
rmse_pct_sarimax = (rmse_sarimax / value_range) * 100
rmse_pct_exp = (rmse_exp / value_range) * 100

# Calculate MAE as a percentage of the value range
mae_pct_arima = (mae_arima / value_range) * 100
mae_pct_sarimax = (mae_sarimax / value_range) * 100
mae_pct_exp = (mae_exp / value_range) * 100

# Print the results
print(f"ARIMA RMSE (IT): {rmse_arima:.2f}, RMSE %:  {rmse_pct_arima:.2f}%,  MAE: {mae_arima:.2f}, MAE %: {mae_pct_arima:.2f}%")
print(f"SARIMAX RMSE (IT): {rmse_sarimax:.2f},RMSE %: {rmse_pct_sarimax: .2f}%, MAE: {mae_sarimax:.2f}, MAE %: {mae_pct_sarimax: .2f}%")
print(f"Exponential Smoothing RMSE (IT): {rmse_exp:.2f}, RMSE %: {rmse_pct_exp:.2f}, % MAE: {mae_exp:.2f}, MAE %: {mae_pct_exp:.2f}%")

# Property Managers SARIMAX:
# GP21 and NP21
gp21 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\GP21.csv", 'GP21', '22/04/2021')
np21 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\NP21.csv", 'NP21', '09/11/2021')

# Combine datasets 
combined_property_managers = combine_datasets(gp21, np21)

# Proceed with analysis and plotting
#plot_registrations_countdown(combined_property_managers, 'Registrations Countdown to Property Managers Conference')
daily_registrations_prop = calculate_daily_registrations(combined_property_managers)
complete_reg_prop = prepare_for_forecasting_with_t_minus(daily_registrations_prop)

# Split the data into training and testing sets
train_endog, train_exog, test_endog, test_exog = split_data_for_forecasting(complete_reg_prop)

# Use Auto ARIMA to find the optimal SARIMAX parameters
order, seasonal_order = auto_arima_forecasting(train_endog, train_exog)

# Fit the SARIMAX model using the optimal parameters
forecast_values, conf_int = sarimax_forecasting(train_endog, train_exog, test_exog, order, seasonal_order)

test_df = test_exog.copy()
test_df['Total Registrations'] = test_endog.values

# Plot the forecast against actual values
plot_forecast_vs_actuals(complete_reg_prop.iloc[:len(train_endog)], test_df, forecast_values, conf_int)

# Property Managers ARIMA
train, test = split_time_series_data(complete_reg_prop, split_ratio=0.8)

# For ARIMA, focusing on the endogenous variable only
train_endog_ar = train['Total Registrations']
test_endog_ar = test['Total Registrations']

# Finding the best ARIMA order
order = find_best_arima_order(train_endog_ar)

# Forecasting with ARIMA
forecast_values_arima, conf_int = arima_forecasting(train_endog_ar, order)

# Plotting the results
plt.figure(figsize=(30, 6))
plt.plot(train.index, train_endog_ar, label='Training Data')
plt.plot(test.index, test_endog_ar, label='Actual Test Data')
plt.plot(test.index, forecast_values_arima, label='ARIMA Forecast')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
#Reverse the x-axis tick labels and remove '.0'
x_ticks = plt.gca().get_xticks()
plt.gca().set_xticklabels([int(x) for x in reversed(x_ticks)])
plt.legend()

plt.xlabel('Date')
plt.ylabel('Total Registrations')
plt.title('ARIMA Forecast vs Actuals')
plt.show()

# Property Managers EXP
train, test = split_time_series_data(complete_reg_prop, split_ratio=0.8)

train_endog_exp= train['Total Registrations']
test_endog_exp = test['Total Registrations']

forecast_values_exp = exp_smoothing(train_endog_exp)

# Plotting the results
plt.figure(figsize=(30, 6))
plt.plot(train.index, train_endog_exp, label='Training Data')
plt.plot(test.index, test_endog_exp, label='Actual Test Data')
plt.plot(test.index, forecast_values_exp, label='Exponential Smoothing Forecast', color='red')
#Reverse the x-axis tick labels and remove '.0'
x_ticks = plt.gca().get_xticks()
plt.gca().set_xticklabels([int(x) for x in reversed(x_ticks)])

plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Registrations')
plt.title('Exponential Smoothing Forecast vs Actuals')
plt.show()

# Calculate RMSE and MAE for ARIMA
rmse_arima = np.sqrt(mean_squared_error(test_endog_ar, forecast_values_arima))
mae_arima = mean_absolute_error(test_endog_ar, forecast_values_arima)

# Calculate RMSE and MAE for SARIMAX
rmse_sarimax = np.sqrt(mean_squared_error(test_endog, forecast_values))
mae_sarimax = mean_absolute_error(test_endog, forecast_values)

#Calculate RMSE and MAE for Exponential Smoothing
rmse_exp = np.sqrt(mean_squared_error(test_endog_exp, forecast_values_exp))
mae_exp = mean_squared_error(test_endog_exp, forecast_values_exp)

# Find the maximum value across all test datasets
max_value = max(test_endog_ar.max(), test_endog.max(), test_endog_exp.max())

# Find the minimum value across all test datasets
min_value = min(test_endog_ar.min(), test_endog.min(), test_endog_exp.min())

# Calculate the range
value_range = max_value - min_value

# Calculate RMSE as a percentage of the value range
rmse_pct_arima = (rmse_arima / value_range) * 100
rmse_pct_sarimax = (rmse_sarimax / value_range) * 100
rmse_pct_exp = (rmse_exp / value_range) * 100

# Calculate MAE as a percentage of the value range
mae_pct_arima = (mae_arima / value_range) * 100
mae_pct_sarimax = (mae_sarimax / value_range) * 100
mae_pct_exp = (mae_exp / value_range) * 100

# Print the results
print(f"ARIMA RMSE (Property): {rmse_arima:.2f}, RMSE %:  {rmse_pct_arima:.2f}%,  MAE: {mae_arima:.2f}, MAE %: {mae_pct_arima:.2f}%")
print(f"SARIMAX RMSE (Property]): {rmse_sarimax:.2f},RMSE %: {rmse_pct_sarimax: .2f}%, MAE: {mae_sarimax:.2f}, MAE %: {mae_pct_sarimax: .2f}%")
print(f"Exponential Smoothing RMSE (Property): {rmse_exp:.2f}, RMSE %: {rmse_pct_exp:.2f}, % MAE: {mae_exp:.2f}, MAE %: {mae_pct_exp:.2f}%")

# Education Managers SARIMAX
# SRM22 and SRM23
srm22 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\SRM22.csv", 'SRM22', '15/06/2022')
srm23 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\SRM23.csv", 'SRM23', '08/06/2023')

# Combine datasets
combined_ed_managers = combine_datasets(srm22, srm23)

# Proceed with analysis and plotting
#plot_registrations_countdown(combined_ed_managers, 'Registrations Countdown to Education Managers Conference')
daily_registrations_ed = calculate_daily_registrations(combined_ed_managers)
complete_reg_ed = prepare_for_forecasting_with_t_minus(daily_registrations_ed)

# Split the data into training and testing sets
train_endog, train_exog, test_endog, test_exog = split_data_for_forecasting(complete_reg_ed)

# Use Auto ARIMA to find the optimal SARIMAX parameters
order, seasonal_order = auto_arima_forecasting(train_endog, train_exog)

# Fit the SARIMAX model using the optimal parameters
forecast_values, conf_int = sarimax_forecasting(train_endog, train_exog, test_exog, order, seasonal_order)

test_df = test_exog.copy()
test_df['Total Registrations'] = test_endog.values

# Plot the forecast against actual values
plot_forecast_vs_actuals(complete_reg_ed.iloc[:len(train_endog)], test_df, forecast_values, conf_int)

# Education Managers ARIMA
train, test = split_time_series_data(complete_reg_ed, split_ratio=0.8)

# For ARIMA, focusing on the endogenous variable only
train_endog_ar = train['Total Registrations']
test_endog_ar = test['Total Registrations']

# Finding the best ARIMA order
order = find_best_arima_order(train_endog_ar)

# Forecasting with ARIMA
forecast_values_arima, conf_int = arima_forecasting(train_endog_ar, order)

# Plotting the results
plt.figure(figsize=(30, 6))
plt.plot(train.index, train_endog_ar, label='Training Data')
plt.plot(test.index, test_endog_ar, label='Actual Test Data')
plt.plot(test.index, forecast_values_arima, label='ARIMA Forecast')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Registrations')
plt.title('ARIMA Forecast vs Actuals')
plt.show()

# Education Managers EXP
train, test = split_time_series_data(complete_reg_ed, split_ratio=0.8)

train_endog_exp= train['Total Registrations']
test_endog_exp = test['Total Registrations']

forecast_values_exp = exp_smoothing(train_endog_exp)

# Plotting the results
plt.figure(figsize=(30, 6))
plt.plot(train.index, train_endog_exp, label='Training Data')
plt.plot(test.index, test_endog_exp, label='Actual Test Data')
plt.plot(test.index, forecast_values_exp, label='Exponential Smoothing Forecast', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Registrations')
plt.title('Exponential Smoothing Forecast vs Actuals')
plt.show()

# Calculate RMSE and MAE for ARIMA
rmse_arima = np.sqrt(mean_squared_error(test_endog_ar, forecast_values_arima))
mae_arima = mean_absolute_error(test_endog_ar, forecast_values_arima)

# Calculate RMSE and MAE for SARIMAX
rmse_sarimax = np.sqrt(mean_squared_error(test_endog, forecast_values))
mae_sarimax = mean_absolute_error(test_endog, forecast_values)

#Calculate RMSE and MAE for Exponential Smoothing
rmse_exp = np.sqrt(mean_squared_error(test_endog_exp, forecast_values_exp))
mae_exp = mean_squared_error(test_endog_exp, forecast_values_exp)

# Find the maximum value across all test datasets
max_value = max(test_endog_ar.max(), test_endog.max(), test_endog_exp.max())

# Find the minimum value across all test datasets
min_value = min(test_endog_ar.min(), test_endog.min(), test_endog_exp.min())

# Calculate the range
value_range = max_value - min_value

# Calculate RMSE as a percentage of the value range
rmse_pct_arima = (rmse_arima / value_range) * 100
rmse_pct_sarimax = (rmse_sarimax / value_range) * 100
rmse_pct_exp = (rmse_exp / value_range) * 100

# Calculate MAE as a percentage of the value range
mae_pct_arima = (mae_arima / value_range) * 100
mae_pct_sarimax = (mae_sarimax / value_range) * 100
mae_pct_exp = (mae_exp / value_range) * 100

# Print the results
print(f"ARIMA RMSE (Education): {rmse_arima:.2f}, RMSE %:  {rmse_pct_arima:.2f}%,  MAE: {mae_arima:.2f}, MAE %: {mae_pct_arima:.2f}%")
print(f"SARIMAX RMSE (Education): {rmse_sarimax:.2f},RMSE %: {rmse_pct_sarimax: .2f}%, MAE: {mae_sarimax:.2f}, MAE %: {mae_pct_sarimax: .2f}%")
print(f"Exponential Smoothing RMSE (Education): {rmse_exp:.2f}, RMSE %: {rmse_pct_exp:.2f}, % MAE: {mae_exp:.2f}, MAE %: {mae_pct_exp:.2f}%")

# Education Property Managers SARIMAX
# MSE21
mse21 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\MSE21.csv", 'MSE21', '24/03/2021')

# Proceed with analysis and plotting
#plot_registrations_countdown(combined_ed_managers, 'Registrations Countdown to Education Managers Conference')
daily_registrations_edp = calculate_daily_registrations(mse21)
complete_reg_edp = prepare_for_forecasting_with_t_minus(daily_registrations_edp)

# Split the data into training and testing sets
train_endog, train_exog, test_endog, test_exog = split_data_for_forecasting(complete_reg_edp)

# Use Auto ARIMA to find the optimal SARIMAX parameters
order, seasonal_order = auto_arima_forecasting(train_endog, train_exog)

# Fit the SARIMAX model using the optimal parameters
forecast_values, conf_int = sarimax_forecasting(train_endog, train_exog, test_exog, order, seasonal_order)

test_df = test_exog.copy()
test_df['Total Registrations'] = test_endog.values

# Plot the forecast against actual values
plot_forecast_vs_actuals(complete_reg_edp.iloc[:len(train_endog)], test_df, forecast_values, conf_int)

# Education Property Managers ARIMA
train, test = split_time_series_data(complete_reg_edp, split_ratio=0.8)

# For ARIMA, focusing on the endogenous variable only
train_endog_ar = train['Total Registrations']
test_endog_ar = test['Total Registrations']

# Finding the best ARIMA order
order = find_best_arima_order(train_endog_ar)

# Forecasting with ARIMA
forecast_values_arima, conf_int = arima_forecasting(train_endog_ar, order)

# Plotting the results
plt.figure(figsize=(30, 6))
plt.plot(train.index, train_endog_ar, label='Training Data')
plt.plot(test.index, test_endog_ar, label='Actual Test Data')
plt.plot(test.index, forecast_values_arima, label='ARIMA Forecast')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Registrations')
plt.title('ARIMA Forecast vs Actuals')
plt.show()

# Education Property Managers EXP
train, test = split_time_series_data(complete_reg_edp, split_ratio=0.8)

train_endog_exp= train['Total Registrations']
test_endog_exp = test['Total Registrations']

forecast_values_exp = exp_smoothing(train_endog_exp)

# Plotting the results
plt.figure(figsize=(30, 6))
plt.plot(train.index, train_endog_exp, label='Training Data')
plt.plot(test.index, test_endog_exp, label='Actual Test Data')
plt.plot(test.index, forecast_values_exp, label='Exponential Smoothing Forecast', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Registrations')
plt.title('Exponential Smoothing Forecast vs Actuals')
plt.show()

# Calculate RMSE and MAE for ARIMA
rmse_arima = np.sqrt(mean_squared_error(test_endog_ar, forecast_values_arima))
mae_arima = mean_absolute_error(test_endog_ar, forecast_values_arima)

# Calculate RMSE and MAE for SARIMAX
rmse_sarimax = np.sqrt(mean_squared_error(test_endog, forecast_values))
mae_sarimax = mean_absolute_error(test_endog, forecast_values)

#Calculate RMSE and MAE for Exponential Smoothing
rmse_exp = np.sqrt(mean_squared_error(test_endog_exp, forecast_values_exp))
mae_exp = mean_squared_error(test_endog_exp, forecast_values_exp)

# Find the maximum value across all test datasets
max_value = max(test_endog_ar.max(), test_endog.max(), test_endog_exp.max())

# Find the minimum value across all test datasets
min_value = min(test_endog_ar.min(), test_endog.min(), test_endog_exp.min())

# Calculate the range
value_range = max_value - min_value

# Calculate RMSE as a percentage of the value range
rmse_pct_arima = (rmse_arima / value_range) * 100
rmse_pct_sarimax = (rmse_sarimax / value_range) * 100
rmse_pct_exp = (rmse_exp / value_range) * 100

# Calculate MAE as a percentage of the value range
mae_pct_arima = (mae_arima / value_range) * 100
mae_pct_sarimax = (mae_sarimax / value_range) * 100
mae_pct_exp = (mae_exp / value_range) * 100

# Print the results
print(f"ARIMA RMSE (Education Property): {rmse_arima:.2f}, RMSE %:  {rmse_pct_arima:.2f}%,  MAE: {mae_arima:.2f}, MAE %: {mae_pct_arima:.2f}%")
print(f"SARIMAX RMSE (Education Property): {rmse_sarimax:.2f},RMSE %: {rmse_pct_sarimax: .2f}%, MAE: {mae_sarimax:.2f}, MAE %: {mae_pct_sarimax: .2f}%")
print(f"Exponential Smoothing RMSE (Education Property): {rmse_exp:.2f}, RMSE %: {rmse_pct_exp:.2f}, % MAE: {mae_exp:.2f}, MAE %: {mae_pct_exp:.2f}%")

# All datasets
srm22 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\SRM22.csv", 'SRM22', '15/06/2022')
srm23 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\SRM23.csv", 'SRM23', '08/06/2023')
gp21 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\GP21.csv", 'GP21', '22/04/2021')
np21 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\NP21.csv", 'NP21', '09/11/2021')
d19 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\D19.csv", 'D19', '19/11/2019')
d21 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\D21.csv", 'D21', '09/12/2021')
mse21 = read_and_preprocess(r"C:\Users\Njula Chakaya\OneDrive\Documents\Masters_Practicals\Collaborative App Devt\Data Files\MSE21.csv", 'MSE21', '24/03/2021')

# Combine datasets
combined = combine_datasets(srm22, srm23,gp21,np21, d19,d21, mse21)

# Proceed with analysis and plotting
#plot_registrations_countdown(combined_ed_managers, 'Registrations Countdown to Education Managers Conference')
daily_registrations_all = calculate_daily_registrations(combined)
complete_reg_all = prepare_for_forecasting_with_t_minus(daily_registrations_all)

# Split the data into training and testing sets
train_endog, train_exog, test_endog, test_exog = split_data_for_forecasting(complete_reg_all)

# Use Auto ARIMA to find the optimal SARIMAX parameters
order, seasonal_order = auto_arima_forecasting(train_endog, train_exog)

# Fit the SARIMAX model using the optimal parameters
forecast_values, conf_int = sarimax_forecasting(train_endog, train_exog, test_exog, order, seasonal_order)

test_df = test_exog.copy()
test_df['Total Registrations'] = test_endog.values

# Plot the forecast against actual values
plot_forecast_vs_actuals(complete_reg_all.iloc[:len(train_endog)], test_df, forecast_values, conf_int)

# All ARIMA
train, test = split_time_series_data(complete_reg_all, split_ratio=0.8)

# For ARIMA, focusing on the endogenous variable only
train_endog_ar = train['Total Registrations']
test_endog_ar = test['Total Registrations']

# Finding the best ARIMA order
order = find_best_arima_order(train_endog_ar)

# Forecasting with ARIMA
forecast_values_arima, conf_int = arima_forecasting(train_endog_ar, order)

# Plotting the results
plt.figure(figsize=(30, 6))
plt.plot(train.index, train_endog_ar, label='Training Data')
plt.plot(test.index, test_endog_ar, label='Actual Test Data')
plt.plot(test.index, forecast_values_arima, label='ARIMA Forecast')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Registrations')
plt.title('ARIMA Forecast vs Actuals')
plt.show()

# All EXP
train, test = split_time_series_data(complete_reg_all, split_ratio=0.8)

train_endog_exp= train['Total Registrations']
test_endog_exp = test['Total Registrations']

forecast_values_exp = exp_smoothing(train_endog_exp)

# Plotting the results
plt.figure(figsize=(30, 6))
plt.plot(train.index, train_endog_exp, label='Training Data')
plt.plot(test.index, test_endog_exp, label='Actual Test Data')
plt.plot(test.index, forecast_values_exp, label='Exponential Smoothing Forecast', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Registrations')
plt.title('Exponential Smoothing Forecast vs Actuals')
plt.show()

# Calculate RMSE and MAE for ARIMA
rmse_arima = np.sqrt(mean_squared_error(test_endog_ar, forecast_values_arima))
mae_arima = mean_absolute_error(test_endog_ar, forecast_values_arima)

# Calculate RMSE and MAE for SARIMAX
rmse_sarimax = np.sqrt(mean_squared_error(test_endog, forecast_values))
mae_sarimax = mean_absolute_error(test_endog, forecast_values)

#Calculate RMSE and MAE for Exponential Smoothing
rmse_exp = np.sqrt(mean_squared_error(test_endog_exp, forecast_values_exp))
mae_exp = mean_squared_error(test_endog_exp, forecast_values_exp)

# Find the maximum value across all test datasets
max_value = max(test_endog_ar.max(), test_endog.max(), test_endog_exp.max())

# Find the minimum value across all test datasets
min_value = min(test_endog_ar.min(), test_endog.min(), test_endog_exp.min())

# Calculate the range
value_range = max_value - min_value

# Calculate RMSE as a percentage of the value range
rmse_pct_arima = (rmse_arima / value_range) * 100
rmse_pct_sarimax = (rmse_sarimax / value_range) * 100
rmse_pct_exp = (rmse_exp / value_range) * 100

# Calculate MAE as a percentage of the value range
mae_pct_arima = (mae_arima / value_range) * 100
mae_pct_sarimax = (mae_sarimax / value_range) * 100
mae_pct_exp = (mae_exp / value_range) * 100

# Print the results
print(f"ARIMA RMSE (All): {rmse_arima:.2f}, RMSE %:  {rmse_pct_arima:.2f}%,  MAE: {mae_arima:.2f}, MAE %: {mae_pct_arima:.2f}%")
print(f"SARIMAX RMSE (All): {rmse_sarimax:.2f},RMSE %: {rmse_pct_sarimax: .2f}%, MAE: {mae_sarimax:.2f}, MAE %: {mae_pct_sarimax: .2f}%")
print(f"Exponential Smoothing RMSE (All): {rmse_exp:.2f}, RMSE %: {rmse_pct_exp:.2f}, % MAE: {mae_exp:.2f}, MAE %: {mae_pct_exp:.2f}%")
