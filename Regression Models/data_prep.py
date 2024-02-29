"""This code reads data from all csv files in the code directory, processes the required data from the csv files into pandas dataframes and combines the dataframes
based on the target audience. These dataframes are then used by the regression models to predict the number of bookings for each target audience. These dataframes
are also used by the finalized prediction model to predict the number of bookings for new data input by the users.

The libraries used in this code and their versions are:
glob2                        0.7
pandas                       2.2.1
os                           


This python file needs to be imported into other code files to use the dataframes created in this file."""


import pandas as pd
import os
import glob

df_dict = {} # Create a dictionary to store the dataframes for each file of a target audience

"""Function to extract data from a csv file and store it in a dataframe dictionary"""
def csv_extracter (filename):
    raw_data = pd.read_csv(filename, skiprows = 1) # Read the data from the csv file into a pandas dataframe
    booking_ref = raw_data.columns[raw_data.columns.str.contains("book", case = False) & raw_data.columns.str.contains("ref", case = False)].tolist()[0] # Find the column with the booking reference
    date = raw_data.columns[raw_data.columns.str.contains("date", case = False)].tolist()[0] # Find the column with the date

    df_raw = raw_data[[booking_ref, date]].copy() # Create a new dataframe with only the booking reference and date columns
    df_raw.rename(columns = {booking_ref: "Booking Reference", date: "Date"}, inplace = True) # Rename the columns
    df_raw.loc[:, "Date"] = pd.to_datetime(df_raw["Date"], dayfirst= True, errors = "coerce") # Convert the date column to datetime format 
    df_raw.sort_values(by = "Date", inplace = True) # Sort the dataframe by date. Not using mergesort because the order of entries with same date is not important.
    df_raw.reset_index(drop = True, inplace = True) # Reset the index of the dataframe
    
    df_dates = pd.DataFrame(index = pd.date_range(start = df_raw["Date"].min(), end = df_raw["Date"].max())) # Create a new DataFrame with a continuous date range
    df_dates.index = df_dates.index.astype(df_raw["Date"].dtype) # Change the index type to match the date column in the original DataFrame
    df_raw = df_dates.merge(df_raw, left_index = True, right_on = "Date", how = "left") # Merge the new DataFrame with the original DataFrame on the date column to include rows with missing dates
    df_raw = df_raw.reset_index(drop = True) # Reset the index of the merged DataFrame
    df_raw['Booking Reference'] = df_raw['Booking Reference'].fillna("none") # Fill the booking references corresponding to the newly added dates with "none". This will ensure booking counts are not affected by these records

    df_dict["df_" + filename] = df_raw # Add the dataframe to the dictionary with the filename as the key
    return df_dict


"""Extract data from csv files from code directory and store in a dataframe dictionary"""
path = os.getcwd() 
for name in glob.glob(path + "\*.csv"):
    filename = os.path.basename(name)
    csv_extracter(filename)


"""Add Event Dates to a List"""
event_dates = ["19/11/2019", "09/12/2021", "22/04/2021", "24/03/2021", "09/11/2021", "15/06/2022", "08/06/2023"]
event_dates = pd.to_datetime(event_dates, dayfirst = True) # Convert the event dates to datetime format


"""Remove bookings made after the event date from corresponding dataframe in the dataframe dictionary"""
for key, event_date in zip(df_dict.keys(), event_dates):
    df_dict[key] = df_dict[key].loc[df_dict[key]["Date"] < event_date] # Remove bookings made after the corresponding event date


"""Make a dictionary of new dataframes with weeks left to event date, booking count and z scores for each week. 
Make another dictionary in which the weeks with z scores greater than 2 are removed from each df (for week-based training of prediction model).
Do the same for days as the timeframe, but z score values are still considered on a per week basis for consistency with initial data analysis"""  
df_weeks_dict = {} # contains dataframes with weeks left to event date, booking count and z scores for each week
df_bookingdates = {} # contains dataframes copied from parent dataframes but with records with no booking reference removed. These records are basically non-existent in csv files and were only added to the parent dataframes to fill in missing dates
df_weeks_noads_dict = {} # contains dataframes from df_weeks_dict with weeks with z scores greater than 2 removed (these weeks are identified as advertisement weeks)
df_days_dict = {} # contains dataframes with days left to event date and booking count for each day
df_days_noads_dict = {} # contains dataframes in which days corresponding to weeks with z scores >= 2 are removed from the daywise dataframes
# need to cleanup this code
for key, event_date in zip(df_dict.keys(), event_dates):
    start_date = df_dict[key]["Date"].min() # Find the start date
    week_start_date = start_date - pd.to_timedelta(start_date.weekday(), unit = "d") # Find the start date of the week
    df_dict[key]["Week No."] = df_dict[key]["Date"].apply(lambda x: (x - week_start_date).days // 7 + 1).astype(int) # Add a new column to the dataframe with the week number starting from 1 for the start date
    
    event_week_no = (event_date - week_start_date).days // 7 + 1 # Find the week number of the event date
    df_dict[key]["Weeks Left"] = event_week_no - df_dict[key]["Week No."] # Add a new column to the dataframe with the weeks left to the event date
    df_dict[key]["Days Left"] = df_dict[key]["Date"].apply(lambda x: (event_date - x).days).astype(int) # Add a new column to the dataframe with the days left to the event date

    df_bookingdates[key] = df_dict[key].copy()
    df_bookingdates[key] = df_bookingdates[key].drop(df_bookingdates[key][df_bookingdates[key]["Booking Reference"] == "none"].index) # Drop records with no booking reference
    df_bookingdates[key].reset_index(drop = True, inplace = True) # Reset the index of the dataframe

    week_range = range(df_dict[key]["Weeks Left"].min(), df_dict[key]["Weeks Left"].max() + 1) # Create a range of week numbers
    df_weeks_dict[key] = pd.DataFrame(index = week_range) # Create a new dataframe
    df_weeks_dict[key]["Weeks to Event"] = week_range # Count down the weeks to the event date
    booking_counts = df_bookingdates[key].groupby(["Weeks Left"]).size().reindex(week_range).fillna(0).astype(int) # Group by weeks left and reindex to match week numbers
    df_weeks_dict[key]["Booking Count"] = booking_counts # Add a new column to the dataframe with the booking count for each week
    df_weeks_dict[key] = df_weeks_dict[key].iloc[::-1].reset_index(drop = True) # Reverse the order of the dataframe and reset the index

    mean_booking_count = df_weeks_dict[key]["Booking Count"].mean()
    std_booking_count = df_weeks_dict[key]["Booking Count"].std(ddof=0)
    df_weeks_dict[key]["Z Score"] = df_weeks_dict[key]["Booking Count"].apply(lambda x: (x - mean_booking_count) / std_booking_count) # Add a column for z scores in the weeks dataframe

    df_weeks_noads_dict[key] = df_weeks_dict[key].copy() # Create a new copy of each weekwise dataframe
    df_weeks_noads_dict[key] = df_weeks_noads_dict[key].drop(df_weeks_noads_dict[key][df_weeks_noads_dict[key]["Z Score"] >= 1.00].index) # Drop records with z scores greater than or equal to 1
    df_weeks_noads_dict[key].reset_index(drop = True, inplace = True) # Reset the index of the dataframe

    day_range = range(df_dict[key]["Days Left"].min(), df_dict[key]["Days Left"].max() + 1) # Create a range of day numbers
    df_days_dict[key] = pd.DataFrame(index = day_range) # Create a new dataframe
    df_days_dict[key]["Days to Event"] = day_range # Count down the days to the event date
    booking_counts_perday = df_bookingdates[key].groupby(["Days Left"]).size().reindex(day_range).fillna(0).astype(int) # Group by days left and reindex to match day numbers
    df_days_dict[key]["Booking Count"] = booking_counts_perday # Add a new column to the dataframe with the booking count for each day
    df_days_dict[key] = df_days_dict[key].iloc[::-1].reset_index(drop = True) # Reverse the order of the dataframe and reset the index

    weeks_ads = list(df_weeks_dict[key].loc[df_weeks_dict[key]["Z Score"] >= 2.00, "Weeks to Event"]) # Create a list of weeks with z scores greater than or equal to 2
    days_ads = list(df_dict[key].loc[df_dict[key]["Weeks Left"].isin(weeks_ads), "Days Left"]) # Create a list of days with z scores greater than or equal to 2
    df_days_dict[key]["Advertisement Status"] = df_days_dict[key]["Days to Event"].apply(lambda x: 1 if x in days_ads else 0) # Add a column to the dataframe to indicate advertisement status


"""Combine dataframes based on target audience"""
"""For IT Managers"""
df_ITM_days = pd.concat([df_days_dict["df_D19.csv"], df_days_dict["df_D21.csv"]], ignore_index = True) # Combine the daywise dataframes for ITM
df_ITM_days.sort_values(by = "Days to Event", inplace = True, ascending = False) # Sort the dataframe by days to event in descending order
df_ITM_days.reset_index(drop = True, inplace = True) # Reset the index of the dataframe


"""For Property Managers"""
df_PM_days = pd.concat([df_days_dict["df_GP21.csv"], df_days_dict["df_NP21.csv"]], ignore_index = True) # Combine the daywise dataframes for PM
df_PM_days.sort_values(by = "Days to Event", inplace = True, ascending = False) # Sort the dataframe by days to event in descending order
df_PM_days.reset_index(drop = True, inplace = True) # Reset the index of the dataframe


"""For Education Property Managers"""
df_EPM_days = df_days_dict["df_MSE21.csv"]


"""For Education Managers"""
df_EM_days = pd.concat([df_days_dict["df_SRM22.csv"], df_days_dict["df_SRM23.csv"]], ignore_index = True) # Combine the daywise dataframes for EM
df_EM_days.sort_values(by = "Days to Event", inplace = True, ascending = False) # Sort the dataframe by days to event in descending order
df_EM_days.reset_index(drop = True, inplace = True) # Reset the index of the dataframe


"""For Other Target Audiences"""
df_other_days = pd.concat([df_days_dict["df_D19.csv"], df_days_dict["df_D21.csv"], df_days_dict["df_GP21.csv"], df_days_dict["df_NP21.csv"], df_days_dict["df_MSE21.csv"],
                            df_days_dict["df_SRM22.csv"], df_days_dict["df_SRM23.csv"]], ignore_index = True) # Combine the daywise dataframes from all files for other target audiences
df_other_days.sort_values(by = "Days to Event", inplace = True, ascending = False) # Sort the dataframe by days to event in descending order
df_other_days.reset_index(drop = True, inplace = True) # Reset the index of the dataframe
# Note that df_other_days contains 1686 records (individual data points) which is the largest dataset we can achieve for training the model