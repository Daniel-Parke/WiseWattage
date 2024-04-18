import logging
import calendar

import pandas as pd
import numpy as np

def calc_load_profile(daily_demand: float = 9.91, daily_std: float = 0.2, hourly_std: float = 0.1,
                      profile: str = "Domestic", country: str = "UK") -> pd.DataFrame:
    """
    Function to calculate an annual load profile with hourly variability.

    The function takes in daily demand, daily standard deviation and hourly standard deviation
    as inputs to generate a load profile. The function also takes in a profile name and country
    to specify the appropriate load profile to use.

    The function first tries to read in a pre-calculated load profile from a CSV file based on the
    provided inputs. If the file cannot be found the function generates a new load profile based
    on the provided inputs.

    The function then scales the energy use values based on the daily demand provided, and applies
    daily and hourly perturbation to the energy use values.

    Arguments:
        daily_demand {float} -- The daily energy demand for the load profile (default: {9.91})
        daily_std {float} -- The standard deviation of the daily perturbation (default: {0.2})
        hourly_std {float} -- The standard deviation of the hourly perturbation (default: {0.1})
        profile {str} -- The type of profile to use (default: {"Domestic"})
        country {str} -- The country to use (default: {"UK"})

    Returns:
        pd.DataFrame -- The adjusted load profile
    """

    # Try to read in an existing load profile based on the inputs
    try:
        load_data = pd.read_csv(f"demand/load_profiles/{country}_{profile}_load_profile_hourly.csv")

    # If the file cannot be found generate a new load profile
    except:
        logging.info(f"Unable to find load profile with chosen input parameters, generating new profile")
        # Create a new date range for the load profile
        load_data = pd.DataFrame(index=pd.date_range(start='1/1/2017', end='12/31/2017', freq='H'))
        # Generate a primary load profile with constant energy use per hour
        load_data['Energy_Use_kWh_Base'] = 9.91 / 365

    # Create two extra columns to store the daily and hourly perturbation
    load_data['Energy_Use_kWh'] = 0
    load_data["Variability_Factor"] = 0

    # Scale energy use relative to base profile, original profile represents 9.91kWh daily use
    scale_factor = daily_demand / (load_data['Energy_Use_kWh_Base'].sum()/365)
    load_data['Energy_Use_kWh_Base'] *= scale_factor

    # Set index to DateTime format instead of range
    load_data['DateTime'] = pd.to_datetime(load_data['DateTime'])
    load_data.set_index('DateTime', inplace=True)

    # Generate daily perturbation values for each day
    daily_deviation = np.random.normal(0, daily_std, size=load_data.resample('D').count().size)

    # Map daily perturbation values to each hour of the corresponding day
    load_data['Daily_Deviation'] = load_data.index.map(lambda d: daily_deviation[d.dayofyear - 1])

    # Generate hourly perturbation values
    load_data['Hourly_Deviation'] = np.random.normal(0, hourly_std, size=len(load_data))

    load_data["Variability_Factor"] = (1 + load_data['Daily_Deviation'] + load_data['Hourly_Deviation'])

    # Apply the combined perturbation to the primary load
    load_data['Energy_Use_kWh'] = load_data['Energy_Use_kWh_Base'] * load_data["Variability_Factor"]

    # Drop the extra columns to leave only the adjusted load
    load_data = load_data.drop(columns=['Daily_Deviation', 'Hourly_Deviation'])
    load_data.reset_index(drop=True, inplace=True)

    return load_data



# Model annual load profile incorporating variablity, save these values to daily and annual demand values.
def initialise_load(self):
    """
    Function to initialise the load profile and set the daily and annual electricity values.

    The function checks if the load profile path is set, if it is it reads the csv data into a pandas
    dataframe and uses the convert_to_hourly function to convert it to hourly data. If the load profile
    path is not set it uses the calc_load_profile function to generate a new profile with the provided
    values for daily demand, daily variability and hourly variability.

    After generating or loading the load profile it sets the new daily and annual electricity values to
    reflect the variability calculations.
    """
    # Calculate Load Profile and return dataframe
    if self.load_profile_path is not None:
        lp_data = pd.read_csv(self.load_profile_path)
        self.load_profile = convert_to_hourly(lp_data)

    if self.load_profile is None:
        self.load_profile = calc_load_profile(daily_demand = self.daily_electric,
                                                daily_std = self.daily_variablity,
                                                hourly_std = self.timestep_variability,
                                                profile = self.profile,
                                                country = self.country)
        
    # Set new daily and annual values to reflect variability calculations updating values
    self.daily_electric = round(self.load_profile.Energy_Use_kWh.sum()/365, 3)
    self.annual_electric = self.daily_electric * 365
    
    logging.info(
        f"Load Profile Generated: Daily Electricity Use: {self.daily_electric}kWh"
        f"Annual Electricity Use: {self.annual_electric}kWh, Daily Variability: {self.daily_variablity*100}%"
        f"Hourly Variability: {self.timestep_variability*100}%"
    )
    logging.info("*******************")



def drop_leap_days(data):
    """
    Function to drop leap days from a pandas TimeSeries or DataFrame

    Arguments:
        data {pd.TimeSeries or pd.DataFrame} -- The data to be processed

    Returns:
        pd.TimeSeries or pd.DataFrame -- The data with leap days removed
    """
    leap_days = []
    years = data.index.year.unique()
    for year in years:
        # Check if a given year is a leap year
        if calendar.isleap(year):
            # Create a timestamp for the potential leap day
            leap_day = pd.Timestamp(year=year, month=2, day=29)
            # Check if the leap day is in the data index
            if leap_day in data.index:
                # Add the leap day to the list of leap days to be removed
                leap_days.append(leap_day)

    # If any leap days were found, drop them from the data
    if leap_days:
        data = data.drop(leap_days)
    return data


# Converts 30 min smart meter data into hourly series
def convert_to_hourly(data):
    """
    Converts a dataframe with 30-minute smart meter data into hourly values.

    The input data is expected to have a DateTime index and columns with even numbered
    indices representing the start of a 30-minute period and odd numbered indices
    representing the end of a 30-minute period. The function will average the values
    from both 30-minute intervals to create hourly values.

    The output will be a DataFrame with a DateTime index and a single column
    named 'Energy_Use_kWh' containing the average hourly energy use.

    Arguments:
        data {pd.DataFrame} -- The input data to be converted to hourly values

    Returns:
        pd.DataFrame -- The converted data
    """
    data_hourly = pd.DataFrame()  # Create empty dataframe to store hourly values
    data_hourly["Date"] = data["Date"]  # Copy date column from input data

    # Loop through every 2 columns in the input data and average the values
    # to create hourly values
    for i in range(1, len(data.columns), 2):
        data_hourly[f'Hour {i//2}'] = data.iloc[:, i] + data.iloc[:, i+1]
        data_hourly[f'Hour {i//2}'] /= 2

    data_hourly['Date'] = pd.to_datetime(data_hourly['Date'], format='mixed')
    data_hourly.set_index('Date', inplace=True)  # Set index to DateTime format

    data_hourly = drop_leap_days(data_hourly)  # Drop any leap days in the data
    data_hourly.reset_index(inplace=True)  # Reset index to default range

    # Melt the DataFrame to go from wide to long format
    melted_data = data_hourly.melt(id_vars=['Date'], var_name='Hour', value_name='Energy_Use_kWh')

    # Convert the 'Hour' column to be just the hour number and convert to Timedelta
    melted_data['Hour'] = pd.to_timedelta(melted_data['Hour'].str.extract(r'Hour (\d+)')[0].astype(int), unit='h')

    melted_data['DateTime'] = melted_data['Date'] + melted_data['Hour']
    melted_data.set_index('DateTime', inplace=True)  # Set index to DateTime format

    # Arrange dataframe into annual values instead of chronological
    melted_data['Month_Day'] = melted_data.index.strftime('%m-%d')
    melted_data.sort_values('Month_Day', inplace=True)
    melted_data.drop(columns=['Month_Day', 'Date', 'Hour'], inplace=True)
    melted_data.reset_index(inplace=True)

    return melted_data
