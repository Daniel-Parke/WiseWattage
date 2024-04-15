import logging
import calendar

import pandas as pd
import numpy as np

def calc_load_profile(daily_demand: float = 9.91, daily_std: float = 0.2, hourly_std: float = 0.1,
                      profile: str = "Domestic", country: str = "UK"):
    
    try:
        load_data = pd.read_csv(f"demand/load_profiles/{country}_{profile}_load_profile_hourly.csv")

    except:
        return print("Unable to find load profile with chosen input parameters")
    
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
        f"Load Profile Generated: Daily Electricity Use: {self.daily_electric}kWh, "
        f"Annual Electricity Use: {self.annual_electric}kWh, Daily Variability: {self.daily_variablity*100}%, "
        f"Hourly Variability: {self.timestep_variability*100}%"
    )
    logging.info("*******************")



def drop_leap_days(data):
    leap_days = []
    years = data.index.year.unique()
    for year in years:
        if calendar.isleap(year):
            leap_day = pd.Timestamp(year=year, month=2, day=29)
            if leap_day in data.index:
                leap_days.append(leap_day)

    if leap_days:
        data = data.drop(leap_days)
    return data


# Converts 30 min smart meter data into hourly series
def convert_to_hourly(data):
    data_hourly = pd.DataFrame()
    data_hourly["Date"] = data["Date"]
    for i in range(1, len(data.columns), 2):
        data_hourly[f'Hour {i//2}'] = data.iloc[:, i] + data.iloc[:, i+1]

    data_hourly['Date'] = pd.to_datetime(data_hourly['Date'], format='mixed')
    data_hourly.set_index('Date', inplace=True)

    data_hourly = drop_leap_days(data_hourly)
    data_hourly.reset_index(inplace=True)

    # Melt the DataFrame to go from wide to long format
    melted_data = data_hourly.melt(id_vars=['Date'], var_name='Hour', value_name='Energy_Use_kWh')

    # Convert the 'Hour' column to be just the hour number and convert to Timedelta
    melted_data['Hour'] = pd.to_timedelta(melted_data['Hour'].str.extract(r'Hour (\d+)')[0].astype(int), unit='h')

    melted_data['DateTime'] = melted_data['Date'] + melted_data['Hour']
    melted_data.set_index('DateTime', inplace=True)

    # Arrange dataframe into annual values instead of chronological
    melted_data['Month_Day'] = melted_data.index.strftime('%m-%d')
    melted_data.sort_values('Month_Day', inplace=True)
    melted_data.drop(columns=['Month_Day', 'Date', 'Hour'], inplace=True)
    melted_data.reset_index(inplace=True)

    return melted_data
