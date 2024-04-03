import logging

import pandas as pd
import numpy as np

def calc_load_profile(daily_demand: float = 9.91, daily_std: float = 0.2, hourly_std: float = 0.1,
                      profile: str = "Domestic", country: str = "UK"):
    
    try:
        load_data = pd.read_csv(f"demand/load_profiles/{country}_{profile}_load_profile_hourly.csv")

    except:
        return print("Unable to find load profile with chosen input parameters")
    
    load_data['Adjusted_Energy_Use_kWh'] = 0
    load_data["Variability_Factor"] = 0

    # Scale energy use relative to base profile, original profile represents 9.91kWh daily use
    scale_factor = daily_demand / (load_data['Energy_Use_kWh'].sum()/365)
    load_data['Energy_Use_kWh'] *= scale_factor

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
    load_data['Adjusted_Energy_Use_kWh'] = load_data['Energy_Use_kWh'] * load_data["Variability_Factor"]

    # Drop the extra columns to leave only the adjusted load
    load_data = load_data.drop(columns=['Daily_Deviation', 'Hourly_Deviation'])

    return load_data


# Model annual load profile incorporating variablity, save these values to daily and annual demand values.
def initialise_load(self):
    # Calculate Load Profile and return dataframe
    self.load_profile = calc_load_profile(daily_demand = self.daily_electric,
                                              daily_std = self.daily_variablity,
                                              hourly_std = self.timestep_variability,
                                              profile = self.profile,
                                              country = self.country)
        
    # Set new daily and annual values to reflect variability calculations updating values
    self.daily_electric = round(self.load_profile.Adjusted_Energy_Use_kWh.sum()/365, 3)
    self.annual_electric = self.daily_electric * 365
    
    logging.info(
        f"Load Profile Generated: Daily Electricity Use: {self.daily_electric}kWh, "
        f"Annual Electricity Use: {self.annual_electric}kWh, Daily Variability: {self.daily_variablity*100}%, "
        f"Hourly Variability: {self.timestep_variability*100}%"
    )
