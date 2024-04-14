import pandas as pd
import numpy as np

from solar.SolarPVModel import SolarPVModel
from demand.Load import Load
from utility.Grid import Grid
from utility.Inverter import Inverter

# Initialise list of variables to be used later
pv_model_variables = ["PV_Gen_kWh_Total", "Combined_PV_Losses_kWh_Total"]

columns_to_drop = ["Energy_Use_kWh_Base", "Variability_Factor",
                   "Net_Energy_kWh", "Net_Energy_Demand_kWh"]

columns_to_keep = ['Energy_Use_kWh', "Renewable_Energy_Use_kWh", 'Grid_Imports_kWh', 'Grid_Exports_kWh', 
                   'PV_Gen_kWh_Total', 'PV_AC_Output_kWh', 'Consumed_Solar_kWh', 'Excess_Solar_kWh',
                   "Battery_SoC_kWh", "Battery_Charge_kWh", "Battery_Discharge_kWh",
                   'Unused_Energy_kWh', 'Combined_PV_Losses_kWh_Total', 'Inverter_Losses_kWh',
                   'Battery_Losses_kWh', "Inverter_Limited_kWh", "Import_Limited", "Export_Limited"]


def initialise_model(self):
# Initialise Grid and Inverter Variables if not provided
    if self.grid is None:
        self.grid = Grid()

    if self.inverter is None:
        self.inverter = Inverter()

    # Initialise Energy Demand if not provided and update Energy Flows
    if self.load is None:
        self.load = Load()

    # Run SolarPVModel, save entire results and add required values to model.model datafarme
    if self.arrays is not None:
        self.pv_model = SolarPVModel(self.site, self.arrays)

    self.model = self.load.load_profile
    self.model["Net_Energy_kWh"] = -self.model["Energy_Use_kWh"]
    self.model["Net_Energy_Demand_kWh"] = self.model["Energy_Use_kWh"]
    self.model["Renewable_Energy_Use_kWh"] = 0

    self.model["Inverter_Losses_kWh"] = 0
    self.model["Inverter_Limited_kWh"] = 0

    self.model["Grid_Imports_kWh"] = 0
    self.model["Grid_Exports_kWh"] = 0
    self.model["Import_Limited"] = 0
    self.model["Export_Limited"] = 0
    self.model["Unused_Energy_kWh"] = 0
    

        
def calc_solar_energy_flow(self, pv_model_variables=pv_model_variables, mo=99999999):
        if self.pv_model is not None:
            pv_data = self.pv_model.combined_model[pv_model_variables]
            self.model = self.model.join(pv_data, how='left')

            if self.inverter.max_output is not None:
                 mo = self.inverter.max_output

            self.model["PV_AC_Output_kWh"] = np.minimum(mo, (self.model["PV_Gen_kWh_Total"]
                                            * self.inverter.inverter_eff))
            self.model["Inverter_Losses_kWh"] += (self.model["PV_AC_Output_kWh"]
                                            * (1-(self.inverter.inverter_eff)))
            self.model["Inverter_Limited_kWh"] = (self.model["PV_Gen_kWh_Total"]
                                            * self.inverter.inverter_eff) - self.model["PV_AC_Output_kWh"]

            # Update Net Energy Flows
            self.model["Net_Energy_kWh"] += self.model["PV_AC_Output_kWh"]
            self.model["Net_Energy_Demand_kWh"] = np.maximum(0, self.model["Net_Energy_Demand_kWh"]
                                                            - self.model["PV_AC_Output_kWh"])
            
            self.model["Excess_Solar_kWh"] = np.maximum(0, self.model["PV_AC_Output_kWh"] - 
                                                        self.model["Energy_Use_kWh"])
            self.model["Unused_Energy_kWh"] += self.model["Excess_Solar_kWh"]
            
            self.model["Consumed_Solar_kWh"] = np.maximum(0, self.model["PV_AC_Output_kWh"]
                                                            - self.model["Excess_Solar_kWh"])
            
            self.model["Renewable_Energy_Use_kWh"] += self.model["Consumed_Solar_kWh"]
        
            

def calc_battery_energy_flow_old(self):
    if self.storage is not None:
        # Initialize the state of charge of the battery
        current_soc = self.storage.useable_capacity * self.storage.initial_charge
        self.model["Battery_SoC_kWh"] = 0.0
        self.model["Battery_Charge_kWh"] = 0.0
        self.model["Battery_Discharge_kWh"] = 0.0
        self.model["Battery_Losses_kWh"] = 0.0

        max_discharge = self.storage.max_discharge_kW
        max_charge = self.storage.max_charge_kW
        eff = self.inverter.inverter_eff
        max_cap = self.storage.useable_capacity
        
        for index, row in self.model.iterrows():
            # If there's demand, we might need to discharge the battery
            if row["Net_Energy_Demand_kWh"] > 0:
                # The potential discharge is the lesser of the demand or the battery's available energy
                potential_discharge = min(row["Net_Energy_Demand_kWh"]/eff, 
                                          current_soc, 
                                          max_discharge)
                
                # Update the battery state of charge
                current_soc -= potential_discharge
                actual_energy_to_use = potential_discharge * eff
                
                # Update the dataframe for the current timestep
                self.model.at[index, "Battery_Discharge_kWh"] = actual_energy_to_use
                self.model.at[index, "Battery_Losses_kWh"] = potential_discharge - actual_energy_to_use
                
                # Update net energy and demand after accounting for inverter efficiency
                self.model.at[index, "Net_Energy_kWh"] += actual_energy_to_use
                self.model.at[index, "Net_Energy_Demand_kWh"] -= actual_energy_to_use
                self.model.at[index, "Renewable_Energy_Use_kWh"] += actual_energy_to_use

            
            # If there's excess solar energy, we might charge the battery
            if row["Net_Energy_kWh"] > 0 and current_soc < max_cap:
                # The potential charge is the lesser of the excess energy, the battery's remaining capacity, or the max charge rate
                potential_charge = min(row["Net_Energy_kWh"], 
                                       max_charge,
                                       max_cap - current_soc)
                
                # Account for inverter efficiency during charging & Update the battery state of charge
                actual_charge = potential_charge * eff
                current_soc += actual_charge
                
                # Update the dataframe for the current timestep
                self.model.at[index, "Battery_Charge_kWh"] = potential_charge
                self.model.at[index, "Net_Energy_kWh"] -= potential_charge
                self.model.at[index, "Unused_Energy_kWh"] -= potential_charge
                self.model.at[index, "Consumed_Solar_kWh"] += potential_charge
                self.model.at[index, "Excess_Solar_kWh"] -= potential_charge
            
            # Assign the updated state of charge to the dataframe
            self.model.at[index, "Battery_SoC_kWh"] = current_soc
            
            # Ensure the Battery SoC does not exceed the usable capacity or drop below the minimum SoC
            current_soc = min(current_soc, max_cap)


from numba import jit

@jit(nopython=True)
def calc_battery_state(n, net_demand, net_energy, initial_soc, max_discharge, max_charge, eff, max_cap,
                       renewable_cons):
    soc_series = np.zeros(n)
    charge_amount = np.zeros(n)
    discharge_amount = np.zeros(n)
    losses = np.zeros(n)
    current_soc = initial_soc
    
    for i in range(n):
        # Discharging
        if net_demand[i] > 0 and current_soc > 0:
            discharge_power = min(net_demand[i] / eff, current_soc, max_discharge)
            actual_discharge = discharge_power * eff

            discharge_amount[i] = actual_discharge
            losses[i] = discharge_power - actual_discharge
            current_soc -= discharge_power
            net_demand[i] -= actual_discharge
            net_energy[i] += actual_discharge
            renewable_cons[i] += actual_discharge

        # Charging
        if net_energy[i] > 0 and current_soc < max_cap:
            charge_power = min(net_energy[i], max_cap - current_soc, max_charge)
            actual_charge = charge_power * eff
            charge_amount[i] = charge_power
            current_soc += actual_charge
            net_energy[i] -= charge_power

        # Ensure SoC does not fall below zero or exceed maximum capacity after each operation
        current_soc = max(min(current_soc, max_cap), 0)
        soc_series[i] = current_soc

    return soc_series, charge_amount, discharge_amount, losses, net_demand, net_energy, renewable_cons

def calc_battery_energy_flow(self):
    if self.storage is not None:
        # Extract the values to numpy arrays for numba acceleration
        n = len(self.model)
        net_demand = self.model["Net_Energy_Demand_kWh"].values.copy()
        net_energy = self.model["Net_Energy_kWh"].values.copy()
        renewable_cons = self.model["Renewable_Energy_Use_kWh"].values.copy()

        # Call the numba-accelerated function
        results = calc_battery_state(
            n, 
            net_demand, 
            net_energy, 
            self.storage.useable_capacity * self.storage.initial_charge,
            self.storage.max_discharge_kW,
            self.storage.max_charge_kW,
            self.inverter.inverter_eff,
            self.storage.useable_capacity,
            renewable_cons
        )

        # Update DataFrame columns with the results
        self.model["Battery_SoC_kWh"] = results[0]
        self.model["Battery_Charge_kWh"] = results[1]
        self.model["Battery_Discharge_kWh"] = results[2]
        self.model["Battery_Losses_kWh"] = results[3]
        self.model["Net_Energy_Demand_kWh"] = results[4]
        self.model["Net_Energy_kWh"] = results[5]
        self.model["Renewable_Energy_Use_kWh"] = results[6]


def calc_grid_energy_flow(self):
    if self.grid.import_allow == True:
        # Calculate grid imports where net energy demand is positive
        self.model["Grid_Imports_kWh"] = np.where(self.model["Net_Energy_Demand_kWh"] > 0,
                                                np.minimum(self.grid.import_limit, 
                                                           self.model["Net_Energy_Demand_kWh"]), 
                                                0)
        self.model["Import_Limited"] = self.model["Net_Energy_Demand_kWh"] - self.model["Grid_Imports_kWh"]
        self.model["Net_Energy_Demand_kWh"] -= self.model["Grid_Imports_kWh"]
        self.model["Net_Energy_kWh"] += self.model["Grid_Imports_kWh"]

    # Calculate grid exports where net energy demand is negative
    if self.grid.export_allow == True:
        self.model["Grid_Exports_kWh"] = np.where(
             self.model["Net_Energy_kWh"] > 0, np.minimum(
                  self.grid.export_limit, self.model["Net_Energy_kWh"]), 
                  0)
        
        self.model["Export_Limited"] = self.model["Net_Energy_kWh"] - self.model["Grid_Exports_kWh"]
        self.model["Unused_Energy_kWh"] -= self.model["Grid_Exports_kWh"]
        self.model["Net_Energy_kWh"] -= self.model["Grid_Exports_kWh"]


def sort_columns(self, columns_to_keep=columns_to_keep, columns_to_drop=columns_to_drop):
        # First, drop specified columns if any are given and they exist in the DataFrame
        self.model.drop(columns=[col for col in columns_to_drop if col in self.model.columns],
                            axis=1, inplace=True)

        # Reorder columns by filtering the list to include only those that are present in the DataFrame
        filtered_columns = [col for col in columns_to_keep if col in self.model.columns]
        self.model = round(self.model[filtered_columns], 3)