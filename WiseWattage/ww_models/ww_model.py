import logging
import numpy as np

from numba import jit

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
    """
    Initialise the main model with the provided data.
    
    If no Grid or Inverter variables are provided, 
    they are initialised as empty objects.
    
    If no Energy Demand is provided, it is initialised
    using the Load object.
    
    The SolarPVModel object is initialised if arrays are provided,
    and the load profile is copied, modified to represent net energy
    demand, and updated with the required values.
    
    The following variables are added to the DataFrame:
    
    * Inverter_Losses_kWh - Total inverter losses
    * Inverter_Limited_kWh - Energy lost due to inverter limitation
    * Grid_Imports_kWh - Energy imported from grid
    * Grid_Exports_kWh - Energy exported to grid
    * Import_Limited - Number of times import was limited
    * Export_Limited - Number of times export was limited
    * Unused_Energy_kWh - Energy not used due to insufficient grid capacity
    """

    # Initialise Grid and Inverter Variables if not provided
    if self.grid is None:
        self.grid = Grid()

    if self.inverter is None:
        self.inverter = Inverter()

    # Initialise Energy Demand if not provided and update Energy Flows
    if self.load is None:
        self.load = Load()

    self.model = self.load.load_profile.copy()

    self.model["Net_Energy_kWh"] = -self.load.load_profile["Energy_Use_kWh"]
    self.model["Net_Energy_Demand_kWh"] = self.load.load_profile["Energy_Use_kWh"]
    self.model["Renewable_Energy_Use_kWh"] = 0

    self.model["Inverter_Losses_kWh"] = 0
    self.model["Inverter_Limited_kWh"] = 0

    self.model["Grid_Imports_kWh"] = 0
    self.model["Grid_Exports_kWh"] = 0
    self.model["Import_Limited"] = 0
    self.model["Export_Limited"] = 0
    self.model["Unused_Energy_kWh"] = 0

    # Run SolarPVModel, save entire results and add required values to model.model datafarme
    if self.arrays is not None:
        self.pv_model = SolarPVModel(self.site, self.arrays)

    logging.info(
            "Main Model Successfully Initialised"
        )
    logging.info("*******************")
    

        
def calc_solar_energy_flow(self, pv_model_variables=pv_model_variables, mo=99999999):
    """
    Calculates the solar PV energy flow and updates the DataFrame with the results.

    The PV model variables are joined to the main model DataFrame and then
    the AC output of the inverter is calculated based on the PV generation
    and the inverter efficiency. The inverter losses are also calculated.
    
    The net energy flow is then updated based on the PV AC output and excess
    solar energy is calculated as the difference between PV AC output
    and energy use. Unused energy is calculated as the excess solar energy
    that is not used to meet energy demand. Consumed solar energy is the
    difference between PV AC output and excess solar energy.

    Parameters
    ----------
    pv_model_variables : list, optional
        The variables from the solar PV model to add to the main model, by default pv_model_variables
    mo : float, optional
        The maximum output of the inverter in kW, by default 99999999
    """
    if self.pv_model is not None:
        # Join the PV model variables to the main model DataFrame
        pv_data = self.pv_model.combined_model[pv_model_variables]
        self.model = self.model.join(pv_data, how='left')

        # Calculate the inverter AC output and losses
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
        
        logging.info(
            "Solar PV simulation & energy flow calculations completed"
        )
        logging.info("*******************")



@jit(nopython=True)
def calc_battery_state(n, net_demand, net_energy, initial_soc, max_discharge, max_charge, eff, max_cap,
                        renewable_cons):
    """
    Calculates the state of charge of a battery over time given a set of inputs.

    Parameters
    ----------
    n : int
        Number of time steps (rows) in the input data frames.
    net_demand : ndarray
        Array of the net demand (kWh) for each time step.
    net_energy : ndarray
        Array of the available renewable energy (kWh) for each time step.
    initial_soc : float
        Initial state of charge of the battery (kWh).
    max_discharge : float
        Maximum discharge rate of the battery (kW).
    max_charge : float
        Maximum charge rate of the battery (kW).
    eff : float
        Efficiency of the inverter (%).
    max_cap : float
        Maximum capacity of the battery (kWh).
    renewable_cons : ndarray
        Array of the renewable energy used at each time step (kWh).

    Returns
    -------
    soc_series : ndarray
        Array of the state of charge of the battery at each time step (kWh).
    charge_amount : ndarray
        Array of the amount of energy charged into the battery at each time step (kWh).
    discharge_amount : ndarray
        Array of the amount of energy discharged from the battery at each time step (kWh).
    losses : ndarray
        Array of the amount of energy lost during charging and discharging at each time step (kWh).
    net_demand : ndarray
        Array of the updated net demand at each time step (kWh).
    net_energy : ndarray
        Array of the updated available renewable energy at each time step (kWh).
    renewable_cons : ndarray
        Array of the updated renewable energy used at each time step (kWh).
    """
    soc_series = np.zeros(n)
    charge_amount = np.zeros(n)
    discharge_amount = np.zeros(n)
    losses = np.zeros(n)
    current_soc = initial_soc

    for i in range(n):
        # Discharging
        if net_demand[i] > 0 and current_soc > 0:
            # The potential discharge is the lesser of the demand, the battery's available energy,
            # and the battery's max discharge rate
            discharge_power = min(net_demand[i] / eff, current_soc, max_discharge)
            # Account for inverter efficiency during discharging
            actual_discharge = discharge_power * eff

            discharge_amount[i] = actual_discharge
            losses[i] = discharge_power - actual_discharge
            current_soc -= discharge_power
            net_demand[i] -= actual_discharge
            net_energy[i] += actual_discharge
            renewable_cons[i] += actual_discharge

        # Charging
        if net_energy[i] > 0 and current_soc < max_cap:
            # The potential charge is the lesser of the excess energy, the battery's remaining capacity, and the max charge rate
            charge_power = min(net_energy[i], max_cap - current_soc, max_charge)
            # Account for inverter efficiency during charging
            actual_charge = charge_power * eff
            charge_amount[i] = charge_power
            current_soc += actual_charge
            net_energy[i] -= charge_power

        # Ensure SoC does not fall below zero or exceed maximum capacity after each operation
        current_soc = max(min(current_soc, max_cap), 0)
        soc_series[i] = current_soc

    return soc_series, charge_amount, discharge_amount, losses, net_demand, net_energy, renewable_cons

def calc_battery_energy_flow(self):
    """
    Simulates the energy flow of the battery and updates the DataFrame
    with the resulting battery state of charge, charge/discharge amounts,
    losses, and updated net energy and renewable energy use.
    """
    if self.battery is not None:
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
            self.battery.useable_capacity * self.battery.initial_charge,
            self.battery.max_discharge_kW,
            self.battery.max_charge_kW,
            self.inverter.inverter_eff,
            self.battery.useable_capacity,
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

        logging.info(
            "Battery simulation & energy flow calculations completed"
        )
        logging.info("*******************")


def calc_grid_energy_flow(self):
    """
    Simulates the energy flow of the grid and updates the DataFrame
    with the resulting grid imports, exports, and unused energy.
    """
    if self.grid.import_allow == True:
        # Calculate grid imports where net energy demand is positive
        self.model["Grid_Imports_kWh"] = np.where(self.model["Net_Energy_Demand_kWh"] > 0,
                                                np.minimum(self.grid.import_limit, 
                                                           self.model["Net_Energy_Demand_kWh"]), 
                                                0)
        self.model["Import_Limited"] = self.model["Net_Energy_Demand_kWh"] - self.model["Grid_Imports_kWh"]
        self.model["Net_Energy_Demand_kWh"] -= self.model["Grid_Imports_kWh"]
        self.model["Net_Energy_kWh"] += self.model["Grid_Imports_kWh"]
        
        # Calculate unused energy in case of excess grid imports
        self.model["Unused_Energy_kWh"] += self.model["Grid_Imports_kWh"]

    # Calculate grid exports where net energy demand is negative
    if self.grid.export_allow == True:
        self.model["Grid_Exports_kWh"] = np.where(
             self.model["Net_Energy_kWh"] > 0, np.minimum(
                  self.grid.export_limit, self.model["Net_Energy_kWh"]), 
                  0)
        
        self.model["Export_Limited"] = self.model["Net_Energy_kWh"] - self.model["Grid_Exports_kWh"]
        self.model["Unused_Energy_kWh"] -= self.model["Grid_Exports_kWh"]
        self.model["Net_Energy_kWh"] -= self.model["Grid_Exports_kWh"]
        
        # Calculate unused energy in case of excess grid exports
        self.model["Unused_Energy_kWh"] += self.model["Grid_Exports_kWh"]

        logging.info(
        f"Grid simulation & energy flow calculations completed"
            )
        logging.info("*******************")


def sort_columns(self, columns_to_keep=columns_to_keep, columns_to_drop=columns_to_drop):
        # First, drop specified columns if any are given and they exist in the DataFrame
        self.model.drop(columns=[col for col in columns_to_drop if col in self.model.columns],
                            axis=1, inplace=True)

        # Reorder columns by filtering the list to include only those that are present in the DataFrame
        filtered_columns = [col for col in columns_to_keep if col in self.model.columns]
        self.model = round(self.model[filtered_columns], 3)

        logging.info(
        f"Model aggregation completed successfully"
            )
        logging.info("*******************")