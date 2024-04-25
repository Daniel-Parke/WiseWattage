import logging
import numpy as np
import pandas as pd

from collections import OrderedDict
from numba import jit

from solar.SolarPVModel import SolarPVModel
from demand.Load import Load
from utility.Grid import Grid
from utility.Inverter import Inverter
from finance.Finance import Finance


# Initialise list of variables to be used later
pv_model_variables = ["PV_Gen_kWh_Total", "Combined_PV_Losses_kWh_Total"]

columns_to_drop = ["Energy_Use_kWh_Base", "Variability_Factor",
                   "Net_Energy_kWh"]

columns_to_keep = ['Energy_Use_kWh', "Renewable_Energy_Use_kWh", 'Grid_Imports_kWh', 'Grid_Exports_kWh', 
                   'PV_Gen_kWh_Total', 'PV_AC_Output_kWh', 'Consumed_Solar_kWh', 'Excess_Solar_kWh',
                   "Battery_SoC_kWh", "Battery_Charge_kWh", "Battery_Discharge_kWh",
                   'Unused_Energy_kWh', 'Combined_PV_Losses_kWh_Total', 'Inverter_Losses_kWh',
                   'Battery_Losses_kWh', "Inverter_Limited_kWh", "Import_Limited", "Export_Limited",
                   "Grid_Imports_£", "Grid_Exports_£", "Net_Energy_Demand_kWh",
                   "Hour_of_Day", "Day_of_Year", "Week_of_Year", "Month_of_Year"]


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

    self.model["Unused_Energy_kWh"] = 0

    self.model["Hour_of_Day"] = self.site.tmy_data["Hour_of_Day"]
    self.model["Day_of_Year"] = self.site.tmy_data["Day_of_Year"]
    self.model["Week_of_Year"] = self.site.tmy_data["Week_of_Year"]
    self.model["Month_of_Year"] = self.site.tmy_data["Month_of_Year"]

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
        self.model["Inverter_Losses_kWh"] += (self.model["PV_Gen_kWh_Total"]
                                        - self.model["PV_AC_Output_kWh"])
        self.model["Inverter_Limited_kWh"] = (self.model["PV_Gen_kWh_Total"]
                                        * self.inverter.inverter_eff) - self.model["PV_AC_Output_kWh"]

        # Update Net Energy Flows
        self.model["Net_Energy_kWh"] += self.model["PV_AC_Output_kWh"]

        self.model["Consumed_Solar_kWh"] = self.model["Net_Energy_Demand_kWh"] - (
            np.maximum(0, self.model["Net_Energy_Demand_kWh"] - self.model["PV_AC_Output_kWh"]))
        
        self.model["Net_Energy_Demand_kWh"] = np.maximum(0, self.model["Net_Energy_Demand_kWh"]
                                                        - self.model["PV_AC_Output_kWh"])
        
        self.model["Excess_Solar_kWh"] = np.maximum(0, self.model["PV_AC_Output_kWh"] - 
                                                    self.model["Consumed_Solar_kWh"])
        
        self.model["Unused_Energy_kWh"] += self.model["Excess_Solar_kWh"]
        
        
        self.model["Renewable_Energy_Use_kWh"] += self.model["Consumed_Solar_kWh"]
        
        logging.info(
            "Solar PV simulation & energy flow calculations completed"
        )
        logging.info("*******************")



@jit(nopython=True)
def calc_battery_state(n, net_demand, net_energy, initial_soc, max_discharge, max_charge, eff, max_cap,
                        renewable_cons, unused_energy):
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
            losses[i] = charge_power - actual_charge
            charge_amount[i] = charge_power
            current_soc += actual_charge
            net_energy[i] -= charge_power
            unused_energy[i] -= charge_power

        # Ensure SoC does not fall below zero or exceed maximum capacity after each operation
        current_soc = max(min(current_soc, max_cap), 0)
        soc_series[i] = current_soc

    return soc_series, charge_amount, discharge_amount, losses, net_demand, net_energy, renewable_cons, unused_energy


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
        unused_energy = self.model["Unused_Energy_kWh"].values.copy()

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
            renewable_cons,
            unused_energy
        )

        # Update DataFrame columns with the results
        self.model["Battery_SoC_kWh"] = results[0]
        self.model["Battery_Charge_kWh"] = results[1]
        self.model["Battery_Discharge_kWh"] = results[2]
        self.model["Battery_Losses_kWh"] = results[3]
        self.model["Net_Energy_Demand_kWh"] = results[4]
        self.model["Net_Energy_kWh"] = results[5]
        self.model["Renewable_Energy_Use_kWh"] = results[6]
        self.model["Unused_Energy_kWh"] = results[7]

        logging.info(
            "Battery simulation & energy flow calculations completed"
        )
        logging.info("*******************")


def calc_grid_energy_flow(self):
    """
    Simulates the energy flow of the grid and updates the DataFrame
    with the resulting grid imports, exports, and unused energy.
    """
    
    # Calculate grid imports
    if self.grid.import_allow == True and self.grid.offgrid == False:
        # Initialise grid import values to 0
        self.model["Grid_Imports_kWh"] = 0
        self.model["Grid_Imports_£"] = 0
        self.model["Import_Limited"] = 0
        
        # Calculate grid imports where net energy demand is positive
        self.model["Grid_Imports_kWh"] = np.where(
             self.model["Net_Energy_Demand_kWh"] > 0, 
             np.minimum(self.grid.import_limit, 
                        self.model["Net_Energy_Demand_kWh"]), 
             0)

        # Calculate import limited energy
        self.model["Import_Limited"] = self.model["Net_Energy_Demand_kWh"] - self.model["Grid_Imports_kWh"]

        # Update net energy demand and total energy used
        self.model["Net_Energy_Demand_kWh"] -= self.model["Grid_Imports_kWh"]
        self.model["Net_Energy_kWh"] += self.model["Grid_Imports_kWh"]

        # Calculate grid import amount paid
        if self.grid.day_night_tariff == True:
            if self.model["Hour_of_Day"].between(1, 8).any():
                self.model["Grid_Imports_£"] = self.model["Grid_Imports_kWh"] * self.grid.import_night
            else:
                self.model["Grid_Imports_£"] = self.model["Grid_Imports_kWh"] * self.grid.import_day
        elif self.grid.day_night_tariff == False:
            self.model["Grid_Imports_£"] = self.model["Grid_Imports_kWh"] * self.grid.import_standard

    # Calculate grid exports
    if self.grid.export_allow == True and self.grid.offgrid == False:
        # Initialise grid export values to 0
        self.model["Grid_Exports_kWh"] = 0
        self.model["Grid_Exports_£"] = 0
        self.model["Export_Limited"] = 0

        # Calculate grid exports where net energy is positive
        self.model["Grid_Exports_kWh"] = np.where(
             self.model["Net_Energy_kWh"] > 0, 
             np.minimum(self.grid.export_limit, 
                        self.model["Net_Energy_kWh"]), 
             0)

        # Calculate export limited energy
        self.model["Export_Limited"] = self.model["Net_Energy_kWh"] - self.model["Grid_Exports_kWh"]

        # Update unused energy and total energy used
        self.model["Unused_Energy_kWh"] -= self.model["Grid_Exports_kWh"]
        self.model["Net_Energy_kWh"] -= self.model["Grid_Exports_kWh"]

        # Calculate grid export amount paid
        if self.grid.day_night_tariff == True:
            if self.model["Hour_of_Day"].between(1, 8).any():
                self.model["Grid_Exports_£"] = self.model["Grid_Exports_kWh"] * self.grid.export_day
            else:
                self.model["Grid_Exports_£"] = self.model["Grid_Exports_kWh"] * self.grid.export_night
        elif self.grid.day_night_tariff == False:
            self.model["Grid_Exports_£"] = self.model["Grid_Exports_kWh"] * self.grid.export_standard

        

        logging.info(
        "Grid simulation & energy flow calculations completed"
            )
        logging.info("*******************")


def safe_get(data, key):
    """ Safely get a value from a dictionary, returning None if the key is not present. """
    return data.get(key, None)


def model_stats(model_results: pd.DataFrame, arrays: list) -> pd.Series:
    """
    Generates a summary of key PV performance metrics over the specified period.

    Parameters:
        model_results (pd.DataFrame): DataFrame containing model results.
        arrays (list): List of PV array configurations used in the model.

    Returns:
        pd.Series: Series with summarized PV performance metrics.
    """
    # Columns to sum
    columns_to_sum = {'Energy_Use_kWh', 'Renewable_Energy_Use_kWh', 'Grid_Imports_kWh',
       'Grid_Exports_kWh', 'PV_Gen_kWh_Total', 'PV_AC_Output_kWh',
       'Consumed_Solar_kWh', 'Excess_Solar_kWh',
       'Battery_Charge_kWh', 'Battery_Discharge_kWh', 'Unused_Energy_kWh',
       'Combined_PV_Losses_kWh_Total', 'Inverter_Losses_kWh',
       'Battery_Losses_kWh', 'Inverter_Limited_kWh', 'Import_Limited',
       'Export_Limited', 'Grid_Imports_£', 'Grid_Exports_£', "Net_Energy_Demand_kWh"}

    # Columns to calculate the mean
    columns_to_mean = {'Battery_SoC_kWh'}

    # Check which columns exist in the DataFrame
    existing_sum_columns = columns_to_sum.intersection(model_results.columns)
    existing_mean_columns = columns_to_mean.intersection(model_results.columns)

    # Initialize a dictionary to hold the summary
    summary = {}

    # Sum the existing specified columns
    for col in existing_sum_columns:
        summary[col] = model_results[col].sum()

    # Calculate the mean for the existing specified columns
    for col in existing_mean_columns:
        summary[col] = model_results[col].mean()


    # Use the safe_get function to handle potentially missing entries in the summary
    summary["Energy_Use_kWh_Annual"] = safe_get(summary, "Energy_Use_kWh")
    summary["Renewable_Energy_Use_kWh_Annual"] = safe_get(summary, "Renewable_Energy_Use_kWh")
    summary["Grid_Imports_kWh_Annual"] = safe_get(summary, "Grid_Imports_kWh")
    summary["Grid_Imports_£_Annual"] = safe_get(summary, "Grid_Imports_£")
    summary["Grid_Exports_kWh_Annual"] = safe_get(summary, "Grid_Exports_kWh")
    summary["Grid_Exports_£_Annual"] = safe_get(summary, "Grid_Exports_£")
    summary["PV_DC_Output_kWh_Annual"] = safe_get(summary, "PV_Gen_kWh_Total")
    summary["PV_AC_Output_kWh_Annual"] = safe_get(summary, "PV_AC_Output_kWh")
    summary["Consumed_Solar_kWh_Annual"] = safe_get(summary, "Consumed_Solar_kWh")
    summary["Excess_Solar_kWh_Annual"] = safe_get(summary, "Excess_Solar_kWh")
    summary["Battery_SoC_Avg_kWh"] = safe_get(summary, "Battery_SoC_kWh")
    summary["Battery_Charge_kWh_Annual"] = safe_get(summary, "Battery_Charge_kWh")
    summary["Battery_Discharge_kWh_Annual"] = safe_get(summary, "Battery_Discharge_kWh")
    summary["Unused_Energy_kWh_Annual"] = safe_get(summary, "Unused_Energy_kWh")
    summary["Combined_PV_Losses_kWh_Annual"] = safe_get(summary, "Combined_PV_Losses_kWh_Total")
    summary["Inverter_Losses_kWh_Annual"] = safe_get(summary, "Inverter_Losses_kWh")
    summary["Battery_Losses_kWh_Annual"] = safe_get(summary, "Battery_Losses_kWh")
    summary["Inverter_Limited_kWh_Annual"] = safe_get(summary, "Inverter_Limited_kWh")
    summary["Import_Limited_Annual"] = safe_get(summary, "Import_Limited")
    summary["Export_Limited_Annual"] = safe_get(summary, "Export_Limited")
    summary["Net_Energy_Demand_kWh_Annual"] = safe_get(summary, "Net_Energy_Demand_kWh")

    # Define the desired order of keys
    desired_order = [
        'Energy_Use_kWh_Annual', 
        'Renewable_Energy_Use_kWh_Annual', 
        'Grid_Imports_kWh_Annual',
        'Grid_Imports_£_Annual',
        'Grid_Exports_kWh_Annual', 
        'Grid_Exports_£_Annual',
        'PV_DC_Output_kWh_Annual', 
        'PV_AC_Output_kWh_Annual',
        'Consumed_Solar_kWh_Annual', 
        'Excess_Solar_kWh_Annual', 
        "Battery_SoC_Avg_kWh",
        'Battery_Charge_kWh_Annual', 
        'Battery_Discharge_kWh_Annual', 
        'Unused_Energy_kWh_Annual',
        'Combined_PV_Losses_kWh_Annual', 
        'Inverter_Losses_kWh_Annual',
        'Battery_Losses_kWh_Annual', 
        'Inverter_Limited_kWh_Annual', 
        'Import_Limited_Annual',
        'Export_Limited_Annual',
        "Net_Energy_Demand_kWh_Annual"
        ]

    # Create an OrderedDict filtering out None values
    ordered_summary = OrderedDict((k, summary[k]) for k in desired_order if summary[k] is not None)

    # Convert to pandas Series and round the values
    summary_series = pd.Series(ordered_summary).round(3)

    return summary_series


class SummaryGrouped:
    """
    Organizes grouped summary statistics of PV system performance.
    
    Parameters:
        summaries (dict): Dictionary with time grouping as keys and summary statistics DataFrames as values.
    """

    def __init__(self, summaries):
        for key, df in summaries.items():
            setattr(self, key.lower(), df)


def model_grouped(model_results: pd.DataFrame) -> SummaryGrouped:
    """
    Generates grouped statistics of PV performance over different time frames.

    Parameters:
        model_results (pd.DataFrame): DataFrame containing model results.

    Returns:
        SummaryGrouped: Object containing DataFrames of grouped statistics.
    """
     # Add a column for Quarterly grouping based on 'Month_of_Year'
    if "Month_of_Year" in model_results.columns:
        model_results['Quarter_of_Year'] = model_results["Month_of_Year"].apply(lambda x: (x - 1) // 3 + 1)
    
    # Define the groupings for different human timeframes
    groupings = {
        "Hourly": "Hour_of_Day",
        "Daily": "Day_of_Year",
        "Weekly": "Week_of_Year",
        "Monthly": "Month_of_Year",
        "Quarterly": "Quarter_of_Year"  # Use the new column for Quarterly grouping
    }

    # Columns to sum and to calculate the mean
    columns_to_sum = ['Energy_Use_kWh', 'Renewable_Energy_Use_kWh', 'Grid_Imports_kWh',
       'Grid_Exports_kWh', 'PV_Gen_kWh_Total', 'PV_AC_Output_kWh',
       'Consumed_Solar_kWh', 'Excess_Solar_kWh',
       'Battery_Charge_kWh', 'Battery_Discharge_kWh', 'Unused_Energy_kWh',
       'Combined_PV_Losses_kWh_Total', 'Inverter_Losses_kWh',
       'Battery_Losses_kWh', 'Inverter_Limited_kWh', 'Import_Limited',
       'Export_Limited', 'Grid_Imports_£', 'Grid_Exports_£', "Net_Energy_Demand_kWh"]
    
    columns_to_mean = ["Battery_SoC_kWh"]

    summaries = {}

    # Gets Hourly and Hour of Day from .items() tuple list
    for timeframe, group_by in groupings.items():
        if group_by and group_by in model_results.columns:
            grouped = model_results.groupby(group_by)

            # Summing specified columns and rounding
            summed = round(grouped[columns_to_sum].sum(), 3)

            # Calculating the mean for specified columns and rounding
            meaned = round(grouped[columns_to_mean].mean(), 3)

            # Combine the summed and meaned results into a single DataFrame
            summary_df = pd.concat([summed, meaned], axis=1)

            # Adds summary dataframe to dictionary with timeframe key
            summaries[timeframe] = summary_df

    # Return an instance of SummaryGrouped with summaries as attributes
    return SummaryGrouped(summaries)


def calc_finance_model(self):
    """
    Calculates the finance model for the energy system.
    """
    self.finance = Finance()



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



def calculate_capex(self):
    # Caluclate inverter CAPEX and replacement costs
    if self.inverter is not None:
        if self.inverter.cost is None:
            if self.pv_model is not None:
                self.inverter.cost = (self.inverter.cost_per_kw * self.pv_model.summary.Total_PV_Capacity_kWp)

        if self.inverter.cost is not None:
            self.capex += self.inverter.cost
            self.replacement_capex = np.maximum(round(self.inverter.cost * (
                (self.project_lifespan / self.inverter.lifespan) - 1), 2), 0)

    # Calculate Battery CAPEX and replacement costs
    if self.battery is not None and self.battery.cost is not None:
        self.capex += self.battery.cost
        lifetime_throughput = (
            self.summary.Battery_Charge_kWh_Annual * self.project_lifespan)
        lifespan_throughput = (self.battery.life_cycles * self.battery.max_capacity)
        throughput_factor = ((lifetime_throughput / lifespan_throughput) - 1)
        lifespan_factor = ((self.project_lifespan / self.battery.lifespan) - 1)

        # First, calculate the maximum of the two factors
        max_cost_factor = max(round(self.battery.cost * throughput_factor, 2), 
                            round(self.battery.cost * lifespan_factor, 2))
        # Ensure the value is not negative
        self.replacement_capex += max(max_cost_factor, 0)

    # Calculate Solar PV CAPEX and Replacement costs
    if self.pv_model is not None and self.pv_model.cost is not None:
        self.capex += self.pv_model.cost 
        self.replacement_capex += np.maximum(round(self.pv_model.cost * (
            (self.project_lifespan / self.pv_model.arrays[0].pv_panel.lifespan) - 1), 2), 0)


def calculate_operation_costs(self):
    if self.grid is not None:
        # Calculate lifetime degradation factor
        if self.pv_model is not None:
            degrade_factor = self.pv_model.summary.PV_Gen_kWh_Lifetime / (
                self.pv_model.summary.PV_Gen_kWh_Annual * self.project_lifespan
                )
            
        elif self.pv_model is None:
            degrade_factor = 1
        
        total_imports_value = self.summary['Grid_Imports_£_Annual'] * self.project_lifespan / degrade_factor

        if self.summary.Grid_Imports_kWh_Annual > 0:
            total_exports_value = self.summary['Grid_Exports_£_Annual'] * self.project_lifespan * degrade_factor
        if self.summary.Grid_Imports_kWh_Annual == 0:
            total_exports_value = 0

        self.opex_cost = round(total_imports_value, 2)
        self.export_value = round(total_exports_value, 2)


def calculate_npc(self):
    self.npc = round(self.capex + self.opex_cost + self.replacement_capex - self.export_value, 2)
