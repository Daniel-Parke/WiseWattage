import logging
import pandas as pd

def initialise_class(self):
    """
    Initialise the Battery class with the battery data based on the
    load_battery_name

    If load_battery_name is None, use the default values for the battery
    and calculate the useable capacity, max discharge and max charge
    """
    if self.load_battery_name is None:
        # If no battery name is provided, use the default values
        # Set useable capacity based on discharge depth
        self.useable_capacity = round(self.max_capacity * self.discharge_depth, 3)
        # Set max discharge based on useable capacity and discharge C rating
        self.max_discharge_kW = self.useable_capacity * self.max_discharge_C
        # Set max charge based on useable capacity and charge C rating
        self.max_charge_kW = self.useable_capacity * self.max_charge_C

    if self.load_battery_name is not None:
        # If a battery name is provided, load the data from the battery list
        load_battery_data(self)

    logging.info(
        """Battery Storage Created: Useable Capacity: %skWh, Max Discharge: %skW, Chemistry: %s""" % (self.useable_capacity, self.max_discharge_kW, self.chemistry)
    )
    logging.info("*******************")


def load_battery_data(self):
    """
    Loads battery data from module_data/battery_list.csv
    based on the provided battery name
    """
    bat_data = pd.read_csv("module_data/battery_list.csv")
    battery = bat_data[bat_data["Name"] == self.load_battery_name]
    
    self.max_capacity = battery["Energy_Capacity_kWh"].iloc[0]
    self.chemistry = battery["Chemistry"].iloc[0]
    self.cost = battery["Cost_£"].iloc[0]

    self.max_discharge_C = battery["C_Rate"].iloc[0]
    self.max_discharge_kW = battery["Max_Continuous_Discharge_kW"].iloc[0]
    self.max_charge_kW = round(self.max_capacity * self.max_charge_C, 3)
    self.useable_capacity = round(self.max_capacity * self.discharge_depth, 3)
    
    if self.chemistry == "LiFePO4":
        self.nominal_voltage = 51.2
        self.cutoff_voltage = 48.8
        self.rated_capacity = round(self.max_capacity / self.nominal_voltage * 1000, 3)
        
    if self.chemistry == "Li-ion":
        self.nominal_voltage = 58.2
        self.cutoff_voltage = 48
        self.rated_capacity = round(self.max_capacity / self.nominal_voltage * 1000, 3)
