import logging
import pandas as pd

def initialise_class(self):
    """
    Initialise the Battery class with the battery data based on the
    load_battery_name

    If load_battery_name is None, use the default values for the battery
    and calculate the useable capacity, max discharge and max charge

    Args:
        None

    Returns:
        None
    """
    if self.load_battery_name is None:
        # If no battery name is provided, use the default values
        # Set useable capacity based on discharge depth
        self.useable_capacity = round(self.max_capacity * (1-self.discharge_depth), 3)
        # Set max discharge based on useable capacity and discharge C rating
        self.max_discharge_kW = self.useable_capacity * self.max_discharge_C
        # Set max charge based on useable capacity and charge C rating
        self.max_charge_kW = self.useable_capacity * self.max_charge_C

    if self.load_battery_name is not None:
        # If a battery name is provided, load the data from the battery list
        load_battery_data(self)

    logging.info(
        """Battery Storage Created:
        Useable Capacity: %skWh,
        Max Discharge: %skW,
        Chemistry: %s
        """ % (self.useable_capacity, self.max_discharge_kW, self.chemistry)
    )
    logging.info("*******************")


def load_battery_data(self):
    """
    Loads battery data from module_data/battery_list.csv
    based on the provided battery name

    Args:
        None

    Returns:
        None
    """
    bat_data = pd.read_csv("module_data/battery_list.csv")
    battery = bat_data[bat_data["Name"] == self.load_battery_name]
    
    self.max_capacity = battery["Energy_Capacity_kWh"].iloc[0]
    """Maximum battery capacity in kWh"""
    self.chemistry = battery["Chemistry"].iloc[0]
    """Chemistry of the battery"""
    self.max_discharge_C = battery["C_Rate"].iloc[0]
    """Max discharge rate of the battery in C"""
    self.max_discharge_kW = battery["Max_Continuous_Discharge_kW"].iloc[0]
    """Max discharge rate of the battery in kW"""
    self.max_charge_kW = round(self.max_capacity * self.max_charge_C, 3)
    """Max charge rate of the battery in kW"""
    self.useable_capacity = round(self.max_capacity * self.discharge_depth, 3)
    """Useable capacity of the battery in kWh"""
    
    if self.chemistry == "LiFePO4":
        """If the battery is a LiFePO4 battery"""
        self.nominal_voltage = 51.2
        """Nominal voltage of the battery in Volts"""
        self.cutoff_voltage = 48.8
        """Cut-off voltage of the battery in Volts"""
        self.rated_capacity = round(self.max_capacity / self.nominal_voltage * 1000, 3)
        """Rated capacity of the battery in Ah"""
        
    if self.chemistry == "Li-ion":
        """If the battery is a Li-ion battery"""
        self.nominal_voltage = 58.2
        """Nominal voltage of the battery in Volts"""
        self.cutoff_voltage = 48
        """Cut-off voltage of the battery in Volts"""
        self.rated_capacity = round(self.max_capacity / self.nominal_voltage * 1000, 3)
        """Rated capacity of the battery in Ah"""
