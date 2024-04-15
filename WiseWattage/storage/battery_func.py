import logging
import pandas as pd

def initialise_class(self):
    if self.load_battery_name is None:
        self.useable_capacity = round(self.max_capacity * (1-self.discharge_depth), 3)
        self.max_discharge_kW = self.useable_capacity * self.max_discharge_C
        self.max_charge_kW = self.useable_capacity * self.max_charge_C

    if self.load_battery_name is not None:
        load_battery_data(self)

    logging.info(
            f"Battery Storage Created: Useable Capacity: {self.useable_capacity}kWh, "
            f"Max Discharge: {self.max_discharge_kW}kW,"
            f" Chemistry: {self.chemistry}"
        )
    logging.info("*******************")


def load_battery_data(self):
    bat_data = pd.read_csv("module_data/battery_list.csv")
    battery = bat_data[bat_data["Name"] == self.load_battery_name]
    
    self.max_capacity = battery["Energy_Capacity_kWh"].iloc[0]
    self.chemistry = battery["Chemistry"].iloc[0]
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
