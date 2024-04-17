import logging
import pandas as pd

def initialise_class(self):
    """
    Function to initialise the class values and set defaults where needed

    The function checks the inverter's efficiency, maximum output and battery charging
    status. If these values are not set it sets them to the default values.

    It then loads the inverter data from the csv file and logs the inverter creation
    details.
    """
    if self.max_output is not None:
            # Set the standing power to 0.3% of the max output if not already set
            if self.standing_power_kW is None:
                self.standing_power_kW = self.max_output * 0.003
    if self.load_inverter_name is not None:
        load_inverter_data(self)

    logging.info(
            "Inverter Created: "
            f"Max Output: {self.max_output}kW, "
            f"Inverter Efficiency: {self.inverter_eff}%,"
            f" Battery Charging: {self.battery_charging}"
        )
    logging.info("*******************")


def load_inverter_data(self):
    """
    Loads inverter data from the csv file. 

    The function reads in the battery_inverter_list.csv file and extracts the
    data relevant to the inverter specified by the user.

    The function sets the inverter's efficiency, maximum output, standing power,
    minimum voltage input, maximum voltage input, and output voltage.

    If the inverter is not a PV & Battery inverter, the battery charging status
    is set to False.

    Args:
        None

    Returns:
        None
    """
    inv_data = pd.read_csv("module_data/battery_inverter_list.csv")
    inverter = inv_data[inv_data["Name"] == self.load_inverter_name]

    self.inverter_eff = round(inverter["Avg_Efficiency"].iloc[0] / 100, 3)
    self.max_output = inverter["Max_Output_Power_kW"].iloc[0]
    self.standing_power_kW = inverter["Night_Tare_Loss_W"].iloc[0]
    self.min_voltage_in = inverter["Vdc_Minimum"].iloc[0]
    self.max_voltage_in = inverter["Vdc_Maximum"].iloc[0]
    self.voltage_out = inverter["Vac_Nominal"].iloc[0]

    if inverter["PV_and_Battery"].iloc[0] == "N":
        self.battery_charging = False
