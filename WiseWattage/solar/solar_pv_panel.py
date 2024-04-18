import logging
import pandas as pd

def initialise_class(self):
    """
    Initialisation method.
    Loads PV panel data from module_data/pv_module_list.csv if a name is specified,
    and logs a message indicating the creation of the Solar PV Panel.
    """
    if self.load_panel_name is not None:
        load_solar_panel_data(self)

    logging.info(
            "Solar PV Panel created: "
            f"Size: {self.panel_kwp}kW, Size: {self.size_m2}m2, "
            f"Efficiency: {self.eff*100}%, Lifespan: {self.lifespan}yrs"
        )
    logging.info("*******************")



def load_solar_panel_data(self):
    """
    Loads PV panel data from module_data/pv_module_list.csv and assigns
    values to class attributes.
    """
    pv_data = pd.read_csv("module_data/pv_module_list.csv", encoding='iso-8859-1')

    # Select PV module from data based on name
    pv_panel = pv_data[pv_data["Name"] == self.load_panel_name]

    # Assign values to class attributes
    self.panel_kwp = round(pv_panel["STC_Wp"].iloc[0] / 1000, 3)
    self.size_m2 = pv_panel["Area_m2"].iloc[0]
    self.eff = pv_panel["Efficiency"].iloc[0]
    self.cell_temp_coeff = pv_panel["Temp_Coeff"].iloc[0]
    self.cell_NOCT = pv_panel["T_NOCT_C"].iloc[0]

    self.I_sc = pv_panel["I_sc"].iloc[0]
    self.V_oc = pv_panel["V_oc"].iloc[0]
    self.I_mp = pv_panel["I_mp"].iloc[0]
    self.V_mp = pv_panel["V_mp"].iloc[0]

    self.material = pv_panel["Technology"].iloc[0]
    self.length_m = pv_panel["Length_m"].iloc[0]
    self.width_m = pv_panel["Width_m"].iloc[0]
    self.weight_kg = pv_panel["Weight_kg"].iloc[0]
    self.cost = pv_panel["Cost_£"].iloc[0]
