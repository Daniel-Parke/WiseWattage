import logging
import pandas as pd

def initialise_class(self):
    if self.load_panel_name is not None:
        load_solar_panel_data(self)

    logging.info(
            f"Solar PV Panel created: Size: {self.panel_kwp}kW, Size: {self.size_m2}m2,"
            f" Efficiency: {self.eff*100}%, Lifespan: {self.lifespan}yrs"
        )
    logging.info("*******************")


def load_solar_panel_data(self):
    pv_data = pd.read_csv("module_data/pv_module_list.csv")
    pv_panel = pv_data[pv_data["Name"] == self.load_panel_name]
    self.panel_kwp = round(pv_panel["STC_Wp"].iloc[0] / 1000, 3)
    self.size_m2 = pv_panel["Area_m2"].iloc[0]
    self.eff = pv_panel["Efficiency"].iloc[0]
    self.cell_temp_coeff = pv_panel["Temp_Coeff"].iloc[0]
    self.cell_NOCT = pv_panel["T_NOCT_C"].iloc[0]
    self.I_sc = pv_panel["I_sc"].iloc[0]
    self.V_oc = pv_panel["V_oc"].iloc[0]
    self.I_mp = pv_panel["I_mp"].iloc[0]
    self.V_mp = pv_panel["V_mp"].iloc[0]
    self.material = pv_panel["Temp_Coeff"].iloc[0]
    self.length_m = pv_panel["Width_m"].iloc[0]
    self.width_m = pv_panel["Technology"].iloc[0]
