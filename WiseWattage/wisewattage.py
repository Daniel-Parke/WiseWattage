from dataclasses import dataclass, field
from typing import List, Dict
from functools import lru_cache

from WiseWattage.jrc_tmy import get_jrc_tmy
from WiseWattage.solar_pv_model import calc_solar_model, combine_array_results
from WiseWattage.solar_pv_model import total_array_results, pv_stats, pv_stats_grouped, SummaryGrouped

import logging

import pandas as pd
import plotly.express as px
import plotly.io as pio


import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.INFO)  # Set the default log level

# Create a file handler for writing logs to a file
file_handler = logging.FileHandler('WattSunandHomes.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Create a stream handler for writing logs to the terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def main():
    pass

@dataclass
class Site:
    name: str = ""
    address: str = ""
    client: str = ""
    latitude: float = 54.60452
    longitude: float = -5.92860
    tmz_hrs_east: int = 0
    tmy_data: pd.DataFrame = field(default=None, init=False)

    def __post_init__(self):
        self.tmy_data = get_jrc_tmy_cached(self.latitude, self.longitude)
        logging.info(f'TMY data obtained for: {self.latitude}, longitude: {self.longitude}')
        logging.info("*******************")

@lru_cache(maxsize=None)  # Cache TMY results
def get_jrc_tmy_cached(latitude, longitude):
    logging.info(f'Fetching TMY data for latitude: {latitude}, longitude: {longitude}')
    return get_jrc_tmy(latitude, longitude)


@dataclass
class SolarPVArray:
    pv_kwp: float = 1
    surface_pitch: float = 35
    surface_azimuth: float = 0
    lifespan: float = 25
    pv_eol_derating: float = 0.88
    cost_per_kWp: float = 1250
    electrical_eff: float = 0.21
    cell_temp_coeff: float = -0.0035
    transmittance_absorptance: float = 0.9
    refraction_index: float = 0.1
    cell_NOCT: float = 42
    ambient_NOCT: float = 20
    e_poa_NOCT: float = 800
    e_poa_STC: float = 1000
    cell_temp_STC: float = 25


@dataclass
class SolarPVModel:
    site: Site
    arrays: List[SolarPVArray]
    models: List[Dict] = field(default_factory=list)
    all_models: pd.DataFrame = field(default=None, init=False)
    combined_model: pd.DataFrame = field(default=None, init=False)
    summary: pd.DataFrame = field(default=None, init=False)
    summary_grouped: SummaryGrouped = field(default=None, init=False)
    albedo: float = 0.2
    timestep: int = 60

    def __post_init__(self):
        self.model_solar_pv()
        self.summary = pv_stats(self.all_models, self.arrays)
        logging.info("Model statistical analysis completed.")
        self.summary_grouped = pv_stats_grouped(self.all_models)
        logging.info("Model statistical grouping completed.")
        logging.info("*******************")

    # Function to run solar PV model and save to results, all_models and combined_model
    def model_solar_pv(self):
        logging.info("*******************")
        logging.info("Starting model calculations for SolarPVModel.")
        logging.info("*******************")
        models = []
        for array in self.arrays:
            log_message = (f"Simulating model for {array.pv_kwp}kWp, {array.surface_pitch} degrees pitch "
               f"& azimuth at {array.surface_azimuth} degrees WoS")
            logging.info(log_message)
            result = calc_solar_model(self.site.tmy_data, self.site.latitude, self.site.longitude,
                                      array.pv_kwp, array.surface_pitch, array.surface_azimuth,
                                      array.lifespan, array.pv_eol_derating, array.electrical_eff,
                                      self.albedo, array.cell_temp_coeff, array.transmittance_absorptance,
                                      array.refraction_index, array.cell_NOCT, array.ambient_NOCT,
                                      array.e_poa_NOCT, array.e_poa_STC, array.cell_temp_STC, self.timestep,
                                      self.site.tmz_hrs_east)
            models.append({"array_specs": array, "model_result": result})
        logging.info("*******************")
        logging.info("Model calculations completed.")
        self.models = models
        self.all_models = combine_array_results(models)
        logging.info("Model data aggregated.")
        self.combined_model = total_array_results(models)
        logging.info("Model data summary complete.")
        logging.info("*******************")

    def save_model_csv(self):
        if self.all_models is not None:
            self.all_models.to_csv(f"Solar_Modelling_{self.site.name}_Combined.csv")
            logging.info("Model data saved to csv file completed.")
        logging.info("*******************")

    # Returns model for # array at # index location in results
    def array_model(self, n):
        try:
            return self.models[n]["model_result"]
        except IndexError:
            return f"There was no model at index {n}, check the number of arrays modelled and try again."


    def model_summary_html_export(self, freq=None, grouped=True):
        org_col_grouped = ["PV_Gen_kWh_Total", "Panel_POA_kWm2_Total", "IAM_Loss_kWm2_Total", "PV_Thermal_Loss_kWh_Total",
                            "ET_HRad_kWm2_Total", "E_Beam_kWm2_Total", "E_Diffuse_kWm2_Total", "E_Ground_kWm2_Total",
                            "Cell_Temp_C_Avg", "T2m"]
        new_col_grouped = ["PV Generation (kWh)", "POA Radiation (kWh)", "AOI Losses (kWh)", "Thermal Losses (kWh)",
                   "ET Horizontal Radiation", "Beam Radiation (kWh)", "Diffuse Radiation (kWh)", "Ground Radiation (kWh)",
                   "Average Cell Temperature (C)", "Average Ambient Temp (C)"]

        org_col_summary = ["PV_Gen_kWh_Annual", "PV_Gen_kWh_Lifetime", "E_POA_kWm2_Annual", "Panel_POA_kWm2_Annual",
                           "IAM_Loss_kWm2_Annual", "PV_Thermal_Loss_kWh_Annual", "E_Beam_kWm2_Annual", "E_Diffuse_kWm2_Annual",
                           "E_Ground_kWm2_Annual", "ET_HRad_kWm2_Annual", "Cell_Temp_C_Avg", "T2m_Avg"]
        new_col_summary = ["PV Generation (kWh)", "Lifetime PV Generation (kWh)", "E POA (kWhm2)", "Panel POA (kWhm2)",
                   "AOI Reflected Loss (kWhm2)", "PV Thermal Loss (kWh)", "Beam Radiation (kWhm2)", "Diffuse Radiation (kWhm2)",
                   "Ground Radiation (kWhm2)","Average Cell Temperature (C)", "Average Ambient Temp (C)"]

        if grouped==True:
            data = getattr(self.summary_grouped, freq, None)
            data_new = data[org_col_grouped].copy()
            data_new.columns = new_col_grouped
            data_html = data_new.to_html()
            return data_html
        else:
            data = self.summary
            data_new = data[org_col_summary].copy()
            index_mapping = dict(zip(org_col_summary, new_col_summary))
            data_new = data_new.rename(index=index_mapping)
            data_dict = data_new.to_dict()
            return data_dict



    def plot_model(self, params, model_index=0, plot_type="line"):
        model_df = self.models[model_index]["model_result"]

        if params and all(param in model_df.columns for param in params):
            # Dynamically select the Plotly Express plotting function based on plot_type
            plot_func = getattr(px, plot_type.lower(), None)

            if plot_func:
                # Call the plotting function dynamically
                fig = plot_func(model_df, x=model_df.index, y=params[:],
                                title=f"Chart showing modelled value for {params} over a year.")
                return fig
            else:
                print(f"Plot type '{plot_type}' is not supported. Please define a valid plot type e.g., 'line', 'bar'.")
        else:
            print("Error: Invalid parameters or parameters not found in DataFrame columns.")


    def plot_combined(self, params, plot_type="line"):
        # Ensure the DataFrame 'combined_model' has the columns specified in 'params'
        if params and all(param in self.combined_model.columns for param in params):
            # Dynamically select the Plotly Express plotting fuction based on plot_type
            plot_func = getattr(px, plot_type.lower(), None)

            if plot_func:
                # Call the plotting function dynamically
                fig = plot_func(self.combined_model, x=self.combined_model.index, y=params[:],
                                title=f"Chart showing aggregated modelled {params} over a year.")
                return fig
            else:
                print(f"Plot type '{plot_type}' is not supported. Please define a valid plot type e.g., 'line', 'bar'.")
        else:
            print("Error: Invalid parameters or parameters not found in DataFrame columns.")


    def plot_sum(self, params, group="daily", plot_type="line"):
        # Dynamically get the group DataFrame based on the 'group' string parameter
        group_df = getattr(self.summary_grouped, group, None)

        # Check if the group DataFrame exists and has the specified columns
        if group_df is not None and params and all(param in group_df.columns for param in params):
            # Dynamically select the Plotly Express plotting function based on plot_type
            plot_func = getattr(px, plot_type, None)

            if plot_func:
                # Call the plotting function dynamically
                fig = plot_func(group_df, x=group_df.index, y=params[:],
                                title=f"Chart showing modelled {params} grouped by {group} values.")
                return fig
            else:
                print(f"Plot type '{plot_type}' is not supported. Please define a valid plot type e.g., 'line', 'bar'.")
        else:
            print("Error: Invalid parameters, group not found, or parameters not found in DataFrame columns.")


if __name__ == "__main__":
    main()
