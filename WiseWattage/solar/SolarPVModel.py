from dataclasses import dataclass, field
from typing import List, Dict
import logging
import os
import pandas as pd
import pickle

from meteo.Site import Site
from solar.SolarPVArray import SolarPVArray
from solar.solar_pv_model import calc_solar_model, combine_array_results
from solar.solar_pv_model import (
    total_array_results,
    pv_stats,
    pv_stats_grouped,
    SummaryGrouped,
)


@dataclass
class SolarPVModel:
    """
    A class representing a Solar PV Model.

    Attributes:
        site (Site): Information about the site.
        arrays (List[SolarPVArray]): List of SolarPVArray objects.
        models (List[Dict]): List of dictionaries containing model results for each array.
        all_models (pd.DataFrame): DataFrame containing all model results.
        combined_model (pd.DataFrame): DataFrame containing combined model results.
        summary (pd.DataFrame): DataFrame containing summary of model results.
        summary_grouped (SummaryGrouped): Object containing grouped summary of model results.
    """

    site: Site
    arrays: List[SolarPVArray]
    models: List[Dict] = field(default_factory=list)
    all_models: pd.DataFrame = field(default=None, init=False)
    combined_model: pd.DataFrame = field(default=None, init=False)
    summary: pd.DataFrame = field(default=None, init=False)
    summary_grouped: SummaryGrouped = field(default=None, init=False)


    def __post_init__(self):
        """
        Post-initialization method.
        Runs model_solar_pv method to perform solar PV modeling and generate summaries.
        """
        # Run solar PV model
        self.model_solar_pv()

        # Generate summary statistics
        self.summary = pv_stats(self.all_models, self.arrays)
        logging.info("Solar PV model statistical analysis completed.")

        # Generate grouped summary statistics
        self.summary_grouped = pv_stats_grouped(self.all_models)
        logging.info("Solar PV model statistical grouping completed.")
        logging.info("*******************")
        logging.info("Solar PV simulation and modelling completed.")
        logging.info("*******************")

    def model_solar_pv(self):
        """
        Method to perform solar PV modeling and generate model results.
        """
        logging.info("*******************")
        logging.info("Starting Solar PV model simulations.")
        logging.info("*******************")
        models = []
        for array in self.arrays:
            log_message = (f"Simulating model - PV Size: {array.pv_kwp}kWp, Pitch: {array.surface_pitch} degrees, "
                           f"Azimuth {array.surface_azimuth} degrees WoS")
            logging.info(log_message)
            result = calc_solar_model(
                self.site.tmy_data,
                self.site.latitude,
                self.site.longitude,
                array.pv_kwp,
                array.surface_pitch,
                array.surface_azimuth,
                array.lifespan,
                array.pv_eol_derating,
                array.albedo,
                array.cell_temp_coeff,
                array.refraction_index,
                array.e_poa_STC,
                array.cell_temp_STC,
                self.site.timestep,
                self.site.tmz_hrs_east,
            )
            models.append({"array_specs": array, "model_result": result})
        
        # Arrange model results into class structure
        logging.info("*******************")
        logging.info("Solar PV model simulations completed.")
        self.models = models
        self.all_models = combine_array_results(models)
        logging.info("Solar PV model data aggregated.")
        self.combined_model = total_array_results(models)
        logging.info("Solar PV model data summary complete.")
        logging.info

    def save_model_csv(self):
        """
        Method to save model results to a CSV file.
        """
        if self.all_models is not None:
            self.all_models.to_csv(f"Solar_Modelling_{self.site.name}_Combined.csv")
            logging.info("Model data saved to csv file completed.")
            logging.info("*******************")
        else:
            logging.info("Model data NOT saved, no model results found.")
            logging.info("*******************")

    # Returns model for # array at # index location in results
    def array_model(self, n):
        """
        Returns model for a specific array at a given index location in results.

        Args:
            n (int): Index of the array.

        Returns:
            str or dict: Model result for the array.
        """
        try:
            return self.models[n]["model_result"]
        except IndexError:
            return f"There was no model at index {n}, check the number of arrays modelled and try again."


    def model_summary_html_export(self, freq=None, grouped=True):
        """
        Export model summary to HTML format.

        Args:
            freq (str): Frequency for grouped summary (e.g., 'daily', 'monthly').
            grouped (bool): Whether to export grouped summary or not.

        Returns:
            str or dict: HTML-formatted summary data or dictionary of summary data.
        """
        try:
            org_col_grouped = [
                "PV_Gen_kWh_Total",
                "Panel_POA_Wm2_Total",
                "IAM_Loss_Wm2_Total",
                "PV_Thermal_Loss_kWh_Total",
                "ET_HRad_Wm2_Total",
                "E_Beam_Wm2_Total",
                "E_Diffuse_Wm2_Total",
                "E_Ground_Wm2_Total",
                "Array_Temp_C_Avg",
                "T2m",
            ]
            new_col_grouped = [
                "PV Generation (kWh)",
                "POA Radiation (Wh)",
                "AOI Losses (Wh)",
                "Thermal Losses (kWh)",
                "ET Horizontal Radiation (Wh)",
                "Beam Radiation (Wh)",
                "Diffuse Radiation (Wh)",
                "Ground Radiation (Wh)",
                "Average Cell Temperature (C)",
                "Average Ambient Temp (C)",
            ]

            org_col_summary = [
                "PV_Gen_kWh_Annual",
                "PV_Gen_kWh_Lifetime",
                "E_POA_Wm2_Annual",
                "Panel_POA_Wm2_Annual",
                "IAM_Loss_Wm2_Annual",
                "PV_Thermal_Loss_kWh_Annual",
                "E_Beam_Wm2_Annual",
                "E_Diffuse_Wm2_Annual",
                "E_Ground_Wm2_Annual",
                "ET_HRad_Wm2_Annual",
                "Array_Temp_C_Avg",
                "T2m_Avg",
            ]
            new_col_summary = [
                "PV Generation (kWh)",
                "Lifetime PV Generation (kWh)",
                "E POA (Whm2)",
                "Panel POA (Whm2)",
                "AOI Reflected Loss (Whm2)",
                "PV Thermal Loss (kWh)",
                "Beam Radiation (Whm2)",
                "Diffuse Radiation (Whm2)",
                "Ground Radiation (Whm2)",
                "Average Cell Temperature (C)",
                "Average Ambient Temp (C)",
            ]

            if grouped:
                data = getattr(self.summary_grouped, freq, None)
                if data is None:
                    raise ValueError(f"No grouped summary data available for frequency '{freq}'.")
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
        except Exception as e:
            logging.info(f"Error exporting model summary: {e}")
            logging.info("*******************")
            return None


    def save_model(self, name="Solar_Model_Results.wwm"):
        # Replace whitespace in site name with underscores and ensure the file name ends with ".wwm"
        site_name_formatted = self.site.name.replace(" ", "_") if self.site.name else ""
        if not name.endswith(".wwm"):
            name = os.path.splitext(name)[0] + ".wwm"
        # Conditionally construct full_name to avoid leading underscore when site_name_formatted is empty
        full_name = f"saved_models/{site_name_formatted + '_' if site_name_formatted else ''}{name}"

        try:
            abs_path = os.path.abspath(full_name)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            
            # Open the file in binary write mode
            with open(abs_path, 'wb') as filehandler_save:
                pickle.dump(self, filehandler_save)
            logging.info(f"Model for {self.site.name} saved successfully to '{abs_path}'")
            logging.info("*******************")
        except Exception as e:
            logging.info(f"Error saving model to '{full_name}': {e}")
            logging.info("*******************")
