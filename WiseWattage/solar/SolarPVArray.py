from dataclasses import dataclass
import logging
import numpy as np

from solar.SolarPVPanel import SolarPVPanel


# Solar PV Array Class


@dataclass
class SolarPVArray:
    """
    A class representing a Solar PV Array.

    Attributes:
        pv_panels (List[SolarPVPanel]): List of Solar PV Panels.
        pv_kwp (float): Rated power of the array in kilowatts peak (kWp).
        surface_pitch (float): Tilt angle of the array surface in degrees.
        surface_azimuth (float): Azimuth angle of the array surface in degrees.
        lifespan (float): Lifespan of the array in years.
        pv_eol_derating (float): End-of-life derating factor.
        albedo (float): Albedo of the array surface.
        cell_temp_coeff (float): Temperature coefficient of the solar cells.
        refraction_index (float): Refraction index of the solar cells.
        e_poa_STC (float): POA Irradiance at Standard Test Conditions (STC).
        cell_temp_STC (float): Cell Temperature at STC.
        cost_per_kWp (float): Cost per kilowatt peak of the array.
        electrical_eff (float): Electrical efficiency of the array.
        area_m2 (float): Area of the array in square meters.
    """
    pv_panels: SolarPVPanel = None
    num_panels: int = None
    pv_kwp: float = 1
    surface_pitch: float = 35
    surface_azimuth: float = 0
    lifespan: float = 25
    pv_eol_derating: float = 0.88
    albedo: float = 0.2
    cell_temp_coeff: float = -0.004
    refraction_index: float = 0.05
    e_poa_STC: float = 1000
    cell_temp_STC: float = 25
    cost_per_kWp: float = 1250
    electrical_eff: float = 0.21
    area_m2: float = None

    def __post_init__(self):
        """
        Post-initialization method.
        Set's values if PVPanelList exists, and logs a message indicating the creation of the Solar PV Array.
        """
        if self.pv_panels is not None:
            self.pv_kwp = round(self.pv_panels.panel_kwp * self.num_panels, 3)
            self.area_m2 = round(self.pv_panels.size_m2 * self.num_panels, 3)

        logging.info(
            f"Solar PV array created: Size: {self.pv_kwp}kW, Size: {self.area_m2}m2,"
            f"Azimuth: {self.surface_azimuth}deg, Lifespan: {self.lifespan}yrs,"
            f"Pitch: {self.surface_pitch}deg, Efficiency: {self.electrical_eff}%"
        )


