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
        e_poa_STC (float): POA Irradiance at Standard Test Conditions (STC).
        cell_temp_STC (float): Cell Temperature at STC.
        cost_per_kWp (float): Cost per kilowatt peak of the array.
        area_m2 (float): Area of the array in square meters.
    """
    pv_panel: SolarPVPanel = None
    num_panels: int = None
    surface_pitch: float = 35
    surface_azimuth: float = 0
    albedo: float = 0.2
    cost_per_kWp: float = 1250
    pv_kwp: float = 1
    area_m2: float = None

    def __post_init__(self):
        """
        Post-initialization method.
        Set's values if PVPanelList exists, and logs a message indicating the creation of the Solar PV Array.
        """
        if self.pv_panel is not None:
            self.pv_kwp = round(self.pv_panel.panel_kwp * self.num_panels, 3)
            self.area_m2 = round(self.pv_panel.size_m2 * self.num_panels, 3)

        logging.info(
            f"Solar PV array created: Size: {self.pv_kwp}kW, Size: {self.area_m2}m2, "
            f"Azimuth: {self.surface_azimuth}deg, Lifespan: {self.pv_panel.lifespan}yrs, "
            f"Pitch: {self.surface_pitch}deg, Efficiency: {self.pv_panel.eff*100}%"
        )


