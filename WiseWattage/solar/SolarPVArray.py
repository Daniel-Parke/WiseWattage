from dataclasses import dataclass

from solar.SolarPVPanel import SolarPVPanel
from solar.solar_pv_array import initialise_array


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
    num_panels: int = 1
    surface_pitch: float = 35
    surface_azimuth: float = 0
    albedo: float = 0.2
    
    pv_kwp: float = 1
    area_m2: float = None

    def __post_init__(self):
        """
        Post-initialization method.
        Set's values if PVPanelList exists, and logs a message indicating the creation of the Solar PV Array.
        """
        initialise_array(self)


