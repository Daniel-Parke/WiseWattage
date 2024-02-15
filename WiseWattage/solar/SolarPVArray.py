from dataclasses import dataclass
import logging


@dataclass
class SolarPVArray:
    """
    A class representing a Solar PV Array.

    Attributes:
        pv_kwp (float): Rated power of the array in kilowatts peak (kWp).
        surface_pitch (float): Tilt angle of the array surface in degrees.
        surface_azimuth (float): Azimuth angle of the array surface in degrees.
        lifespan (float): Lifespan of the array in years.
        pv_eol_derating (float): End-of-life derating factor.
        cost_per_kWp (float): Cost per kilowatt peak of the array.
        electrical_eff (float): Electrical efficiency of the array.
        cell_temp_coeff (float): Temperature coefficient of the solar cells.
        transmittance_absorptance (float): Transmittance absorptance factor.
        refraction_index (float): Refraction index of the solar cells.
        cell_NOCT (float): Nominal Operating Cell Temperature (NOCT) of the solar cells.
        ambient_NOCT (float): Ambient NOCT of the solar cells.
        e_poa_NOCT (float): Plane of Array Irradiance (POA) NOCT.
        e_poa_STC (float): POA Irradiance at Standard Test Conditions (STC).
        cell_temp_STC (float): Cell Temperature at STC.
    """

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

    def __post_init__(self):
        """
        Post-initialization method.
        Logs a message indicating the creation of the Solar PV Array.
        """
        logging.info(
            f"Solar PV array created: Size: {self.pv_kwp}kW, Pitch: {self.surface_pitch}deg,"
            f" Azimuth: {self.surface_azimuth}deg, Lifespan: {self.lifespan}yrs"
        )
        logging.info("*******************")
