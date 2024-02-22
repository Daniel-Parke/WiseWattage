from dataclasses import dataclass
import logging


@dataclass
class SolarPVPanel:
    """
    A class representing a Solar PV Panel.

    Attributes:
        panel_kwp (float): Rated power of the panel in kilowatts peak (kWp).
        size_m2 (float): Size of the array in square meters.
        eff (float): Electrical efficiency of the array.
        cell_temp_coeff (float): Temperature coefficient of the solar cells.
        cell_NOCT (float): Nominal Operating Cell Temperature (NOCT) of the solar cells.
        lifespan (float): Lifespan of the array in years.
        pv_eol_derating (float): End-of-life derating factor.
        I_sc (float): Short-circuit current of the solar cells.
        V_oc (float): Open-circuit voltage of the solar cells.
        I_mp (float): Maximum power current of the solar cells.
        V_mp (float): Maximum power voltage of the solar cells.
        material (str): Material of the solar cells.
        ambient_NOCT (float): Ambient NOCT of the solar cells.
        e_poa_NOCT (float): Plane of Array Irradiance (POA) NOCT.
        e_poa_STC (float): POA Irradiance at Standard Test Conditions (STC).
        cell_temp_STC (float): Cell Temperature at STC.
    """

    panel_kwp: float = 0.3538
    size_m2: float = 1.990
    eff: float = 0.2237
    cell_temp_coeff: float = -0.004
    cell_NOCT: float = 48
    lifespan: float = 25
    pv_eol_derating: float = 0.88
    I_sc: float = 11.21
    V_oc: float = 48.30
    I_mp: float = 10.70
    V_mp: float = 41.60
    material: str = "Mono-crystalline"
    bifacial: bool = False
    length_m: float = None
    width_m: float = None

    def __post_init__(self):
        """
        Post-initialization method.
        Logs a message indicating the creation of the Solar PV Panel.
        """
        logging.info(
            f"Solar PV Panel created: Size: {self.panel_kwp}kW, Size: {self.size_m2}m2,"
            f" Efficiency: {self.eff}%, Lifespan: {self.lifespan}yrs"
        )


