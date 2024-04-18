from dataclasses import dataclass

from utility.inverter_func import initialise_class

@dataclass
class Inverter:
    inverter_eff: float = 0.95
    max_output: float = None
    standing_power_kW: float = None
    battery_charging: bool = True
    prioritise_charging: bool = True
    daytime_discharge: bool = True
    grid_charging: bool = False
    max_charge: float = None
    min_voltage_in: float = None
    max_voltage_in: float = None
    voltage_out: float = 240
    lifespan: float = 10
    cost_per_kw: float = 200
    cost: float = None
    load_inverter_name: str = None


    def __post_init__(self):
        initialise_class(self)
