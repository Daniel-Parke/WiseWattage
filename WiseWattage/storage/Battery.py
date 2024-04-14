from dataclasses import dataclass
from storage.battery_func import initialise_class

@dataclass
class Battery:
    max_capacity: float = 14.336
    discharge_depth: float = 0.2
    nominal_voltage: float = 51.2
    rated_capacity: float = 280
    max_discharge_C: float = 1
    max_charge_C: float = 0.5
    useable_capacity: float = 11.469
    max_discharge_kW: float = 14.336
    max_charge_kW: float = 7.168
    life_cycles:int = 5000
    cutoff_voltage: float = 48.8
    initial_charge:float = 0.5
    chemistry: str = "LiFePO4"
    exports_allowed:bool = False

    def __post_init__(self):
        initialise_class(self)
