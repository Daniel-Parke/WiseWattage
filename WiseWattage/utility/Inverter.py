from dataclasses import dataclass

@dataclass
class Inverter:
    inverter_eff: float = 0.95
    max_output: float = None
    standing_power_kW: float = None
    battery_charging: bool = True
    prioritise_charging: bool = True
    daytime_discharge:bool = True
    grid_charging:bool = False
    max_charge: float = None
    min_voltage_in: float = None
    max_voltage_in: float = None
    voltage_out:float = 240


    def __post_init__(self):
        if self.max_output is not None:
            self.standing_power_kW = self.max_output * 0.003