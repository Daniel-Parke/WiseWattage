from dataclasses import dataclass

@dataclass
class Grid:
    import_standard: float = 0.2982
    export_standard: float = 0.1422
    emissions_kgCO2: float = 0.322
    day_night_tariff: bool = False
    import_day: float = 0.3378
    import_night: float = 0.1804
    export_day: float = 0.1422
    export_night: float = 0.1422
    night_start: int = 1
    night_end: int = 8
    standing_charge: float = 0.0958
    connection_fee: float = 0    # No fee if system already has grid connection 
    tariff_name: str = "Power_NI"
    import_allow: bool = True
    export_allow: bool = True