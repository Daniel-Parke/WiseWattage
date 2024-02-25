from dataclasses import dataclass

@dataclass
class Grid:
    import_allow: bool = True
    export_allow: bool = True
    import_day: float = 0.2974
    import_night: float = 0.1742
    export_day: float = 0.1422
    export_night: float = 0.1422
    night_start: int = 1
    night_end: int = 8
    