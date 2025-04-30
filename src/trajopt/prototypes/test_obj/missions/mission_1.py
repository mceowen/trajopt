

import Mission


def model_aerodynamics_impl(a: float, b: float, c: float):
    # This is a placeholder for the actual implementation
    return a+b+c

mission = Mission(
    {"some": "config"}
)
mission.model_aerodynamics = model_aerodynamics_impl