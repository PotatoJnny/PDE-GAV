"""Public differential convention for the NS manufactured questions."""

from __future__ import annotations


def forcing_from_derivatives(
    *,
    velocity_x,
    velocity_y,
    d_t_velocity_x,
    d_t_velocity_y,
    d_x_velocity_x,
    d_y_velocity_x,
    d_x_velocity_y,
    d_y_velocity_y,
    laplacian_velocity_x,
    laplacian_velocity_y,
    d_x_pressure,
    d_y_pressure,
    viscosity,
):
    """Apply the declared operator pointwise to compatible array-like values."""

    forcing_x = (
        d_t_velocity_x
        + velocity_x * d_x_velocity_x
        + velocity_y * d_y_velocity_x
        + d_x_pressure
        - viscosity * laplacian_velocity_x
    )
    forcing_y = (
        d_t_velocity_y
        + velocity_x * d_x_velocity_y
        + velocity_y * d_y_velocity_y
        + d_y_pressure
        - viscosity * laplacian_velocity_y
    )
    return forcing_x, forcing_y


def divergence(*, d_x_velocity_x, d_y_velocity_y):
    return d_x_velocity_x + d_y_velocity_y
