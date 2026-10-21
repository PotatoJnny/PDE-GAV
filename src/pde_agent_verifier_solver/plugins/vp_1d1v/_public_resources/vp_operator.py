"""Public pointwise differential convention for forced 1D1V VP."""


def kinetic_source(
    *,
    distribution,
    d_t_distribution,
    d_x_distribution,
    d_v_distribution,
    velocity,
    electric_field,
):
    """Apply d_t f + v d_x f + E d_v f pointwise."""

    del distribution
    return (
        d_t_distribution
        + velocity * d_x_distribution
        + electric_field * d_v_distribution
    )


def poisson_residual(
    *,
    d_x_electric_field,
    density,
    neutralizing_background,
):
    """Apply d_x E - (M0 - n_background) pointwise."""

    return d_x_electric_field - (density - neutralizing_background)
