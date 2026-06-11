"""
Integration helpers for SIMPLM mineral inventory parametrization.
"""

from .logger import create_logger

logger = create_logger("simpl_m")


def _get_xarray():
    import xarray as xr

    return xr


def _load_simplm_parametrize_inventories():
    try:
        from simplm_parametrization import parametrize_inventories
    except ImportError as exc:
        raise ImportError(
            "SIMPLM mineral inventory parametrization requires the optional "
            "'simplm_parametrization' package. Install it before running metals "
            "updates with use_simplm_parametrization=True, or leave that option "
            "disabled."
        ) from exc

    return parametrize_inventories


def apply_simplm_parametrization(database, iam_payload):
    """
    Apply SIMPLM mineral inventory parametrization and return the updated database.

    SIMPLM currently mutates Wurst databases in place, but accepting an explicit
    returned list keeps the integration stable if SIMPLM changes that contract.
    """

    parametrize_inventories = _load_simplm_parametrize_inventories()
    result = parametrize_inventories(fg_db=database, iam_data=iam_payload)

    if result is None:
        return database

    if isinstance(result, list):
        return result

    raise TypeError(
        "simplm_parametrization.parametrize_inventories must return None or a "
        f"list of database activities, not {type(result).__name__}."
    )


def build_simplm_iam_payload(metals):
    """
    Build the IAM payload consumed by SIMPLM mineral inventory parametrization.
    """

    return {
        "scenario": {
            "model": metals.model,
            "pathway": metals.scenario,
            "year": int(metals.year),
        },
        "iam_data": extract_iam_variables(metals.iam_data, metals.year),
        "region_mapping": {
            region: metals.geo.iam_to_ecoinvent_location(region)
            for region in metals.geo.iam_regions
        },
        # a variable containing ore grade extraction curves for each metal should be added here
    }


def apply_iam_driven_inventory_adjustments(metals):
    """
    Apply IAM-driven SIMPLM mineral inventory parametrization to a Metals object.
    """

    metals.combined_storage = build_simplm_iam_payload(metals)
    metals.database = apply_simplm_parametrization(
        metals.database, metals.combined_storage
    )
    metals.build_db_indexes()


def _select_vars(values, include=(), exclude=()):
    return [
        v
        for v in values
        if all(token in str(v) for token in include)
        and all(token not in str(v) for token in exclude)
    ]


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _available_vars(data, variables):
    requested = _as_list(variables)
    if "variables" not in data.coords:
        return []

    available = {str(v) for v in data.coords["variables"].values}
    return [v for v in requested if str(v) in available]


def _select_year(data, year):
    if "year" not in data.coords:
        return data

    years = data.coords["year"].values.tolist()
    if year in years:
        return data.sel(year=year)

    return data.interp(year=year)


def _zero_like_year(data, year):
    selected = _select_year(data, year)
    if "variables" in selected.dims:
        return selected.sum("variables") * 0
    return selected * 0


def _select_var_or_zero(data, year, variable, label=None):
    if variable not in _available_vars(data, [variable]):
        logger.warning(
            "IAM variable '%s' missing for %s; using zero.",
            variable,
            label or variable,
        )
        return _zero_like_year(data, year)

    return _select_year(data, year).sel(variables=variable).squeeze(drop=True)


def _sum_vars_or_zero(data, year, variables, label):
    available_vars = _available_vars(data, variables)
    if not available_vars:
        logger.warning(
            "No IAM variables available for %s; using zero.",
            label,
        )
        return _zero_like_year(data, year)

    return _select_year(data, year).sel(variables=available_vars).sum("variables")


def _share(data, year, num_var, den_vars, label=None):
    xr = _get_xarray()
    available_den_vars = _available_vars(data, den_vars)
    if not available_den_vars:
        logger.warning(
            "No denominator IAM variables available for %s; using zero.",
            label or num_var,
        )
        return _zero_like_year(data, year)

    if num_var not in _available_vars(data, [num_var]):
        logger.warning(
            "IAM variable '%s' missing for %s; using zero.",
            num_var,
            label or num_var,
        )
        return _zero_like_year(data, year)

    selected = _select_year(data, year)
    denominator = selected.sel(variables=available_den_vars).sum("variables")
    numerator = selected.sel(variables=num_var)
    return xr.where(denominator > 0, numerator / denominator, 0.0)


def _share_change(data, target_year, base_year, num_var, den_vars, label=None):
    xr = _get_xarray()
    target_share = _share(data, target_year, num_var, den_vars, label=label)
    base_share = _share(data, base_year, num_var, den_vars, label=label)
    return xr.where(base_share > 0, target_share - base_share, 0.0)


def extract_iam_variables(iam_data, year):
    """
    Extract IAM indicators used as proxies to find parameter assumptions for
    metal inventory improvements.
    """

    xr = _get_xarray()
    target_year = year
    base_year = 2020

    # Steel BOF/BF efficiencies
    steel_raw = _select_var_or_zero(
        iam_data.steel_technology_efficiencies,
        target_year,
        "steel - primary - BF/BOF",
        label="Steel BF/BOF efficiency",
    )
    steel = xr.where(steel_raw > 0, 1 - (1 / steel_raw), 0.0).assign_coords(
        variables="Steel BF/BOF efficiency"
    )

    # Wind and solar shares in electricity mix
    renewables = _sum_vars_or_zero(
        iam_data.electricity_mix,
        target_year,
        ["Wind Onshore", "Wind Offshore", "Solar CSP"],
        label="Wind and PV renewable shares in electricity mix",
    ).assign_coords(
        variables="Wind and PV renewable shares in electricity mix"
    )

    # Electric truck share in road freight, for the highest weight class available (currently 40t)
    fleet = iam_data.road_freight_fleet
    vars_40t = _select_vars(
        fleet.coords["variables"].values,
        include=["40 metric ton"],
    )
    fleet_share = _share(
        fleet,
        target_year,
        "truck, battery electric, 40 metric ton",
        vars_40t,
        label="Electric truck share (highest weight class)",
    ).assign_coords(variables="Electric truck share (highest weight class)")

    # Changes in electricity and hydrogen shares of final energy use in other industry sectors
    final_energy = iam_data.final_energy_use
    other_industry_vars = _select_vars(
        final_energy.coords["variables"].values,
        include=["Industry - Other"],
        exclude=[","],
    )

    elec_change = _share_change(
        final_energy,
        target_year,
        base_year,
        "Industry - Other - Elec",
        other_industry_vars,
        label="Electricity share other industry",
    ).assign_coords(variables="Electricity share other industry")

    h2_change = _share_change(
        final_energy,
        target_year,
        base_year,
        "Industry - Other - H2",
        other_industry_vars,
        label="Hydrogen share other industry",
    ).assign_coords(variables="Hydrogen share other industry")

    # Changes in solid biomass share of solid final energy use in other industry sectors
    solid_vars = _select_vars(
        final_energy.coords["variables"].values,
        include=["Industry - Other - Solid"],
    )

    bio_target = _share(
        final_energy,
        target_year,
        "Industry - Other - Solid Biomass",
        solid_vars,
        label="Solid biomass share other industry",
    )
    bio_base = _share(
        final_energy,
        base_year,
        "Industry - Other - Solid Biomass",
        solid_vars,
        label="Solid biomass share other industry",
    )
    biomass_change = xr.where(
        bio_target > bio_base, bio_target - bio_base, 0.0
    ).assign_coords(variables="Solid biomass share other industry")

    arrays = [steel, renewables, fleet_share, elec_change, h2_change, biomass_change]

    return xr.concat(
        [arr.drop_vars("year", errors="ignore") for arr in arrays],
        dim="variables",
    )
