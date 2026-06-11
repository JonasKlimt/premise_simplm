import builtins
import sys
import types

import numpy as np
import pytest

import premise.simpl_m as simpl_m_module


def _real_xarray():
    xr = pytest.importorskip("xarray")
    if getattr(xr.DataArray, "__module__", "") == "builtins":
        pytest.skip("real xarray is required for this test")
    return xr


def _iam_array(variables, years, values_by_variable, region="World"):
    xr = _real_xarray()
    data = np.array(
        [[values_by_variable[variable] for variable in variables]],
        dtype=float,
    )
    return xr.DataArray(
        data,
        dims=("region", "variables", "year"),
        coords={"region": [region], "variables": variables, "year": years},
    )


def _values_by_variable(data_array, region="World"):
    return {
        str(variable): float(data_array.sel(variables=variable, region=region).values)
        for variable in data_array.coords["variables"].values
    }


def test_simplm_loader_returns_parametrize_function_from_optional_package(monkeypatch):
    module = types.ModuleType("simplm_parametrization")

    def fake_parametrize_inventories(*args, **kwargs):
        return None

    module.parametrize_inventories = fake_parametrize_inventories
    monkeypatch.setitem(sys.modules, "simplm_parametrization", module)

    assert (
        simpl_m_module._load_simplm_parametrize_inventories()
        is fake_parametrize_inventories
    )


def test_simplm_loader_raises_clear_error_when_optional_package_missing(monkeypatch):
    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "simplm_parametrization", raising=False)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "simplm_parametrization":
            raise ModuleNotFoundError("No module named 'simplm_parametrization'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="use_simplm_parametrization=True"):
        simpl_m_module._load_simplm_parametrize_inventories()


def test_apply_simplm_parametrization_keeps_in_place_database(monkeypatch):
    database = []
    iam_payload = {"scenario": {"year": 2030}}
    captured = {}

    def fake_parametrize_inventories(**kwargs):
        captured.update(kwargs)
        kwargs["fg_db"].append({"name": "added in place"})
        return None

    monkeypatch.setattr(
        simpl_m_module,
        "_load_simplm_parametrize_inventories",
        lambda: fake_parametrize_inventories,
    )

    result = simpl_m_module.apply_simplm_parametrization(database, iam_payload)

    assert result is database
    assert database == [{"name": "added in place"}]
    assert captured == {"fg_db": database, "iam_data": iam_payload}


def test_apply_simplm_parametrization_accepts_returned_database(monkeypatch):
    database = [{"name": "original"}]
    returned_database = [{"name": "returned"}]
    iam_payload = {"scenario": {"year": 2030}}

    def fake_parametrize_inventories(**kwargs):
        return returned_database

    monkeypatch.setattr(
        simpl_m_module,
        "_load_simplm_parametrize_inventories",
        lambda: fake_parametrize_inventories,
    )

    result = simpl_m_module.apply_simplm_parametrization(database, iam_payload)

    assert result is returned_database


def test_apply_simplm_parametrization_rejects_invalid_return_type(monkeypatch):
    def fake_parametrize_inventories(**kwargs):
        return {"database": []}

    monkeypatch.setattr(
        simpl_m_module,
        "_load_simplm_parametrize_inventories",
        lambda: fake_parametrize_inventories,
    )

    with pytest.raises(TypeError, match="must return None or a list"):
        simpl_m_module.apply_simplm_parametrization(
            [], {"scenario": {"year": 2030}}
        )


def test_build_simplm_iam_payload(monkeypatch):
    class DummyGeo:
        iam_regions = ["World", "EUR"]

        def iam_to_ecoinvent_location(self, region):
            return {"World": ["GLO", "RoW"], "EUR": ["RER"]}[region]

    metals = types.SimpleNamespace(
        model="image",
        scenario="SSP2-Base",
        year=2030,
        geo=DummyGeo(),
        iam_data="raw iam data",
    )
    monkeypatch.setattr(
        simpl_m_module,
        "extract_iam_variables",
        lambda iam_data, year: ("iam variables", iam_data, year),
    )

    assert simpl_m_module.build_simplm_iam_payload(metals) == {
        "scenario": {"model": "image", "pathway": "SSP2-Base", "year": 2030},
        "iam_data": ("iam variables", "raw iam data", 2030),
        "region_mapping": {"World": ["GLO", "RoW"], "EUR": ["RER"]},
    }


def test_extract_iam_variables_interpolates_target_year():
    final_energy_vars = [
        "Industry - Other - Elec",
        "Industry - Other - H2",
        "Industry - Other - Solid Biomass",
        "Industry - Other - Solid Coal",
    ]
    iam_data = types.SimpleNamespace(
        steel_technology_efficiencies=_iam_array(
            ["steel - primary - BF/BOF"],
            [2020, 2030],
            {"steel - primary - BF/BOF": [1.0, 1.2]},
        ),
        electricity_mix=_iam_array(
            ["Wind Onshore", "Wind Offshore", "Solar CSP"],
            [2020, 2030],
            {
                "Wind Onshore": [0.1, 0.2],
                "Wind Offshore": [0.1, 0.1],
                "Solar CSP": [0.0, 0.1],
            },
        ),
        road_freight_fleet=_iam_array(
            [
                "truck, battery electric, 40 metric ton",
                "truck, diesel, 40 metric ton",
            ],
            [2020, 2030],
            {
                "truck, battery electric, 40 metric ton": [0.0, 40.0],
                "truck, diesel, 40 metric ton": [100.0, 60.0],
            },
        ),
        final_energy_use=_iam_array(
            final_energy_vars,
            [2020, 2030],
            {
                "Industry - Other - Elec": [20.0, 40.0],
                "Industry - Other - H2": [1.0, 9.0],
                "Industry - Other - Solid Biomass": [10.0, 20.0],
                "Industry - Other - Solid Coal": [69.0, 31.0],
            },
        ),
    )

    values = _values_by_variable(simpl_m_module.extract_iam_variables(iam_data, 2025))

    assert values == {
        "Steel BF/BOF efficiency": pytest.approx(1 - (1 / 1.1)),
        "Wind and PV renewable shares in electricity mix": pytest.approx(0.3),
        "Electric truck share (highest weight class)": pytest.approx(0.2),
        "Electricity share other industry": pytest.approx(0.1),
        "Hydrogen share other industry": pytest.approx(0.04),
        "Solid biomass share other industry": pytest.approx(
            (15 / (15 + 50)) - (10 / (10 + 69))
        ),
    }


def test_extract_iam_variables_uses_zero_for_missing_optional_variables():
    iam_data = types.SimpleNamespace(
        steel_technology_efficiencies=_iam_array(
            ["other steel"],
            [2020, 2030],
            {"other steel": [1.0, 1.0]},
        ),
        electricity_mix=_iam_array(
            ["Hydro"],
            [2020, 2030],
            {"Hydro": [0.5, 0.5]},
        ),
        road_freight_fleet=_iam_array(
            ["truck, diesel, 20 metric ton"],
            [2020, 2030],
            {"truck, diesel, 20 metric ton": [1.0, 1.0]},
        ),
        final_energy_use=_iam_array(
            ["Industry - Buildings - Elec"],
            [2020, 2030],
            {"Industry - Buildings - Elec": [1.0, 1.0]},
        ),
    )

    values = _values_by_variable(simpl_m_module.extract_iam_variables(iam_data, 2025))

    assert values == {
        "Steel BF/BOF efficiency": 0.0,
        "Wind and PV renewable shares in electricity mix": 0.0,
        "Electric truck share (highest weight class)": 0.0,
        "Electricity share other industry": 0.0,
        "Hydrogen share other industry": 0.0,
        "Solid biomass share other industry": 0.0,
    }


def test_apply_iam_driven_inventory_adjustments_reindexes_adapter_result(monkeypatch):
    metals = types.SimpleNamespace(
        database=[{"name": "original"}],
        combined_storage=None,
    )
    iam_payload = {"scenario": {"year": 2030}}

    returned_database = [{"name": "returned"}]
    captured = {}
    calls = []

    def fake_apply_simplm_parametrization(database, iam_payload):
        calls.append("apply_simplm_parametrization")
        captured["database"] = database
        captured["iam_payload"] = iam_payload
        return returned_database

    def fake_build_db_indexes():
        calls.append("build_db_indexes")
        assert metals.database is returned_database

    monkeypatch.setattr(
        simpl_m_module,
        "build_simplm_iam_payload",
        lambda metals: iam_payload,
    )
    monkeypatch.setattr(
        simpl_m_module,
        "apply_simplm_parametrization",
        fake_apply_simplm_parametrization,
    )
    metals.build_db_indexes = fake_build_db_indexes

    simpl_m_module.apply_iam_driven_inventory_adjustments(metals)

    assert metals.database is returned_database
    assert calls == ["apply_simplm_parametrization", "build_db_indexes"]
    assert captured["database"] == [{"name": "original"}]
    assert captured["iam_payload"] is iam_payload
    assert metals.combined_storage is iam_payload
