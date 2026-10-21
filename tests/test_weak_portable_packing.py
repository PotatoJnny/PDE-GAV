"""Small QR reconstruction tests; no model assets, CUDA, or training are loaded."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from pde_agent_verifier_solver.plugins.vpfp.coherent_diffusion import VPFPCoherentMacroConfig
from pde_agent_verifier_solver.plugins.vpfp.coherent_velocity_diffusion import (
    VPFPVelocityMacroConfig, VPFPVelocityMacroTransform)
from pde_agent_verifier_solver.plugins.vpfp.hybrid_stress_diffusion import VPFPHybridStressJointMacroEDM
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
from pde_agent_verifier_solver.plugins.vpfp.weak_portable_base import (
    PACKING_QR_ATOL, PACKING_QR_GRAM_ATOL, _load_state, _state, _state_sha)


def _small_typed_model(degree=8):
    # Preserve the real, exact class/config/transform path checked by the loader
    # without constructing the large learned network: that is not under test.
    model = VPFPHybridStressJointMacroEDM.__new__(VPFPHybridStressJointMacroEDM)
    nn.Module.__init__(model)
    full = VPFPStructuredEDMConfig(control_count=33, grid_size=3, rho_max_mode=1,
        kinetic_max_mode=1, hermite_channels=1, width=8, depth=1)
    model.config = VPFPVelocityMacroConfig(
        VPFPCoherentMacroConfig(full, maximum_mode=1, coherent_mode=1), drive_degree=degree)
    model.transform = VPFPVelocityMacroTransform(model.config)
    model.register_buffer("unrelated_geometry", torch.tensor([1., 2.]))
    model.weight = nn.Parameter(torch.tensor([.125]))
    return model


def _saved(model):
    state = _state(model)
    return {"state": state, "state_sha256": _state_sha(state)}


def _rehash(saved):
    saved["state_sha256"] = _state_sha(saved["state"])
    return saved


@pytest.mark.parametrize("degree", [4, 8])
def test_exact_fixed_qr_records_geometry_and_restores_saved_bytes(degree):
    model = _small_typed_model(degree)
    saved = _saved(model)
    report, = _load_state(model, saved)
    assert report["constructor_bytes_equal"] is True
    assert report["max_absolute_difference"] == 0.
    assert report["stored_normalized_gram_max_error"] < PACKING_QR_GRAM_ATOL
    assert report["saved_bytes_restored"] is True
    assert _state_sha(model.state_dict()) == saved["state_sha256"]


@pytest.mark.parametrize("degree", [4, 8])
def test_rounding_sized_fixed_qr_drift_loads_original_checkpoint_bytes(degree):
    model = _small_typed_model(degree)
    saved = _saved(model)
    saved["state"]["transform.packing"][5, 1] += PACKING_QR_ATOL / 4
    saved["state"]["weight"] += 1  # learned weights come from the authenticated asset
    _rehash(saved)
    assert not torch.equal(model.transform.packing, saved["state"]["transform.packing"])
    report, = _load_state(model, saved)
    assert report["constructor_bytes_equal"] is False
    assert 0 < report["max_absolute_difference"] <= PACKING_QR_ATOL
    assert report["relative_tolerance"] == 0
    assert report["saved_bytes_restored"] is True
    assert torch.equal(model.transform.packing, saved["state"]["transform.packing"])
    assert _state_sha(model.state_dict()) == saved["state_sha256"]


@pytest.mark.parametrize("change", ["large_entry", "sign_flip", "column_swap"])
def test_substantive_packing_change_rejected(change):
    model = _small_typed_model()
    saved = _saved(model)
    packing = saved["state"]["transform.packing"]
    if change == "large_entry":
        packing[5, 1] += 1e-4
    elif change == "sign_flip":
        packing[:, 1] *= -1
    else:
        packing[:, [1, 2]] = packing[:, [2, 1]]
    with pytest.raises(ValueError, match="QR packing reconstruction differs"):
        _load_state(model, _rehash(saved))


def test_other_immutable_buffers_remain_bitwise_strict():
    model = _small_typed_model()
    saved = _saved(model)
    saved["state"]["unrelated_geometry"][0] += torch.finfo(torch.float32).eps
    with pytest.raises(ValueError, match="immutable buffer differs: unrelated_geometry"):
        _load_state(model, _rehash(saved))


def test_deterministic_load_does_not_relax_packing():
    model = _small_typed_model()
    saved = _saved(model)
    saved["state"]["transform.packing"][5, 1] += PACKING_QR_ATOL / 4
    with pytest.raises(ValueError, match="immutable buffer differs: transform.packing"):
        _load_state(model, _rehash(saved), deterministic=True)


def test_same_named_buffer_in_unrelated_class_remains_strict():
    model = nn.Module()
    model.transform = nn.Module()
    model.transform.register_buffer("packing", torch.ones(33, 10))
    saved = _saved(model)
    saved["state"]["transform.packing"][0, 0] += torch.finfo(torch.float32).eps
    with pytest.raises(ValueError, match="immutable buffer differs: transform.packing"):
        _load_state(model, _rehash(saved))


@pytest.mark.parametrize("corruption", ["hash", "shape", "dtype", "inventory", "nonfinite"])
def test_strict_saved_state_integrity_is_not_relaxed(corruption):
    model = _small_typed_model()
    saved = _saved(model)
    if corruption == "hash":
        saved["state"]["transform.packing"][5, 1] += PACKING_QR_ATOL / 4
    elif corruption == "shape":
        saved["state"]["transform.packing"] = saved["state"]["transform.packing"][:-1]
        _rehash(saved)
    elif corruption == "dtype":
        saved["state"]["transform.packing"] = saved["state"]["transform.packing"].double()
        _rehash(saved)
    elif corruption == "inventory":
        del saved["state"]["weight"]
        _rehash(saved)
    else:
        saved["state"]["transform.packing"][0, 0] = torch.nan
    with pytest.raises(ValueError):
        _load_state(model, saved)


def test_constructor_gram_geometry_checked_even_when_saved_bytes_match():
    model = _small_typed_model()
    model.transform.packing.zero_()
    with pytest.raises(ValueError, match="QR packing reconstruction differs"):
        _load_state(model, _saved(model))


def test_final_loaded_bytes_are_independently_checked(monkeypatch):
    model = _small_typed_model()
    saved = _saved(model)
    original_load = model.load_state_dict

    def corrupted_load(state, **kwargs):
        result = original_load(state, **kwargs)
        model.transform.packing[5, 1] += PACKING_QR_ATOL / 4
        return result

    monkeypatch.setattr(model, "load_state_dict", corrupted_load)
    with pytest.raises(ValueError, match="preserve all original tensor bytes"):
        _load_state(model, deepcopy(saved))
