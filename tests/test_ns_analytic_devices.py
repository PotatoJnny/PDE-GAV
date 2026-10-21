"""Resolve analytic query placement from resident coefficients, not CUDA aliases."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy/ns/src"))
from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
from agent_verifier.neural_generator.structured_ns_answer_first import (
    ContinuousStructuredNSMMS, StructuredNSCaseCoefficients, StructuredNSProjectionConfig,
)
from agent_verifier.neural_generator.structured_ns_analytic import (
    analytic_forcing, analytic_quantities, analytic_velocity, analytic_velocity_pressure,
)


_POINTS = np.array([[0., .2, .31], [.02, .7, .31], [.2, .8, .31],
                    [.45, .4, .31], [.7, .2, .31], [.99, .9, .31]], dtype=np.float64)


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _field(device):
    rng = torch.Generator().manual_seed(217)
    coefficients = StructuredNSCaseCoefficients(
        psi=torch.randn((4, 12), generator=rng, dtype=torch.float64) * 1e-4,
        pressure=torch.randn((4, 12), generator=rng, dtype=torch.float64) * 3e-4,
    )
    return ContinuousStructuredNSMMS(
        coefficients,
        config=StructuredNSProjectionConfig(spatial_mode_x=3, spatial_mode_y=2,
                                            time_order=3, geometry_mask_transition=.05),
        geometry_spec=ContinuousGeometrySpec(
            domain_bounds=(0., 1., 0., 1.),
            obstacle_polygons=(((.3, .3), (.6, .3), (.45, .6)),),
        ),
        domain_bounds=(0., 1., 0., 1.), viscosity=.037, t0=.2, t1=.57, device=device,
    )


def _reference():
    # The original autograd field is created independently on the CPU, before
    # any metadata alias mutation, and never passes through the new adapter.
    field = _field("cpu")
    with torch.inference_mode(False), torch.enable_grad():
        return {name: value.detach().clone() for name, value in
                field.quantities(_POINTS, include_laplacian_components=True).items()}


def _assert_all_outputs(field, expected):
    target = field.coefficients.psi.device
    for points in (_POINTS, torch.from_numpy(_POINTS)):
        for hint in (None, True, False):
            quantities = analytic_quantities(field, points, uniform_time=hint,
                                              include_laplacian_components=True)
            assert quantities.keys() == expected.keys()
            groups = [(tuple(quantities), tuple(quantities.values()))]
            for function, names in (
                (analytic_forcing, ("forcing_u", "forcing_v")),
                (analytic_velocity, ("u", "v")),
                (analytic_velocity_pressure, ("u", "v", "pressure")),
            ):
                groups.append((names, function(field, points, uniform_time=hint)))
            for names, outputs in groups:
                for name, output in zip(names, outputs, strict=True):
                    assert output.device == target, name
                    torch.testing.assert_close(output.detach().cpu(), expected[name],
                                               rtol=2e-10, atol=1e-9, msg=name)


@pytest.mark.parametrize("alias", ("cpu:0", "cuda"))
def test_cpu_resident_coefficients_override_field_device_alias(alias):
    expected = _reference()
    field = _field("cpu")
    # This deliberately stale metadata is a CPU-only regression for trusting a
    # requested device label instead of the field's resident numerical tensors.
    field.device = torch.device(alias)
    _assert_all_outputs(field, expected)
    assert field.coefficients.psi.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an allocated GPU")
def test_single_gpu_bare_cuda_alias_uses_actual_coefficient_device():
    expected = _reference()
    with torch.cuda.device(0):
        field = _field("cuda")
        assert field.device.index is None
        assert field.coefficients.psi.device == torch.device("cuda:0")
        _assert_all_outputs(field, expected)
        assert torch.cuda.current_device() == 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two allocated GPUs")
@pytest.mark.parametrize("field_index, query_index", ((0, 1), (1, 0)))
def test_other_thread_current_gpu_does_not_relocate_field_queries(field_index, query_index):
    expected = _reference()
    with torch.cuda.device(field_index):
        field = _field("cuda")
    assert field.device.index is None
    assert field.coefficients.psi.device == torch.device("cuda", field_index)

    def query_on_another_device():
        torch.cuda.set_device(query_index)
        _assert_all_outputs(field, expected)
        # The analytic API targets existing tensors without changing the
        # caller's CUDA device, so unrelated work keeps its original placement.
        assert torch.cuda.current_device() == query_index

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(query_on_another_device).result(timeout=30)
