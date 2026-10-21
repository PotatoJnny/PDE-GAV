#!/usr/bin/env python3
"""CPU sensitivity replay of frozen Poisson solvers; no Agent calls or training.

The original evaluation is preserved. This is an author-side, in-process
functional diagnostic, not a new budget-matched Agent experiment or speed test.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from scipy.interpolate import make_interp_spline
from pde_agent_verifier_solver.sdk import AnswerQuery, Case, CaseSource, CaseVisibility
from pde_agent_verifier_solver.sdk.public_sessions import bind_public_session_factory
from pde_agent_verifier_solver.plugins.poisson_diagnostic import PoissonField, public_data

CENTERS = np.linspace(0., 1., 16)


@lru_cache(maxsize=32)
def scale(length, expected_rms):
    x = np.linspace(0., 1., max(8193, int(np.ceil(16/length))+1))
    k = np.exp(-.5*((x[:, None]-CENTERS)/length)**2)
    variance = (x*(1-x))**2*np.mean(k*k, axis=1)
    return expected_rms/np.sqrt(2*np.trapezoid(variance, x))


@dataclass(frozen=True)
class ProbeField:
    amplitude: float
    coefficients: tuple[float, ...]
    length: float
    expected_rms: float

    def noise(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        d = x[:, None]-CENTERS
        k = np.exp(-.5*(d/self.length)**2)/np.sqrt(len(CENTERS))
        z = np.asarray(self.coefficients)
        g = k@z
        gp = (-d/self.length**2*k)@z
        gpp = ((d*d/self.length**4-1/self.length**2)*k)@z
        w = x*(1-x)
        s = scale(self.length, self.expected_rms)*abs(self.amplitude)
        return s*w*g, s*(-2*g+2*(1-2*x)*gp+w*gpp)

    def solution(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        v, _ = self.noise(x)
        return self.amplitude*np.sin(np.pi*x)+v

    def forcing(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        _, vpp = self.noise(x)
        return self.amplitude*np.pi**2*np.sin(np.pi*x)-vpp


class ForcingOracle:
    def __init__(self, field):
        self.field = field
        self.calls = 0
        self.points = 0

    def query_forcing(self, points):
        x = np.asarray(points, dtype=float)
        if (x.ndim != 2 or x.shape[1] != 1 or len(x) > 4096 or len(x) < 1
                or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1)):
            raise ValueError('invalid public forcing request')
        self.calls += 1; self.points += len(x)
        return self.field.forcing(x[:, 0])[:, None]


def rel(pred, truth):
    return float(np.linalg.norm(np.asarray(pred)-truth)/np.linalg.norm(truth))


def frozen_program(run, stage):
    info = json.loads((run/stage/'result.json').read_text())
    manifest = Path(info['candidate'])
    entry = manifest.parent/json.loads(manifest.read_text())['entrypoint']
    spec = importlib.util.spec_from_file_location('frozen_poisson_'+stage, entry)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, str(entry)


def replay(program, field, x, dense):
    case = Case('cpu-sensitivity-case', CaseSource.GENERATED, CaseVisibility.HIDDEN, public_data())
    oracle = ForcingOracle(field)
    started = time.perf_counter()
    with bind_public_session_factory(lambda case: oracle, case, allowed_methods=('query_forcing',)):
        handle = program.prepare_solution(case, solve_budget=10.)
        # Reveal diagnostic coordinates only after preparation, as in the formal ABI.
        query = AnswerQuery.at_points(fields=('u',), points=tuple({'x': float(v)} for v in x))
        qdense = AnswerQuery.at_points(fields=('u',), points=tuple({'x': float(v)} for v in dense))
        prediction = np.asarray(handle.evaluate(query).fields['u'])
        pdense = np.asarray(handle.evaluate(qdense).fields['u'])
    return {'original_512_error': rel(prediction, field.solution(x)),
            'dense_4095_error': rel(pdense, field.solution(dense)),
            'public_query_calls': oracle.calls, 'public_query_points': oracle.points,
            'diagnostic_wall_seconds': time.perf_counter()-started}


class FineControl:
    """Author numerical control; generic uniform sampling and Green integration."""
    @staticmethod
    def prepare_solution(case, solve_budget=None):
        from pde_agent_verifier_solver.sdk import open_public_case_session, AnswerView, QueryKind
        oracle = open_public_case_session(case)
        x = np.linspace(0., 1., 32769)
        f = np.concatenate([np.asarray(oracle.query_forcing(part[:, None])).reshape(-1)
                            for part in np.array_split(x, 9)])
        a = make_interp_spline(x, f, k=5).antiderivative(2)
        a0, delta = float(a(0.)), float(a(1.))-float(a(0.))
        class Prepared:
            def evaluate(self, query):
                z = np.asarray([p['x'] for p in query.points])
                return AnswerView(QueryKind.POINTS, {'u': tuple(-a(z)+a0+delta*z)},
                                  points=query.points, shape=(len(z),))
        return Prepared()


def check_mms(field):
    h = min(1e-4, field.length/100)
    x = (CENTERS[:, None]+np.asarray([-.8,0.,.8])[None,:]*field.length).reshape(-1)
    x = x[(x > 2*h)&(x < 1-2*h)]
    u = field.solution
    second = (-u(x+2*h)+16*u(x+h)-30*u(x)+16*u(x-h)-u(x-2*h))/(12*h*h)
    residual = rel(-second, field.forcing(x))
    boundary = float(np.max(np.abs(u([0., 1.]))))
    if not (residual < 1e-5 and boundary < 1e-14):
        raise AssertionError(f'MMS qualification failed: {residual=}, {boundary=}')
    return residual


def run_probe(run, output, lengths, strengths):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    bank = json.loads((run/'data/terminal.json').read_text())
    original = json.loads((run/'results.json').read_text())['results']
    children = [r for r in bank['cases'] if r['gaussian_coefficients']]
    methods = {}; paths = {}
    for stage in ('common','without_p'):
        methods[stage], paths[stage] = frozen_program(run, stage)
    _, with_path = frozen_program(run, 'with_p')
    assert Path(with_path).read_bytes() == Path(paths['common']).read_bytes()
    methods['author_fine_32769'] = FineControl()
    dense = np.arange(1, 4096)/4096
    quadrature_x = np.linspace(0., 1., 16385)
    rows = []
    for length in lengths:
        for strength in strengths:
            for index, item in enumerate(children):
                field = ProbeField(item['amplitude'], tuple(item['gaussian_coefficients']), length, strength)
                x = np.asarray(item['test_x'])
                reference = field.solution(x)
                base = item['amplitude']*np.sin(np.pi*x)
                base_f = np.pi**2*base
                origin = length == .2 and strength == .1
                if origin:
                    old = PoissonField(item['amplitude'], tuple(item['gaussian_coefficients']))
                    np.testing.assert_array_equal(field.solution(x), old.solution(x))
                    np.testing.assert_array_equal(field.forcing(x), old.forcing(x))
                mms = check_mms(field)
                # Author-written restricted controls; never supplied to an Agent.
                shortcut = rel(field.forcing(x)/np.pi**2, reference)
                first_coefficient = 2/np.pi**2*np.trapezoid(
                    field.forcing(quadrature_x)*np.sin(np.pi*quadrature_x), quadrature_x)
                projected = rel(first_coefficient*np.sin(np.pi*x), reference)
                row = {'parent_index':index, 'case_id':item['case']['case_id'],
                       'length_scale':length, 'expected_relative_rms':strength,
                       'actual_solution_change':rel(reference,base),
                       'actual_forcing_change':rel(field.forcing(x),base_f),
                       'mms_residual':mms, 'author_scalar_shortcut_error':shortcut,
                       'author_one_mode_error':projected, 'methods':{}}
                for name, method in methods.items():
                    metrics = replay(method, field, x, dense)
                    if origin and name in original:
                        old_row = next(r for r in original[name]['cases'] if r['case_id']==item['case']['case_id'])
                        np.testing.assert_allclose(metrics['original_512_error'], old_row['error'], rtol=1e-8, atol=2e-14)
                    row['methods'][name] = metrics
                rows.append(row)
            group = rows[-len(children):]
            print(json.dumps({'length':length,'strength':strength,'mean_error':{
                name:float(np.mean([r['methods'][name]['original_512_error'] for r in group]))
                for name in methods}}), flush=True)
    summaries=[]
    for length in lengths:
        for strength in strengths:
            group=[r for r in rows if r['length_scale']==length and r['expected_relative_rms']==strength]
            summary={'length_scale':length,'expected_relative_rms':strength}
            for name in ('actual_solution_change','actual_forcing_change','author_scalar_shortcut_error','author_one_mode_error'):
                v=[r[name] for r in group]
                summary[name]={'mean':float(np.mean(v)),'min':min(v),'max':max(v)}
            summary['methods']={name:{metric:float(np.mean([r['methods'][name][metric] for r in group]))
                for metric in ('original_512_error','dense_4095_error','public_query_points')}
                for name in methods}
            summaries.append(summary)
    report={'schema':'poisson_perturbation_sensitivity_v1','status':'completed','cpu_only':True,
            'agent_calls':0,'trained_generators':0,'original_run':str(run),'frozen_solver_paths':paths,
            'with_p_program_identical_to_common':True,'original_perturbation_reproduced':True,
            'scope':'Post-hoc functional sensitivity, not new Agent results or official speed measurements.',
            'fixed_directions':'All 8 archived test Gaussian coefficient draws; no rejection or direction selection.',
            'length_scales':lengths,'expected_relative_rms_values':strengths,'centers':CENTERS.tolist(),
            'numerical_checks_passed':True,'max_mms_residual':max(r['mms_residual'] for r in rows),
            'summaries':summaries,'cases':rows,'wall_seconds':time.perf_counter()-started}
    (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'status':'completed','wall_seconds':report['wall_seconds']}))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'.paper_runs/poisson_perturbed_gaussian_cpu_20260923')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--length-scales',type=float,nargs='+',default=[.2,.05,.02,.01,.005,.002])
    p.add_argument('--strengths',type=float,nargs='+',default=[.1,.3,1.])
    args=p.parse_args()
    if any(v<=0 or not np.isfinite(v) for v in args.length_scales+args.strengths):
        p.error('lengths and strengths must be finite and positive')
    run_probe(args.run.resolve(),args.output.resolve(),args.length_scales,args.strengths)

if __name__=='__main__':
    main()
