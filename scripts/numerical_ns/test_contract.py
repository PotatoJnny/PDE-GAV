"""Small transport and frozen-cohort checks; no private truth enters a backend."""
import json,os,sys,tempfile,shutil,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from qualify import Analytic
import submission
from pde_agent_verifier_solver.sdk import AnswerQuery
ROOT=Path(__file__).resolve().parents[2]/'.paper_runs/ns_numerical_baselines_20260924'

class ContractTests(unittest.TestCase):
    def test_public_rpc_finishes_before_hidden_points(self):
        analytic=Analytic(True);runtime=json.loads((ROOT/'control/runtime.json').read_text())
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            d=Path(temporary)
            for name in ['backend.py','wire.py','mac.py']:shutil.copy2(Path(__file__).with_name(name),d/name)
            (d/'profile.json').write_text(json.dumps({**runtime,'method':'mac','resolution':8,'steps':20}))
            old_cwd=Path.cwd();os.chdir(d)
            calls=[]
            def query(op,points):calls.append(op);return analytic.query(op,np.asarray(points))
            session=SimpleNamespace(query_initial_velocity=lambda p:query('I',p),query_forcing=lambda p:query('F',p),query_boundary_velocity=lambda p:query('B',p))
            old_file,old_open=submission.__file__,submission.open_public_case_session
            submission.__file__=str(d/'submission.py');submission.open_public_case_session=lambda _:session
            try:
                handle=submission.prepare_solution(SimpleNamespace(public_data=analytic.metadata))
                self.assertTrue(calls.count('F')>=40)
                # Hidden positions do not exist in this process until prepare ends.
                rng=np.random.default_rng(2026)
                points=[{'t':.081,'x':float(x),'y':float(y)} for x,y in rng.uniform(.02,.98,(20,2))]
                before=len(calls);result=handle.evaluate(AnswerQuery.at_points(fields=['velocity_x','velocity_y'],points=points))
                self.assertEqual(len(calls),before)
                self.assertEqual(result.points,tuple(points));self.assertIsNotNone(handle.process.poll())
                truth=analytic.values(*np.array([[p['t'],p['x'],p['y']] for p in points]).T)[0]
                predicted=np.column_stack([result.fields['velocity_x'],result.fields['velocity_y']])
                self.assertLess(np.linalg.norm(predicted-truth)/np.linalg.norm(truth),.05)
            finally:
                submission.__file__=old_file;submission.open_public_case_session=old_open
                os.chdir(old_cwd)
    def test_cohorts_disjoint(self):
        co=json.loads((ROOT/'control/cohorts.json').read_text())
        for task in ('piv','obstacle'):
            sets=[set(co[task+'_'+s]['case_ids']) for s in ('calibration','validation','terminal')]
            self.assertEqual([len(s) for s in sets],[4,24,96])
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    def test_execution_phase_adapter_preserves_query(self):
        from dataclasses import replace
        from pde_agent_verifier_solver.plugins.ns.case_bank import _plan
        from pde_agent_verifier_solver.sdk import EvaluationPhase
        co=json.loads((ROOT/'control/cohorts.json').read_text())
        for cohort in co.values():
            original=_plan(json.loads(Path(cohort['manifest']).read_text())['cases'][0]['plan'])
            adapted=replace(original,phase=EvaluationPhase.TERMINAL)
            self.assertEqual(original.query,adapted.query);self.assertEqual(original.settings,adapted.settings)
if __name__=='__main__':unittest.main()
