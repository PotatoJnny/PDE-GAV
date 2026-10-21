#!/usr/bin/env python3
"""Run measured Fixed/Full through the original NS segmented experiment dispatcher."""
from __future__ import annotations

import os
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]


def _private_calibration_records(records):
    """Take a controller-only view without changing the Fixed development set.

    Numerical common calibration likewise loads its original split with hidden
    visibility.  These copies are only for S0 calibration, never promotion or
    terminal membership.  Build the public RPC provider from the same copies so
    its exact Case identity check still applies.
    """
    from dataclasses import replace
    from pde_agent_verifier_solver.sdk import CaseVisibility

    rows = tuple(records)
    if not rows or any(row.case.visibility is not CaseVisibility.DEVELOPMENT for row in rows):
        raise ValueError('calibration requires original development Cases')
    return tuple(replace(row, case=replace(row.case, visibility=CaseVisibility.HIDDEN)) for row in rows)


def preflight(config_root,batch,tokens,job):
    from run_canonical_ns_campaign import _load_config,_validate_config,_compose
    from run_ns_draft_job import validate_prepared_banks,_require_finite_selection,_write,PAPER_PROTOCOL
    from prepare_ns_mixed_evaluation import prepare
    from pde_agent_verifier_solver.engine import CampaignController
    from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
    from pde_agent_verifier_solver.plugins.ns.case_bank import load_ns_continuous_case_bank,NSBankSessionProvider
    from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import ExperimentalMMSProblem
    from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
    from pde_agent_verifier_solver.reporting.costs import MeasuredGenerator
    from pde_agent_verifier_solver.sdk import GenerationPurpose, DatasetInput, DatasetKind, CaseAccessKind
    import numpy as np
    workspace=batch/'preflight'/f'attempt-{job}';workspace.mkdir(parents=True,exist_ok=False)
    config,path=_load_config(config_root/'full_s101.json');_validate_config(config)
    run=workspace/'reference_runtime';run.mkdir();composition=None
    try:
        composition=_compose(config=config,config_path=path,run_root=run,
            codex_binary=Path(os.environ.get('CODEX_BINARY', 'codex')),auth_file=Path(os.environ.get('CODEX_AUTH_FILE', os.path.expanduser('~/.codex/auth.json'))),
            worker_gpus=tokens[:2],controller_gpu=tokens[2])
        composition.generator=MeasuredGenerator(composition.generator,workspace/'generator_operations.jsonl')
        controller=CampaignController(campaign=composition.campaign,dataset=composition.dataset,
            problem=composition.problem,task=composition.task,generator=composition.generator,
            agents=composition.agents,agent_execution=composition.execution,
            public_problem_bundle=composition.public_problem_bundle,candidate_execution=composition.candidate_execution)
        prepared=workspace/'mixed_evaluation'
        prepare(controller,prepared,promotion_from_config=True,terminal_from_config=True,
                selection_suite=composition.campaign.rounds[0].round_hidden)
        checks=validate_prepared_banks(prepared)
        bank=load_ns_continuous_case_bank(prepared/'fixed_selection.json',expected_purpose=GenerationPurpose.PROMOTION)
        settings={**composition.campaign.task_settings,'default_query':composition.campaign.evaluator.default_query}
        verifier=HiddenVerifier(bank.fixed_evaluation(composition.task.create_evaluation(settings)),
            composition.campaign.evaluator.default_query,settings,seed=101,
            generator=bank.generator,generator_state=bank.generator_state,candidate_execution=composition.candidate_execution)
        evaluation=verifier.evaluate(composition.campaign.initial_candidate,bank.records)
        _require_finite_selection(evaluation,bank.records)
        _write(workspace/'s0_mixed_selection.json',{'status':'passed','case_count':len(evaluation.cases),
            'primary_error':evaluation.primary_error,'cases':[{'case_id':r.case_id,'primary_error':r.primary_error} for r in evaluation.cases],
            'scope':'Same full192 mixed S0 selection preflight as active numerical; no final-test scores.'})
        fixed_config=json.loads((config_root/'fixed_s101.json').read_text())
        fixed_path=Path(fixed_config['dataset']['manifest'])
        if fixed_config['dataset'].get('kind') == 'experimental_mms':
            fixed=ExperimentalMMSProblem().prepare_dataset(DatasetInput('fixed-input-check',DatasetKind.EXPERIMENTAL,location=str(fixed_path)))
            development=fixed.development_cases
            rows=_private_calibration_records(development[:50])
            provider=NSBankSessionProvider((SimpleNamespace(records=rows,manifest_sha256=sha256(fixed_path.read_bytes()).hexdigest(),
                _answers={row.case.case_id:row.truth._answer for row in rows}),))
            boundary_sessions=((provider,rows[:1]),(bank.generator,bank.records[:1]))
        else:
            from pde_agent_verifier_solver.plugins.ns.common_evaluation import load_numerical_split
            from pde_agent_verifier_solver.sdk import CaseVisibility
            development=load_numerical_split(fixed_path,split='development',visibility=CaseVisibility.DEVELOPMENT)
            rows=_private_calibration_records(development[:50])
            provider=None
            boundary_sessions=((bank.generator,bank.records[:1]),)
            for record in rows:
                arrays=record.case.public_data['arrays']
                if 'boundary_velocity' in arrays:
                    with np.load(record.truth.case_dir/'hidden_truth_cache.npz',allow_pickle=False) as raw:
                        np.testing.assert_array_equal(arrays['initial_velocity'],raw['u_true'][0])
                        np.testing.assert_array_equal(arrays['boundary_velocity'],raw['u_true'][:,:,arrays['outer_boundary_mask']])
        if len(development)!=512:raise ValueError('Fixed must have512 original-source development windows')
        boundary_checks=[]
        for selected_provider,selected_rows in boundary_sessions:
            if 'query_boundary_velocity' not in selected_provider.case_session_methods(CaseAccessKind.PUBLIC_PROBLEM):
                raise ValueError('declared open boundary is not exposed through public bank session')
            for record in selected_rows:
                answer=record.truth._answer; xmin,xmax,ymin,ymax=answer.domain_bounds;t0,t1=answer.time_interval
                query=np.array([[(t0+t1)/2,xmin,(ymin+ymax)/2],[t1,(xmin+xmax)/2,ymax]])
                session=selected_provider.open_case_session(selected_provider.state,record.case,CaseAccessKind.PUBLIC_PROBLEM)
                actual=np.asarray(session.query_boundary_velocity(query));expected=np.asarray(answer.evaluate_velocity(query))
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12);session.close()
                boundary_checks.append({'case_id':record.case.case_id,'maximum_absolute_error':float(np.max(np.abs(actual-expected)))})
        calibration_settings={**settings,'official_point_count':128}
        calibration_verifier=HiddenVerifier(NSSolverTask().create_evaluation(calibration_settings),
            composition.campaign.evaluator.default_query,calibration_settings,seed=12345,
            generator=provider,generator_state=provider.state if provider is not None else None,candidate_execution=composition.candidate_execution)
        calibration=calibration_verifier.evaluate(composition.campaign.initial_candidate,rows)
        _require_finite_selection(calibration,rows)
        _write(workspace/'s0_calibration/summary.json',{'status':'completed','n_cases':50,'n_failed_program_cases':0,
            'primary_error':calibration.primary_error,'scope':'50 original-development inputs,128 points; finite arrays or explicitly declared continuous legacy input.',
            'boundary_checks':boundary_checks,'cases':[{'case_id':r.case_id,'primary_error':r.primary_error} for r in calibration.cases]})
        published=batch/'control/mixed_evaluation'
        if published.exists():raise FileExistsError(published)
        prepared.rename(published);validate_prepared_banks(published)
        _write(batch/'control/preflight_passed.json',{'job_id':job,'status':'passed','protocol_revision':PAPER_PROTOCOL,
            'workspace':str(workspace),'comparison_suite':str(published/'comparison_suite.json'),
            'bank_checks':checks,'completed_at_unix':time.time(),
            'scope':'Real original Generator,G5000 on64 feedback,exact192/768 banks,S0 selection192 and original calibration50; no coding Agent or terminal solver scores.'})
        return 0
    finally:
        if composition is not None:
            for service in (composition.execution,composition.candidate_execution):
                close=getattr(service,'close',None)
                if callable(close):close()


if __name__=='__main__':
    from run_ns_draft_job import main
    raise SystemExit(main(methods=('fixed','full'),preflight_fn=preflight))
