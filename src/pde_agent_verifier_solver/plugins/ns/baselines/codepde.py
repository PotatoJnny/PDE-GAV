"""Independent port of CodePDE's repeated-sampling and conditional-debug policy.

This module does not execute or redistribute the authors' implementation. The
algorithm was checked against the commit and source hashes in
_vendor/CODEPDE_PROVENANCE.json. The upstream sources contain no license file;
no upstream license or permission is inferred here.

Each sample starts from scratch. Execution failures and nonfinite errors may
receive another trial, up to a bound that includes the initial generation.
The first finite error ends that sample, however poor the score. Generation
or result-parsing exceptions end the sample immediately. The next sample is
independent. A final nonfinite result is retained in the trace but is ineligible
for delivery. Selecting the best finite sample on fixed validation, with an
earlier-sample tie break, is an explicit NS addition: upstream only emits a CSV.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Callable, Mapping


PROVENANCE_PATH = Path(__file__).with_name('_vendor') / 'CODEPDE_PROVENANCE.json'


@dataclass(frozen=True)
class CodePDERepeatedSampleConfig:
    num_repeated_samples: int = 2
    num_debugging_trials_per_sample: int = 5

    def __post_init__(self):
        for name in ('num_repeated_samples', 'num_debugging_trials_per_sample'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')

    @property
    def maximum_generation_calls(self):
        return self.num_repeated_samples * self.num_debugging_trials_per_sample


@dataclass(frozen=True)
class CodePDEEvaluation:
    """An execution result, rather than a penalty that conflates all failures.

    A nonzero exit_code triggers execution debugging. For exit_code zero,
    relative_error is required; NaN/Inf trigger numerical debugging. Callbacks
    should return execution failures here, and raise for parsing/infrastructure
    failures, which have different upstream control flow.
    """
    exit_code: int
    relative_error: float | None = None
    stdout: str = ''
    stderr: str = ''

    def __post_init__(self):
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise TypeError('exit_code must be an integer')
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError('execution stdout/stderr must be text')
        if self.exit_code == 0:
            value = self.relative_error
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('successful execution needs a parsed numerical error')
            if math.isfinite(value) and value < 0:
                raise ValueError('NS validation error must be nonnegative')


@dataclass(frozen=True)
class CodePDETrial:
    sample_index: int
    trial_index: int
    generation_mode: str
    parent_key: str | None
    candidate_key: str | None = None
    evaluation: CodePDEEvaluation | None = None
    status: str = 'started'
    exception: str | None = None


@dataclass(frozen=True)
class CodePDEGenerationRequest:
    sample_index: int
    trial_index: int
    generation_mode: str
    parent_key: str | None
    history: tuple[CodePDETrial, ...]


@dataclass(frozen=True)
class CodePDESearchResult:
    selected_key: str | None
    selection_error: float | None
    trials: tuple[CodePDETrial, ...]
    samples: tuple[Mapping[str, object], ...]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_repeated_sample(
    config: CodePDERepeatedSampleConfig, *,
    generate: Callable[[CodePDEGenerationRequest], str | None],
    evaluate: Callable[[str], CodePDEEvaluation],
    initial_key: str | None = None,
    output_directory: str | Path | None = None,
    adaptation_details: Mapping[str, object] | None = None,
    fatal_exceptions: tuple[type[BaseException], ...] = (),
) -> CodePDESearchResult:
    """Execute the source-audited policy with frozen-program callbacks.

    A None generation result denotes parsing/nomination failure and aborts that
    sample; it is not an executable program available for debugging. Raise an
    exception from generate to preserve the detailed cause in the trace.
    Request.history contains only this sample's ordered failed attempts. The
    callback controls how their frozen code and numerical feedback are rendered.
    No S0 is permitted, and no test/terminal evaluation callback is accepted.
    """
    if not isinstance(config, CodePDERepeatedSampleConfig):
        raise TypeError('config must be CodePDERepeatedSampleConfig')
    if initial_key is not None:
        raise ValueError('CodePDE repeated_sample requires cold start; S0 is not permitted')
    root = Path(output_directory).expanduser().resolve() if output_directory is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        if any((root / name).exists() for name in ('codepde_adaptation.json', 'codepde_trials.json')):
            raise FileExistsError('CodePDE output already exists; use a new baseline run')
    metadata = {
        'schema': 'codepde_repeated_sample_ns_adaptation_v1',
        'upstream': json.loads(PROVENANCE_PATH.read_text()),
        'implementation': 'independent_policy_reimplementation',
        'author_code_executed': False,
        'local_adapter_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'paper_end_to_end_reproduction': False,
        'initial_candidate_policy': 'cold_start_only_S0_rejected',
        'num_repeated_samples': config.num_repeated_samples,
        'num_debugging_trials_per_sample': config.num_debugging_trials_per_sample,
        'maximum_generation_calls': config.maximum_generation_calls,
        'trial_budget_includes_initial_generation': True,
        'stop_each_sample': 'first_finite_validation_error',
        'delivery_selection': 'NS_addition_minimum_finite_fixed_validation_error_earlier_sample_ties',
        'terminal_selection': False,
        'fatal_exception_types': [kind.__name__ for kind in fatal_exceptions],
        'adaptation_details': dict(adaptation_details or {}),
    }
    trials, samples = [], []
    selected, best_error = None, math.inf

    def persist(status):
        if root is not None:
            _write(root / 'codepde_adaptation.json', {
                **metadata, 'status': status, 'generation_calls': len(trials),
                'selected_key': selected,
                'selection_error': best_error if selected is not None else None,
                'samples': samples,
            })
            _write(root / 'codepde_trials.json', [asdict(trial) for trial in trials])

    persist('running')
    try:
        for sample_index in range(config.num_repeated_samples):
            history = []
            parent, mode = None, 'initial'
            sample_status = 'generation_failed'
            for trial_index in range(1, config.num_debugging_trials_per_sample + 1):
                request = CodePDEGenerationRequest(sample_index, trial_index, mode, parent, tuple(history))
                row = dict(sample_index=sample_index, trial_index=trial_index,
                           generation_mode=mode, parent_key=parent)
                trials.append(CodePDETrial(**row))
                persist('running')
                try:
                    key = generate(request)
                    if not isinstance(key, str) or not key.strip():
                        raise ValueError('generation produced no frozen candidate (parsing/nomination failure)')
                except Exception as error:
                    trials[-1] = CodePDETrial(**row, status='generation_failed',
                                             exception=f'{type(error).__name__}: {error}')
                    sample_status = 'generation_failed'
                    if isinstance(error, fatal_exceptions):
                        raise
                    break
                row['candidate_key'] = key
                try:
                    feedback = evaluate(key)
                    if not isinstance(feedback, CodePDEEvaluation):
                        raise TypeError('evaluate must return CodePDEEvaluation')
                except Exception as error:
                    trials[-1] = CodePDETrial(**row, status='evaluation_failed',
                                             exception=f'{type(error).__name__}: {error}')
                    sample_status = 'evaluation_failed'
                    if isinstance(error, fatal_exceptions):
                        raise
                    break
                row['evaluation'] = feedback
                if feedback.exit_code != 0:
                    status, mode = 'execution_error', 'debugging_execution_error'
                    sample_status = 'execution_failed'
                elif not math.isfinite(feedback.relative_error):
                    status, mode = 'nonfinite', 'debugging_nan_inf'
                    sample_status = 'nonfinite'
                else:
                    status, sample_status = 'finite', 'completed'
                trial = CodePDETrial(**row, status=status)
                trials[-1] = trial
                history.append(trial)
                if status == 'finite':
                    if feedback.relative_error < best_error:
                        selected, best_error = key, float(feedback.relative_error)
                    break
                parent = key
            final = trials[-1]
            samples.append({
                'sample_index': sample_index, 'status': sample_status,
                'trials': final.trial_index, 'candidate_key': final.candidate_key,
                'relative_error': (final.evaluation.relative_error
                                   if final.evaluation is not None else None),
            })
            persist('running')
    except BaseException:
        persist('interrupted')
        raise
    persist('completed')
    return CodePDESearchResult(selected, best_error if selected is not None else None,
                               tuple(trials), tuple(samples))


def run_codepde_ns(config, *, initial_key, propose, score, history_for, failure_for=None):
    """Bridge the existing NS baseline callbacks without relabelling refinement.

    Config needs codepde_num_repeated_samples and
    codepde_num_debugging_trials_per_sample, plus output_directory. propose has
    the existing (parent, history, phase) signature. It receives only this
    sample's prior Agent notes and numerical feedback. One proposal is an Agent
    turn, not a claim about the number of internal model calls.

    failure_for should return the existing selection_results.failure string.
    In its absence, the adapter can observe only score(key): an infinite score
    is classified as nan_inf, without inventing an underlying execution error.
    """
    from pde_agent_verifier_solver.engine.agent_execution import AgentInfrastructureError

    def generate(request):
        history = []
        for trial in request.history:
            prior = history_for(trial.candidate_key)
            notes = prior[-1][0] if prior else 'No Agent notes.'
            feedback = trial.evaluation
            explanation = (
                f'CodePDE sample {request.sample_index}, prior trial {trial.trial_index}.\n'
                f'Frozen candidate: {trial.candidate_key}\n'
                f'Outcome: {trial.status}; exit_code={feedback.exit_code}; '
                f'validation error={feedback.relative_error!r}.\n'
                f'Output (last 5000 characters):\n{feedback.stdout[-5000:]}\n'
                f'Errors (last 5000 characters):\n{feedback.stderr[-5000:]}'
            )
            history.append((f'{notes}\n\n{explanation}', None))
        return propose(
            request.parent_key, tuple(history),
            f'codepde_sample_{request.sample_index}_trial_{request.trial_index}:{request.generation_mode}',
        )

    def evaluate(key):
        error = float(score(key))
        failure = failure_for(key) if failure_for is not None else None
        if failure and failure != 'nonfinite_selection_error':
            return CodePDEEvaluation(exit_code=1, stderr=str(failure))
        return CodePDEEvaluation(exit_code=0, relative_error=error, stderr=str(failure or ''))

    result = run_repeated_sample(
        CodePDERepeatedSampleConfig(
            config.codepde_num_repeated_samples,
            config.codepde_num_debugging_trials_per_sample,
        ),
        initial_key=initial_key, generate=generate, evaluate=evaluate,
        output_directory=config.output_directory,
        fatal_exceptions=(AgentInfrastructureError,),
        adaptation_details={
            'generation': 'one_project_Agent_turn_per_policy_generation_call',
            'internal_LLM_call_count_controlled_by_this_policy': False,
            'transcript': 'branch_local_Agent_notes_feedback_and_frozen_program_identities',
            'current_failed_candidate': 'frozen_parent_bundle_supplied_to_propose',
            'no_nomination': 'generation_failure_aborts_sample; details_remain_in_baseline_turns',
            'rejected_raw_drafts': 'not_exposed_by_frozen_bundle_callback; syntax_ABI_rejection_cannot_be_repaired_as_a_parent',
            'execution': 'fixed_hidden_NS_validation_mean_primary_error',
            'validation_cache': 'owned_by_shared_NS_score_callback',
            'failure_classification': ('selection_failure_reason' if failure_for is not None
                                       else 'numeric_score_only_execution_cause_unavailable'),
            'infrastructure_exceptions': 'abort_run_instead_of_counting_as_scientific_sample_failures',
            'rate_limit_sleep': 'left_to_model_provider_callback',
            'upstream_prompt_and_solver_ABI': 'replaced_by_NS_Agent_and_frozen_bundle_contract',
        },
    )
    return result.selected_key
