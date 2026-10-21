from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.vpfp.terminal_handoff import SavedTerminalReplay, verify_official_aliases
from pde_agent_verifier_solver.engine.candidate_case import candidate_case_id
from pde_agent_verifier_solver.sdk import (Case, CaseRecord, CaseSource, CaseVisibility,
    GenerationKind, GeneratorMode, GenerationPurpose, GenerationRequest, AnswerQuery)


def setup_replay():
    def record(name, round_index, parent=None):
        return CaseRecord(Case(case_id=name, source=CaseSource.PERTURBED if parent else CaseSource.GENERATED,
            visibility=CaseVisibility.HIDDEN, public_data={}, round_index=round_index,
            parent_case_id=parent), object())
    base, child = record("base", 0), record("base-child", 0, "base")
    guided, guided_child = record("guided", 1), record("guided-child", 1, "guided")
    hidden = {"common_base/base": (base,), "common_base/base_perturbed": (child,),
              "common_guided/guided": (guided,), "common_guided/guided_perturbed": (guided_child,)}
    query = AnswerQuery.at_points(fields=("rho",), points=({"t": .5, "x": 0., "y": 0.},))
    queries = {row.case.case_id: query for rows in hidden.values() for row in rows}
    manifest = {"case_count": 4, "first_guidance_sha256": "g1", "manifest_sha256": "bank",
                "cohort_ids": ["common_base", "common_guided"]}
    calls = []
    def generate(*args):
        calls.append(args)
        return ("fresh-nonterminal",)
    backend = SimpleNamespace(generate=generate)
    replay = SavedTerminalReplay(backend, hidden, queries, manifest)
    return replay, backend, calls


def request(kind=GenerationKind.BASE, round_index=0, **kwargs):
    args = dict(round_index=round_index, kind=kind, count=1, mode=GeneratorMode.ONLINE,
                purpose=GenerationPurpose.TERMINAL, visibility=CaseVisibility.HIDDEN, seed=999)
    args.update(kwargs)
    return GenerationRequest(**args)


def issue_bank(replay):
    base = replay.generate(None, None, request())[0]
    replay.generate(None, None, request(GenerationKind.PERTURBED, anchors=(base.case,)))
    guide = SimpleNamespace(checkpoint_sha256="g1")
    guided = replay.generate(None, guide, request(GenerationKind.GUIDED, 1))[0]
    replay.generate(None, guide, request(GenerationKind.PERTURBED, 1, anchors=(guided.case,)))


def test_bank_replay_preserves_saved_root_children_without_sampling_or_training():
    replay, backend, calls = setup_replay()
    original = backend.generate
    replay.install()
    issue_bank(replay)
    assert not calls
    assert replay._issued == set(replay.queries)
    assert replay.calls == 4
    replay.close()
    assert backend.generate is original
    replay.close()


def test_nonterminal_generation_is_unchanged_and_bank_cannot_be_exposed():
    replay, _, calls = setup_replay()
    development = request(purpose=GenerationPurpose.DEVELOPMENT, visibility=CaseVisibility.DEVELOPMENT)
    assert replay.generate("base", "guide", development) == ("fresh-nonterminal",)
    assert calls == [("base", "guide", development, None)]
    with pytest.raises(PermissionError):
        replay.generate(None, None, request(visibility=CaseVisibility.DEVELOPMENT))
    assert not replay._issued


@pytest.mark.parametrize("problem", ["missing_guide", "wrong_guide", "future_round", "repeat", "unknown_parent", "early_child"])
def test_invalid_terminal_replay_never_falls_back_to_native_generation(problem):
    replay, _, calls = setup_replay()
    with pytest.raises(ValueError):
        if problem == "missing_guide":
            replay.generate(None, None, request(GenerationKind.GUIDED, 1))
        elif problem == "wrong_guide":
            replay.generate(None, SimpleNamespace(checkpoint_sha256="g2"), request(GenerationKind.GUIDED, 1))
        elif problem == "future_round":
            replay.generate(None, SimpleNamespace(checkpoint_sha256="g1"), request(GenerationKind.GUIDED, 2))
        elif problem == "repeat":
            replay.generate(None, None, request())
            replay.generate(None, None, request())
        else:
            anchor = replay.hidden["common_base/base"][0].case
            if problem == "unknown_parent":
                from dataclasses import replace
                anchor = replace(anchor, case_id="unknown")
            replay.generate(None, None, request(GenerationKind.PERTURBED, anchors=(anchor,)))
    assert not calls


def test_sealed_bank_checks_order_and_private_points_without_recomputing_truth(monkeypatch):
    replay, _, calls = setup_replay()
    issue_bank(replay)
    from pde_agent_verifier_solver.plugins.vpfp import formal_evaluation_bank
    monkeypatch.setattr(formal_evaluation_bank, "_terminal_records", lambda *_: replay.hidden)
    query = replay.queries["base"]
    controller = SimpleNamespace(
        _hidden_plan_seed_provider=lambda *_: lambda index, case_id: 101,
        campaign=SimpleNamespace(task_settings={}, evaluator=SimpleNamespace(default_query=query)),
        evaluation=SimpleNamespace(make_plan=lambda *_a, **_kw: SimpleNamespace(query=query)))
    report = replay.verify_sealed(controller)
    assert report["case_count"] == 4
    assert report["native_draws_recomputed"] == report["reference_values_recomputed"] == 0
    assert not calls
    changed = AnswerQuery.at_points(fields=("rho",), points=({"t": .6, "x": 0., "y": 0.},))
    controller.evaluation.make_plan = lambda *_a, **_kw: SimpleNamespace(query=changed)
    with pytest.raises(ValueError, match="scoring points"):
        replay.verify_sealed(controller)


def test_official_comparison_preserves_every_batch_order_not_old_scores():
    replay, _, _ = setup_replay()
    hidden = {"base": replay.hidden["common_base/base"] + replay.hidden["common_guided/guided"],
              "perturbed": replay.hidden["common_base/base_perturbed"]}
    expected = [tuple(candidate_case_id(row.case.case_id) for row in rows) for rows in hidden.values()]
    report = verify_official_aliases(hidden, expected[::-1])
    assert report["case_count"] == 3 and report["old_scores_imported"] is False
    with pytest.raises(ValueError):
        verify_official_aliases(hidden, [expected[0][::-1], expected[1]])
    with pytest.raises(ValueError):
        verify_official_aliases(hidden, expected[:1])
