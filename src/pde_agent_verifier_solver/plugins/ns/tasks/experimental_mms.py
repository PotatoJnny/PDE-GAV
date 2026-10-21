"""Experimental-data NS task using the standard solver ABI and score."""
from pde_agent_verifier_solver.sdk import AnswerQuery, CaseSource, EvaluationPhase, ProgramInfo
from .solver import NSEvaluationProgram, NSSolverTask, _private_development_query


class ExperimentalMMSEvaluationProgram(NSEvaluationProgram):
    """Check scientific split ownership before hidden truth or solver execution."""

    def make_plan(self, case, phase, settings, seed, request=None):
        if case.source is CaseSource.INITIAL and phase in (
            EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL,
        ):
            split = case.public_data.get("distribution_lineage", {}).get("split")
            expected = "verifier" if phase is EvaluationPhase.OFFICIAL else "final_test"
            if split != expected:
                raise ValueError(
                    f"experimental MMS {phase.value} Original must use {expected}, "
                    f"received {split!r}; reserve the entire final_test split for Terminal"
                )
        return super().make_plan(case, phase, settings, seed, request)


class NSExperimentalMMSSolverTask(NSSolverTask):
    info = ProgramInfo(
        "ns-experimental-mms-solver", "Experimental-data NS solver",
        "Find a solver for continuous NS problems manufactured from reconstructed experimental fields.",
    )

    def setup_fields(self):
        # This task evaluates the reconstructed two-component velocity. The
        # manufactured pressure convention is fixed in prepared data.
        return tuple(field for field in super().setup_fields()
                     if field.name not in {"primary_metric", "include_pressure"})

    def create_evaluation(self, settings):
        if settings.get("include_pressure", False) is not False:
            raise ValueError("experimental MMS scores velocity only; include_pressure must be false")
        if settings.get("primary_metric", "velocity_relative_l2") != "velocity_relative_l2":
            raise ValueError("experimental MMS primary_metric is fixed to velocity_relative_l2")
        query = settings.get("default_query")
        if isinstance(query, AnswerQuery) and set(query.fields) != {"velocity_x", "velocity_y"}:
            raise ValueError("experimental MMS evaluation requests both velocity components only")
        return ExperimentalMMSEvaluationProgram.from_settings(
            program_id="ns-full-field-solver-evaluation-v1",
            settings=settings,
            private_development_query=_private_development_query,
            execute_candidate=self.execute_candidate,
            score_candidate=self.score_candidate,
        )

    def agent_brief(self, settings):
        return (
            "Solve the declared incompressible NS initial-boundary problem. "
            "Use the public initial-velocity, forcing and boundary-velocity queries. "
            "Boundary velocity may be nonzero and time dependent; the Case declares its geometry. "
            "The experimental observations and reconstruction coefficients are not solver inputs. "
            "Pressure is an explicitly manufactured model, not measured experimental pressure. "
            + super().agent_brief(settings)
        )

    def execute_candidate(self, artifact, case, plan):
        if "observations" in case.public_data or "project_reconstruction" in case.public_data:
            raise ValueError("experimental MMS solver requires manufactured inputs, not a PIV reconstruction Case")
        if case.public_data.get("boundary_condition", {}).get("kind") != "manufactured_velocity_trace":
            raise ValueError("experimental MMS solver needs its declared manufactured velocity boundary")
        return super().execute_candidate(artifact, case, plan)
