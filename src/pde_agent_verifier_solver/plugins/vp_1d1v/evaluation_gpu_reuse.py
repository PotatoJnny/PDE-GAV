"""Connect Artemis' numerical pool to the shared hidden-evaluation scope."""

from pde_agent_verifier_solver.engine.evaluation_resources import HiddenEvaluationResourceScope


class ArtemisHiddenEvaluationScope(HiddenEvaluationResourceScope):
    def __init__(self, binding, candidate_execution, *, backend_settings, devices,
                 events=None, pool_factory=None):
        self.backend = binding.plugin.generator._backend()
        self.backend_settings = dict(backend_settings)

        def create_pool(*, cases, devices):
            if pool_factory is None:
                from .parallel_oracle import ArtemisParallelOraclePool
                factory = ArtemisParallelOraclePool
            else:
                factory = pool_factory
            return factory(self.backend.source, backend_settings=self.backend_settings,
                           devices=devices, cases=cases)

        super().__init__(
            candidate_execution, devices=devices, pool_factory=create_pool,
            activate=lambda pool: setattr(self.backend, "_parallel_oracle_pool", pool),
            deactivate=lambda: setattr(self.backend, "_parallel_oracle_pool", None),
            events=events,
        )
