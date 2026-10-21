"""VPFP's thin numerical-pool binding to the shared phase resource scope."""
from contextlib import contextmanager

from pde_agent_verifier_solver.engine.evaluation_resources import HiddenEvaluationResourceScope


class VPFPHiddenEvaluationScope(HiddenEvaluationResourceScope):
    def __init__(self, binding, candidate_execution, *, devices, events=None, pool_factory=None,
                 workers_per_gpu=1, oracle_variant="structured_reuse_v1",
                 source_request_block=8192, macro_point_block=512):
        from .parallel_oracle import _validated_workers_per_gpu
        from .throughput_oracle import oracle_execution_options
        self.workers_per_gpu = _validated_workers_per_gpu(workers_per_gpu)
        self.oracle_execution_options = oracle_execution_options(oracle_variant=oracle_variant,
            source_request_block=source_request_block, macro_point_block=macro_point_block)
        self.runtime = binding.plugin.generator._backend().runtime
        if getattr(self.runtime, "oracle_kernel", None) != "structured_reuse_v1":
            raise ValueError("VPFP evaluation GPU reuse requires structured_reuse_v1")
        if (getattr(candidate_execution, "oracle_inline_array_bytes", None) != 8 * 1024 * 1024
                or getattr(candidate_execution, "oracle_trusted_results", None) is not True):
            raise ValueError("VPFP evaluation GPU reuse requires 8 MiB trusted transport")

        def create_pool(*, cases, devices):
            from .parallel_oracle import VPFPParallelOraclePool
            factory = VPFPParallelOraclePool if pool_factory is None else pool_factory
            # Preserve the default factory contract for existing integrations.
            options = {} if self.workers_per_gpu == 1 else {"workers_per_gpu": self.workers_per_gpu}
            if self.oracle_execution_options["oracle_variant"] != "structured_reuse_v1":
                options.update(self.oracle_execution_options)
            return factory(self.runtime, cases=cases, devices=devices, **options)

        super().__init__(candidate_execution, devices=devices, pool_factory=create_pool,
                         activate=lambda pool: setattr(self.runtime, "_parallel_oracle_pool", pool),
                         deactivate=lambda: setattr(self.runtime, "_parallel_oracle_pool", None),
                         events=events)

    @contextmanager
    def __call__(self, *, cases, unavailable_gpu_devices=()):
        with super().__call__(cases=cases, unavailable_gpu_devices=unavailable_gpu_devices) as pool:
            if pool is not None:
                # The shared scope counts devices; VPFP may explicitly expose
                # multiple independent Oracle hosts on each healthy device.
                self.execution.oracle_host_count *= self.workers_per_gpu
            yield pool

    def _emit(self, event_type, message, payload):
        payload = dict(payload)
        if "gpu_devices" in payload:
            count = len(payload["gpu_devices"])
            payload.update(gpu_device_count=count, workers_per_gpu=self.workers_per_gpu,
                           oracle_process_count=count * self.workers_per_gpu,
                           **self.oracle_execution_options)
            if "oracle_workers" in payload:
                payload["oracle_workers"] = count * self.workers_per_gpu
        super()._emit(event_type, message, payload)
