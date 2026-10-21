"""Bind development evaluation to the trusted requesting Agent, not the artifact author."""

class AgentCandidateExecution:
    def __init__(self, service, agent_id):
        self._service = service
        self.agent_id = agent_id

    def __getattr__(self, name):
        return getattr(self._service, name)

    @property
    def gpu_devices(self):
        return self._service.development_gpu_devices[self.agent_id]

    def execute_cases(self, artifact, cases, plans, evaluation, **kwargs):
        if "_development_agent_id" in kwargs:
            raise ValueError("the Controller owns the evaluation route")
        return self._service.execute_cases(artifact, cases, plans, evaluation,
            _development_agent_id=self.agent_id, **kwargs)

    def execution_settings_snapshot(self):
        return {**self._service.execution_settings_snapshot(),
                "requester_agent_id": self.agent_id,
                "gpu_devices": list(self.gpu_devices)}
