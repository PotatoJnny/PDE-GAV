"""Reuse the existing process lifecycle for a solver's optional cross-Case ABI."""
class BatchOracleView:
    def __init__(self, host, case_ids):
        self.host = host
        self.case_ids = case_ids
        host.state.register_wait_group(case_ids)
        self.state = _BatchStateView(host.state, case_ids)

    def manifest_for_case(self, case_id):
        return self.host.manifest_for_case(case_id)

    def finish_case(self, _case_id, *, drain_timeout_seconds):
        # Drain every member even when one fails cleanup.
        import time
        deadline = time.monotonic() + drain_timeout_seconds
        error = None
        for case_id in self.case_ids:
            try:
                self.host.finish_case(case_id, drain_timeout_seconds=max(0., deadline-time.monotonic()))
            except BaseException as failure:
                if error is None:
                    error = failure
                else:
                    from .candidate_execution import _attach_execution_error
                    _attach_execution_error(error, failure)
        if error is not None:
            raise error


class _BatchStateView:
    def __init__(self, state, case_ids):
        self._state = state
        self.case_ids = case_ids

    def queue_wait_seconds(self, _case_id):
        return self._state.group_wait_seconds(self.case_ids)

    def execution_metrics(self, case_id):
        return self._state.execution_metrics(case_id)

    def failure_diagnostics(self, _case_id):
        return tuple({**row, "case_id": case_id} for case_id in self.case_ids
                     for row in self._state.failure_diagnostics(case_id))
