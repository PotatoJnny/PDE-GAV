"""Time-aware retries for an independently calibrated source-mesh guard.

Every attempted frame still passes the unchanged all-Hermite pilot guard.
Rejected frames must be recomputed by the caller on its ordinary source mesh.
Only the retry schedule changes; no error tolerance is weakened.
"""
from __future__ import annotations
import math


class SpatialTrialRetry:
    def __init__(self, guard, *, nu, t_interval, initial_after=None):
        self.guard = guard
        self.nu = float(nu)
        self.t0, self.t1 = map(float, t_interval)
        if (not math.isfinite(self.nu) or self.nu < 0
                or not math.isfinite(self.t0) or not math.isfinite(self.t1)
                or self.t1 <= self.t0):
            raise ValueError('finite nonnegative collision rate and ordered time interval required')
        self.calibrated = bool(guard.enabled)
        self.trial_after = (self.t0 + 4/self.nu if self.nu > 0 else math.inf)
        if initial_after is not None:
            self.trial_after = float(initial_after)
            if math.isnan(self.trial_after):
                raise ValueError('non-NaN initial retry time required')
        self.backoff = .5/self.nu if self.nu > 0 else math.inf
        self.endpoint_rejected = False
        self.failures = []

    @property
    def enabled(self):
        return self.calibrated and not self.endpoint_rejected and self.trial_after <= self.t1

    def should_attempt(self, t):
        t = float(t)
        return self.enabled and self.t0 <= t <= self.t1 and t >= self.trial_after

    def accept(self, t, modes, pilot_values):
        """Guard one acquired trial frame and update the future trial schedule."""
        t = float(t)
        if not self.should_attempt(t):
            raise ValueError('trial frame is outside its calibrated retry schedule')
        # Capture calibration once, before rejection-based disabling can occur.
        # A guard that failed its original calibration can never be re-enabled.
        self.guard.enabled = self.calibrated
        accepted = self.guard.accept(modes, pilot_values)
        if not accepted:
            self.failures.append(t)
            self.trial_after = max(self.trial_after, t + self.backoff)
            self.endpoint_rejected = t == self.t1
            # The retry now moves to later time. Preserve total diagnostics but
            # reset the guard's local consecutive-failure disable state.
            self.guard.consecutive_rejections = 0
        self.guard.enabled = self.calibrated and not self.endpoint_rejected
        return accepted

    def diagnostics(self):
        return dict(self.guard.diagnostics(), calibrated=self.calibrated,
                    retry_enabled=self.enabled, trial_after=self.trial_after,
                    retry_backoff=self.backoff, endpoint_rejected=self.endpoint_rejected,
                    rejected_times=list(self.failures))
