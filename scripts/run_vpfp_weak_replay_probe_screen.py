"""Inference-only compatibility class for the frozen weak ReplayEDM checkpoint.

This is not the historical training driver. Constructor and identity methods
are retained verbatim; the inherited numerical denoiser/sample implementation
is source-checked by weak_pde_gav_binding. Training is deliberately unavailable.
"""
from pde_agent_verifier_solver.plugins.vpfp.source_aware_training import VPFPSourceAwareTrainingEDM

ARMS=('baseline','fixed4','rotating4','stratified32')


class ReplayEDM(VPFPSourceAwareTrainingEDM):
    def __init__(self,config,macro_config,*,spatial_domain_lengths,replay,probe_arm):
        if probe_arm not in ARMS:raise ValueError('unregistered probe arm')
        super().__init__(config,macro_config,arm='baseline',spatial_domain_lengths=spatial_domain_lengths)
        self.replay_context,self.probe_arm=replay,probe_arm
        self.checkpoint_schema='vpfp_replay_probe_'+probe_arm+'_innovation_checkpoint_v1'
    def identity(self):
        result=super().identity()
        result.update(probe_arm=self.probe_arm,replay=self.replay_context.binding,
            metrics='sigma_bins_original_EDM_only_source_separate_not_assigned_to_original_sigma',
            objective='original_EDM_plus_generated_state_source_budget_not_teacher_MSE')
        return result

    def loss(self, *args, **kwargs):
        raise RuntimeError('portable accepted weak Base is inference-only; use the original training archive for retraining')
