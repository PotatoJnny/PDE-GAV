"""Accepted weak H2 + seed0 fixed20k base and lossless sampling-coordinate capture.

This adapter neither fits a model nor re-encodes the quantized answer.  Sampling
delegates to the original joint sampler and original Heun implementation.  The
caller runs real loads/draws in a compute allocation and registers fresh roots.
"""
from __future__ import annotations

from copy import copy, deepcopy
from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np  # Frozen native initialization order: NumPy BEFORE Torch.
import torch

from pde_agent_verifier_solver.plugins.vpfp.innovation_binding import (
    InnovationMovingFrameSampler, _module_sha, json_sha)
from pde_agent_verifier_solver.plugins.vpfp.hybrid_stress_diffusion import (
    H2_SCHEMA, load_hybrid_stress_macro_checkpoint)
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import _seeds

SCHEMA = 'vpfp_weak_guidance_accepted_base_v1'
MAIN_CHECKPOINT = {
    'path': '${PDE_GAV_RUN_ROOT}/vpfp-weak-replay-main-seed0-5357246/main/segment-000/checkpoints/fixed-step-020000.pt',
    'sha256': 'dad163578cbe7da8bd934ff16dc99bbc4155e66175d2c834db3e69d80df4fd18'}


def _load_h2_prior(item, sampler, device):
    """Do NOT use the ancestral composition's drive4 prior for fresh H2 draws."""
    macro, normalizer, payload = load_hybrid_stress_macro_checkpoint(
        item['macro_checkpoint']['path'], item['macro_checkpoint']['sha256'], device=device)
    if (payload['schema'] != H2_SCHEMA or payload['training']['step'] != 5000
            or payload['training']['seed'] != item['macro_training_seed']
            or payload['corpus_manifest_sha256'] != sampler.identity['corpus_manifest_sha256']
            or payload['base_decoder_sha256'] != sampler.identity['assets']['base_decoder']['sha256']
            or macro.config.base_macro_config != sampler.micro.macro_config
            or normalizer.fit_metadata['sample_ids'] != sampler.micro_norm.fit_metadata['sample_ids']):
        raise ValueError('accepted H2 prior and micro TRAIN lineage/geometry differ')
    return macro.eval().requires_grad_(False), normalizer.eval().requires_grad_(False)


def load_base(device='cuda', *, guard=lambda: None):
    """Return (fresh accepted sampler, full continuous A decoder, bound receipt).

The old sampler is used unmodified for the original20k strict replay load.
Only after that load is a new H2+20k joint sampler constructed.  All original
normalizers and complete innovation coordinates remain unchanged.
"""
    import run_vpfp_weak_main_stage_audit as stage
    import check_vpfp_weak_generated_source_exposure as exposure
    files = {}; guard()
    inputs = stage.freeze_inputs(files)
    checkpoint = inputs['mains']['0']['checkpoints']['20000']
    if checkpoint != MAIN_CHECKPOINT:
        raise ValueError('only explicitly accepted seed0 fixed20k EMA is a guidance base')
    ancestral, decoder = stage.pilot.load_initial(inputs['initial']['0'], device)
    physical = stage.pilot.context_for(ancestral, decoder, inputs['budget'],
        inputs['budget_item']['sha256'], stage.jp.SOURCE_WEIGHT)
    replay = stage.screen.ReplayContext(physical, inputs['teachers']['0'],
        replica=0, device=device, guard=guard)
    micro, binding = stage.load_stage(inputs, replica=0, step=20000,
        sampler=ancestral, decoder=decoder, replay=replay)
    prior = exposure.freeze_teacher_assets(0, files)
    if prior['composition'] != inputs['initial']['0']['composition']:
        raise ValueError('accepted H2 prior and20k ancestral composition differ')
    macro, prior_norm = _load_h2_prior(prior, ancestral, device)
    # Active assets are explicit; the old manifest remains ancestry, not a
    # false claim that its old micro/prior checkpoint is the sampled model.
    identity = deepcopy(ancestral.identity)
    identity['ancestral_identity'] = deepcopy(ancestral.identity)
    identity.pop('manifest_sha256', None)
    identity.update(schema=SCHEMA, accepted_working_baseline=True,
        acceptance_scope='approximate_weak_shape_and_spatial_coverage_working_use',
        guidance_qualified=False, scientific_admission=False,
        micro_weights='accepted_seed0_fixed20000_EMA_not_raw_last_best_or1000',
        main_experiment_sha256=binding)
    identity['assets']['macro_checkpoint'] = deepcopy(prior['macro_checkpoint'])
    identity['assets']['conditional_checkpoint'] = deepcopy(checkpoint)
    identity['assets']['training_contract'] = {
        'macro': {'step': 5000, 'seed': prior['macro_training_seed']},
        'conditional': {'step': 20000, 'seed': stage.jp.SEEDS[0]}}
    identity['sampling']['macro_prior'] = 'frozen_H2_split_head_fixed5000'
    identity['sampling']['condition_prior'] = 'explicit_registered_physical_nu_temperature'
    sampler = InnovationMovingFrameSampler(macro, prior_norm, micro, ancestral.micro_norm,
        ancestral.macro_norm, ancestral.transform, identity=identity, steps=64)
    ancestral.assert_frozen(); sampler.assert_frozen(); guard()
    stage.auditing.old.check_frozen(files)
    receipt = {'schema': SCHEMA, 'replica': 0, 'fixed_step': 20000,
        'checkpoint': deepcopy(checkpoint), 'macro_checkpoint': deepcopy(prior['macro_checkpoint']),
        'main_result': deepcopy(stage.MAIN_RESULTS['0']), 'main_experiment_sha256': binding,
        'model_state_sha256': _module_sha(micro), 'macro_state_sha256': _module_sha(macro),
        'sampler_identity': deepcopy(identity), 'sampler_identity_sha256': json_sha(identity),
        'frozen_inputs': files, 'frozen_input_sha256': json_sha(files),
        'native_strict_load_passed': True, 'sampling_steps': 64,
        'new_training_steps': 0, 'new_PDE_solves': 0, 'new_samples': 0,
        'fresh_joint_draw_uses_H2_not_ancestral_drive4': True,
        'working_base_accepted': True, 'guidance_qualified': False}
    return sampler, decoder, receipt


class _CaptureModel:
    """Read-only proxy: reuse the bound model's original sampler, not an ODE copy."""
    def __init__(self, model, stage, head, random_seed):
        self.model, self.stage, self.head, self.random_seed = model, stage, head, random_seed
        self.white = self.condition = None
        self.diagnostics = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def sample(self, conditions, seeds, *, steps=64, **kwargs):
        self.condition = conditions.detach().clone()
        self.seeds, self.calls, self.calls_per_root = _seeds(seeds), 0, 2*steps-1
        if self.head is None or self.head.strength == 0:
            white = self.model.sample(conditions, seeds, steps=steps, **kwargs)
        else:
            # Native sample dispatches self.denoise through this proxy; all
            # other members/methods remain those of the frozen original model.
            white = self.model.sample.__func__(self, conditions, seeds, steps=steps, **kwargs)
            if self.calls != len(self.seeds)*self.calls_per_root:
                raise ValueError('native Heun denoise call contract changed')
        self.white = white.detach().clone()
        return white

    def denoise(self, noisy, sigma, conditions):
        clean = self.model.denoise(noisy, sigma, conditions)
        root_position, call = divmod(self.calls, self.calls_per_root)
        seed = self.seeds[root_position]
        random_seed = None if self.random_seed is None else int.from_bytes(sha256(
            f'{SCHEMA}:random-guidance:{self.random_seed}:{self.stage}:{seed}:{call}'.encode()
            ).digest()[:8], 'little') % 2**63
        corrected, diagnostics = self.head.correct(clean, noisy, sigma, conditions[:, :2],
            macro_context=conditions[:, 2:] if self.stage == 'micro' else None,
            random_seed=random_seed)
        if (not isinstance(corrected, torch.Tensor) or corrected.shape != clean.shape
                or corrected.dtype != clean.dtype or corrected.device != clean.device
                or not bool(torch.isfinite(corrected).all())):
            raise ValueError('guide correction must preserve finite native denoised tensor ABI')
        self.diagnostics.append({'noise_seed': seed, 'denoise_call': call,
            'random_seed': random_seed, 'diagnostics': diagnostics})
        self.calls += 1
        return corrected


def draw_base(sampler, conditions, root_seeds, *, macro_guide=None, micro_guide=None, random_seed=None):
    """Capture actual clean white draws and the exact micro conditioning vector.

Guides implement ``correct(clean,noisy,sigma,nu_T,macro_context=...,random_seed=...)``.
At strength0 the original sample method is called directly.  No sampled answer
is re-encoded, no ancestor model is modified, and no PDE solve is performed.
All returned captured tensors are detached CPU tensors, matching native m/z.
"""
    if not isinstance(sampler, InnovationMovingFrameSampler):
        raise TypeError('native complete innovation joint sampler required')
    if random_seed is not None:
        random_seed = _seeds([random_seed])[0]
    roots = _seeds(root_seeds)
    for stage_name, head in (('macro', macro_guide), ('micro', micro_guide)):
        if head is not None and (not callable(getattr(head, 'correct', None))
                or getattr(head, 'stage', stage_name) != stage_name):
            raise ValueError('guide stage/correction interface differs')
    sampler.assert_frozen()
    view = copy(sampler)
    view.macro = _CaptureModel(sampler.macro, 'macro', macro_guide, random_seed)
    view.micro = _CaptureModel(sampler.micro, 'micro', micro_guide, random_seed)
    try:
        draw = view.sample_conditions(conditions, roots)
        draw.update(white_m=view.macro.white.cpu(), white_z=view.micro.white.cpu(),
            condition_m=view.micro.condition[:, 2:].cpu(),
            conditions=view.macro.condition.cpu(), roots=list(roots),
            guidance_diagnostics={'macro': view.macro.diagnostics, 'micro': view.micro.diagnostics},
            captured_original_white_not_reencoded=True, sampling_steps=sampler.steps)
        return draw
    finally:
        sampler.assert_frozen()
