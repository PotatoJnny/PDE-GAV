"""Guarded free-streaming interaction-frame interpolation of exact source data.

All decisions use the complete causal source panel during preparation. A full
Fourier row uses one time interpolant across velocity, and conjugate rows make
the same decision. The baseline source remains available for rejected rows.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import make_interp_spline

GATE_RATIO = 0.25
MIN_RELATIVE_DEFECT = 1.e-10
CELL_RELATIVE_RESOLUTION = .03
CELL_ABSOLUTE_MACRO_BOUND = 1.e-10
FINE_EXCEPTION_MACRO_BUDGET = 3.e-5
FINE_TOTAL_EXCEPTION_RMS_CAP = 1.e-4


def improve_coefficients(panel, times, kx, velocity, quadrature,
                         coefficients, transform, validate=None):
    """Return raw/frame B-spline coefficients and a whole-row phase mask.

    Even-indexed slices predict the held-out odd slices. The weighted L2
    error must improve fourfold in *both* members of a conjugate pair. This
    is an error indicator, not an interpolation certificate; synthetic
    transport checks remain necessary before using it in a candidate.
    """
    nt, nx, nv = panel.shape
    info = dict(method='interaction_frame_row_gate', source_nodes=nt,
                accepted_rows=0, gate_ratio=GATE_RATIO, attempted=False)
    mask = torch.zeros((nx, 1), dtype=torch.bool, device=panel.device)
    if nt < 33 or nt % 2 == 0:
        info['reason'] = 'requires_complete_odd_panel'
        return coefficients, mask, info
    info['attempted'] = True
    fsource = torch.fft.fft(panel, dim=1)
    fsource[:, nx//2] = 0.
    dtype, device = fsource.dtype, fsource.device
    relative_times = torch.as_tensor(np.asarray(times)-times[0], dtype=torch.float64, device=device)
    frequencies = torch.as_tensor(np.asarray(kx)[:, None]*np.asarray(velocity)[None, :],
                                  dtype=torch.float64, device=device)
    phase = torch.exp(1j*relative_times[:, None, None]*frequencies[None])
    demodulated = phase*fsource
    ncoarse = (nt+1)//2
    nheld = (nt-1)//2
    test_matrix = make_interp_spline(times[::2], np.eye(ncoarse), k=7, axis=0)(times[1::2])
    test_matrix = torch.as_tensor(test_matrix, dtype=dtype, device=device)
    raw_predicted = (test_matrix@fsource[::2].reshape(ncoarse, -1)).reshape(nheld, nx, nv)
    frame_predicted = (test_matrix@demodulated[::2].reshape(ncoarse, -1)).reshape(nheld, nx, nv)
    weights = torch.as_tensor(quadrature, dtype=torch.float64, device=device)[None, None, :]
    raw_point = torch.sum(weights*torch.abs(raw_predicted-fsource[1::2])**2, dim=0)
    frame_point = torch.sum(weights*torch.abs(frame_predicted-demodulated[1::2])**2, dim=0)
    norm_point = torch.sum(weights*torch.abs(fsource[1::2])**2, dim=0)
    raw_error = raw_point.sum(dim=1)
    frame_error = frame_point.sum(dim=1)
    norm = norm_point.sum(dim=1)
    accept = ((frame_error <= GATE_RATIO**2*raw_error)
              & (raw_error > MIN_RELATIVE_DEFECT**2*norm)
              & (norm > 1.e-24*torch.max(norm))
              & torch.isfinite(raw_error) & torch.isfinite(frame_error))
    preliminary = accept.clone()
    # A dominant resonant core must not hide an unresolved weak hot tail.
    # Preserve one interpolation operator for the entire Fourier row, but
    # veto it if any significant DG velocity cell is unresolved in the frame.
    # The absolute exception bounds sampled raw-moment defects by Cauchy--
    # Schwarz; it is only a holdout indicator, not a continuum certificate.
    cell_order = 6
    if nv % cell_order:
        accept[:] = False
        info['cell_veto_reason'] = 'unsupported_velocity_layout'
    else:
        ncells = nv//cell_order
        raw_cell = raw_point.reshape(nx, ncells, cell_order).sum(dim=2)
        frame_cell = frame_point.reshape(nx, ncells, cell_order).sum(dim=2)
        norm_cell = norm_point.reshape(nx, ncells, cell_order).sum(dim=2)
        moment_scales = np.array([1.0041253663782175, .4788345668393978,
                                  2.930763214320111, 1.9801548942236534])
        moment_factor = np.max(np.sqrt(np.sum(
            (np.asarray(quadrature)[None, :]*np.asarray(velocity)[None, :]**
             (2*np.arange(4))[:, None]).reshape(4, ncells, cell_order), axis=2))
            /moment_scales[:, None], axis=0)
        moment_factor = torch.as_tensor(moment_factor, device=device)[None, :]
        macro_bound = ((times[-1]-times[0])/nx
                       *torch.sqrt(frame_cell/len(times[1::2]))*moment_factor)
        harmless = macro_bound <= CELL_ABSOLUTE_MACRO_BOUND
        resolved = frame_cell <= CELL_RELATIVE_RESOLUTION**2*norm_cell
        improves = frame_cell <= raw_cell
        cell_pass = torch.all(harmless | (resolved & improves), dim=1)
        info['rows_before_cell_veto'] = int(accept.sum().item())
        accept &= cell_pass
        info['cell_relative_resolution'] = CELL_RELATIVE_RESOLUTION
        info['cell_absolute_macro_bound'] = CELL_ABSOLUTE_MACRO_BOUND
    # Optional independent exact slices assess the complete final spline.
    # Missing/incomplete validation keeps the strict even-to-odd cell gate.
    info['validation_attempted'] = False
    info['validation_completed'] = False
    if validate is not None and bool(torch.any(preliminary).item()) and nv % cell_order == 0:
        checked = validate()
        if checked is not None:
            validation_times, validation_samples, validation_info = checked
            info.update(validation_info)
            actual = torch.fft.fft(torch.as_tensor(validation_samples,
                                   dtype=panel.dtype, device=device), dim=1)
            actual[:, nx//2] = 0.
            ncheck = len(validation_times)
            fine_design = make_interp_spline(times, np.eye(nt), k=7, axis=0)(validation_times)
            fine_design = torch.as_tensor(fine_design, dtype=dtype, device=device)
            raw_fine = (fine_design@fsource.reshape(nt, -1)).reshape(ncheck, nx, nv)
            frame_fine = (fine_design@demodulated.reshape(nt, -1)).reshape(ncheck, nx, nv)
            check_times = torch.as_tensor(np.asarray(validation_times)-times[0],
                                          dtype=torch.float64, device=device)
            check_phase = torch.exp(1j*check_times[:, None, None]*frequencies[None])
            fine_raw_point = torch.sum(weights*torch.abs(raw_fine-actual)**2, dim=0)
            fine_frame_point = torch.sum(weights*torch.abs(frame_fine-check_phase*actual)**2, dim=0)
            fine_norm_point = torch.sum(weights*torch.abs(actual)**2, dim=0)
            fine_raw = fine_raw_point.sum(dim=1)
            fine_frame = fine_frame_point.sum(dim=1)
            info['_validated_row_error'] = fine_frame
            fine_norm = fine_norm_point.sum(dim=1)
            fine_accept = (preliminary & (fine_frame <= GATE_RATIO**2*fine_raw)
                           & (fine_raw > MIN_RELATIVE_DEFECT**2*fine_norm)
                           & torch.isfinite(fine_raw) & torch.isfinite(fine_frame))
            fine_raw_cell = fine_raw_point.reshape(nx, ncells, cell_order).sum(dim=2)
            fine_frame_cell = fine_frame_point.reshape(nx, ncells, cell_order).sum(dim=2)
            fine_norm_cell = fine_norm_point.reshape(nx, ncells, cell_order).sum(dim=2)
            fine_bound = ((times[-1]-times[0])/nx
                          *torch.sqrt(fine_frame_cell/ncheck)*moment_factor)
            fine_good_cells = ((fine_frame_cell <= CELL_RELATIVE_RESOLUTION**2*fine_norm_cell)
                               & (fine_frame_cell <= fine_raw_cell))
            # A weak warm field-response component can share a Fourier row
            # with a cold resonant beam. Sum the sampled macro-defect proxy
            # over every exceptional cell; the budget is per complete row.
            exceptional_bound = torch.where(fine_good_cells, 0., fine_bound).sum(dim=1)
            info['_row_exception_bound'] = exceptional_bound
            fine_pass = exceptional_bound <= FINE_EXCEPTION_MACRO_BUDGET
            strict_count = int(accept.sum().item())
            accept = fine_accept & fine_pass
            # Accumulate Fourier-row exception indicators by Parseval's RMS
            # norm. Prune conjugate pairs together until the cap is met.
            partner = torch.as_tensor((-np.arange(nx)) % nx, device=device)
            accept &= accept[partner]
            accept[0] = False
            accept[nx//2] = False
            selected_cpu = accept.cpu().numpy().copy()
            exception_cpu = exceptional_bound.detach().cpu().numpy()
            pair_costs = [(float(exception_cpu[j]**2+exception_cpu[-j]**2),j)
                          for j in range(1,nx//2) if selected_cpu[j]]
            total_proxy_squared = sum(cost for cost,j in pair_costs)
            before_pruning = np.sqrt(total_proxy_squared)
            pruned = 0
            for cost,j in sorted(pair_costs,reverse=True):
                if total_proxy_squared <= FINE_TOTAL_EXCEPTION_RMS_CAP**2:
                    break
                selected_cpu[j] = selected_cpu[-j] = False
                total_proxy_squared = max(0.,total_proxy_squared-cost)
                pruned += 1
            accept = torch.as_tensor(selected_cpu,dtype=torch.bool,device=device)
            info.update(validation_completed=True, strict_coarse_rows=strict_count,
                        validated_rows_before_pairing=int(accept.sum().item()),
                        fine_exception_macro_budget=FINE_EXCEPTION_MACRO_BUDGET,
                        fine_exception_rms_cap=FINE_TOTAL_EXCEPTION_RMS_CAP,
                        fine_exception_rms_before_pruning=float(before_pruning),
                        fine_exception_rms_after_pruning=float(np.sqrt(total_proxy_squared)),
                        fine_pruned_pairs=pruned,
                        maximum_exception_macro_bound=float(exceptional_bound.max().item()),
                        validation_times=[float(t) for t in validation_times],
                        fine_raw_defect_norm=float(fine_raw.sum().sqrt().item()),
                        fine_frame_defect_norm=float(fine_frame.sum().sqrt().item()))
    reverse = torch.as_tensor((-np.arange(nx)) % nx, device=device)
    if not info['validation_completed']:
        accept[:] = False
    accept &= accept[reverse]
    accept[0] = False
    accept[nx//2] = False
    mask = accept[:, None]
    accepted_rows = int(torch.sum(accept).item())
    info['accepted_rows'] = accepted_rows
    info['reason'] = 'accepted' if accepted_rows else 'all_rejected'
    if accepted_rows:
        selected = demodulated[:, accept]
        frame_coefficients = (transform.to(dtype)@selected.reshape(nt, -1)).reshape(nt, accepted_rows, nv)
        coefficients[:, accept] = frame_coefficients
    info['raw_defect_norm'] = float(torch.sum(raw_error).sqrt().item())
    info['frame_defect_norm'] = float(torch.sum(frame_error).sqrt().item())
    return coefficients, mask, info
