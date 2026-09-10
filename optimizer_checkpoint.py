"""Round-boundary checkpoints for the standard bounded Nelder--Mead search.

The entire sorted simplex and its scores are saved, so continuing requires no
repeat scoring and preserves reflection, expansion, contraction and shrink
choices. The nonadaptive coefficients and bounds handling match scipy.optimize
minimize(method='Nelder-Mead'), used by the previous GisAPR driver.
"""
import os
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeResult


def atomic_npz(filename, **arrays):
    """Keep the previous complete checkpoint intact until the new save closes."""
    filename = Path(filename)
    temporary = filename.with_name(filename.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, filename)


def checkpoint_nelder_mead(objective, x0, bounds, options, *, resume_file=None,
                           additional_iterations=20,
                           checkpoint_prefix='my_simplex_state_round_'):
    """Run or resume the driver's standard nonadaptive Nelder--Mead search.

    nit follows SciPy: the evaluated initial simplex is iteration 1. Checkpoints
    are written at initialization and after each completed simplex update.
    A continuation adds iterations to its saved nit and keeps saved tolerances,
    bounds, evaluations, all simplex vertices, scores, and best-point history.
    """
    if resume_file:
        with np.load(resume_file, allow_pickle=False) as data:
            if str(data['optimizer'].item()) != 'GisAPR_Nelder_Mead_v1':
                raise ValueError('Not a GisAPR simplex checkpoint')
            simplex = data['simplex'].copy()
            values = data['values'].copy()
            lower, upper = data['lower'].copy(), data['upper'].copy()
            nit, nfev = int(data['nit']), int(data['nfev'])
            xatol, fatol = float(data['xatol']), float(data['fatol'])
            history = [row.copy() for row in data['history']]
        if len(np.asarray(x0)) != simplex.shape[1]:
            raise ValueError('Checkpoint dimensions do not match --Simplex_dimensions')
        if additional_iterations < 0:
            raise ValueError('Additional simplex iterations must be nonnegative')
        maxiter = nit + int(additional_iterations)
    else:
        x0 = np.asarray(x0, dtype=float).ravel()
        lower = np.broadcast_to(bounds.lb, x0.shape).copy()
        upper = np.broadcast_to(bounds.ub, x0.shape).copy()
        if np.any(lower > upper):
            raise ValueError('Simplex lower bounds must not exceed upper bounds')
        x0 = np.clip(x0, lower, upper)
        if options.get('initial_simplex') is not None:
            simplex = np.asarray(options['initial_simplex'], dtype=float).copy()
        else:
            simplex = np.tile(x0, (len(x0) + 1, 1))
            for dimension in range(len(x0)):
                simplex[dimension + 1, dimension] = (1.05 * x0[dimension]
                    if x0[dimension] != 0 else 0.00025)
        if simplex.shape != (len(x0) + 1, len(x0)):
            raise ValueError('initial_simplex must have shape (N+1, N)')
        history = [simplex[0].copy()]
        simplex = np.where(simplex > upper, 2 * upper - simplex, simplex)
        simplex = np.clip(simplex, lower, upper)
        values = np.array([float(objective(vertex)) for vertex in simplex])
        nfev = len(simplex)
        # Preserve SciPy's initial ordering, including its second stable state
        # sort when equal objective scores occur.
        for _ in range(2):
            order = np.argsort(values)
            simplex, values = simplex[order], values[order]
        nit = 1
        xatol, fatol = options.get('xatol', 1e-4), options.get('fatol', 1e-4)
        maxiter = int(options.get('maxiter', len(x0) * 200))

    def evaluate(point):
        nonlocal nfev
        value = float(objective(point))
        nfev += 1
        return value

    def save():
        if checkpoint_prefix is not None:
            atomic_npz(f'{checkpoint_prefix}{nit}.npz',
                       optimizer=np.array('GisAPR_Nelder_Mead_v1'),
                       simplex=simplex, values=values, lower=lower, upper=upper,
                       nit=np.array(nit), nfev=np.array(nfev),
                       xatol=np.array(xatol), fatol=np.array(fatol),
                       maxiter=np.array(maxiter), history=np.asarray(history))

    def converged():
        return (np.max(np.abs(simplex[1:] - simplex[0])) <= xatol
                and np.max(np.abs(values[1:] - values[0])) <= fatol)

    if not resume_file:
        save()
    while nit < maxiter and not converged():
        centroid = np.add.reduce(simplex[:-1], axis=0) / simplex.shape[1]
        reflected = np.clip(2 * centroid - simplex[-1], lower, upper)
        reflected_value = evaluate(reflected)
        shrink = False
        if reflected_value < values[0]:
            expanded = np.clip(3 * centroid - 2 * simplex[-1], lower, upper)
            expanded_value = evaluate(expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
        elif reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
        elif reflected_value < values[-1]:
            contracted = np.clip(1.5 * centroid - 0.5 * simplex[-1], lower, upper)
            contracted_value = evaluate(contracted)
            if contracted_value <= reflected_value:
                simplex[-1], values[-1] = contracted, contracted_value
            else:
                shrink = True
        else:
            contracted = np.clip(0.5 * centroid + 0.5 * simplex[-1], lower, upper)
            contracted_value = evaluate(contracted)
            if contracted_value < values[-1]:
                simplex[-1], values[-1] = contracted, contracted_value
            else:
                shrink = True
        if shrink:
            for vertex in range(1, len(simplex)):
                simplex[vertex] = np.clip(simplex[0] + 0.5 * (simplex[vertex] - simplex[0]), lower, upper)
                values[vertex] = evaluate(simplex[vertex])
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        nit += 1
        history.append(simplex[0].copy())
        save()
    success = converged() and nit < maxiter
    message = 'Optimization converged.' if success else 'Maximum iterations reached.'
    result = OptimizeResult(x=simplex[0].copy(), fun=float(values[0]), nit=nit,
                            nfev=nfev, success=success, status=0 if success else 2,
                            message=message,
                            final_simplex=(simplex.copy(), values.copy()))
    if options.get('return_all'):
        result['allvecs'] = history
    if options.get('disp'):
        print(message, 'Iterations:', nit, 'Evaluations:', nfev)
    return result
