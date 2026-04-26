#!/usr/bin/env python3
"""
Canonical anisotropy helpers shared across the repo.

The MATLAB multi-zone extraction scripts are the authoritative source for the
DIC-based anisotropy values because they apply the full zone-quality workflow
(R^2 threshold, CV(Eyy) threshold, weighted pooling, and IQR rejection).
Python-side CSV recomputation is retained only as a diagnostic fallback.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.io import loadmat


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ANISOTROPY_RESULT_FILE = 'anisotropy_results.mat'

_DIRECTION_RESULT_FILES = {
    '00': 'r0_result.mat',
    '45': 'r45_result.mat',
    '90': 'r90_result.mat',
}

_DIRECTION_RESULT_KEYS = {
    '00': 'r0_final',
    '45': 'r45_final',
    '90': 'r90_final',
}


def _to_scalar(value):
    return float(np.asarray(value).squeeze())


def compute_weighted_direction_value(specimen_values, weights):
    """
    Apply weighted averaging with IQR-based outlier rejection.

    Returns the weighted value and a boolean mask with the same length as the
    input arrays indicating which specimen values were retained.
    """
    values = np.asarray(specimen_values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid_mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    inlier_mask = np.zeros(values.shape, dtype=bool)
    if not np.any(valid_mask):
        return np.nan, inlier_mask

    valid_values = values[valid_mask]
    valid_weights = weights[valid_mask]
    valid_inliers = np.ones(valid_values.shape, dtype=bool)

    if valid_values.size >= 3:
        q1, q3 = np.quantile(valid_values, [0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            valid_inliers = (valid_values >= lower) & (valid_values <= upper)
            if not np.any(valid_inliers):
                valid_inliers = np.ones(valid_values.shape, dtype=bool)

    valid_indices = np.where(valid_mask)[0]
    inlier_mask[valid_indices[valid_inliers]] = True
    weighted_value = np.average(valid_values[valid_inliers], weights=valid_weights[valid_inliers])
    return float(weighted_value), inlier_mask


def load_direction_result(direction, base_dir=DATA_DIR):
    """Load one directional MATLAB result file if it exists."""
    if direction not in _DIRECTION_RESULT_FILES:
        raise ValueError('direction must be one of 00, 45, 90')

    path = os.path.join(base_dir, _DIRECTION_RESULT_FILES[direction])
    if not os.path.exists(path):
        return None

    mat = loadmat(path)
    summary = np.asarray(mat.get('r_summary', np.empty((0, 4))), dtype=float)
    if summary.ndim == 1 and summary.size:
        summary = summary.reshape(1, -1)
    if summary.size:
        sort_idx = np.argsort(summary[:, 0])
        summary = summary[sort_idx]

    specimen_summary = []
    for row in summary:
        specimen_summary.append({
            'specimen': f"{int(row[0]):02d}",
            'r': float(row[1]),
            'r_std': float(row[2]),
            'n_good': int(round(row[3])),
        })

    specimen_means = np.array([item['r'] for item in specimen_summary], dtype=float)
    specimen_weights = np.array([item['n_good'] for item in specimen_summary], dtype=float)
    weighted_mean, inlier_mask = compute_weighted_direction_value(specimen_means, specimen_weights)

    return {
        'direction': direction,
        'path': path,
        'method': 'matlab_multizone_summary',
        'r_final': _to_scalar(mat[_DIRECTION_RESULT_KEYS[direction]]),
        'r_weighted': _to_scalar(mat.get(_DIRECTION_RESULT_KEYS[direction].replace('_final', '_weighted'), [[np.nan]])),
        'r_robust': _to_scalar(mat.get(_DIRECTION_RESULT_KEYS[direction].replace('_final', '_robust'), [[np.nan]])),
        'specimen_summary': specimen_summary,
        'specimen_std': float(np.std(specimen_means, ddof=1)) if specimen_means.size > 1 else 0.0,
        'recomputed_weighted_mean': weighted_mean,
        'inlier_mask': inlier_mask.tolist(),
    }


def load_canonical_anisotropy(base_dir=DATA_DIR):
    """Load the canonical anisotropy reference from MATLAB outputs."""
    direction_results = {}
    for direction in ['00', '45', '90']:
        result = load_direction_result(direction, base_dir=base_dir)
        if result is None:
            return {
                'available': False,
                'description': 'MATLAB anisotropy result files are missing; using Python fallback.',
                'direction_results': {},
                'source_files': [],
            }
        direction_results[direction] = result

    anisotropy_path = os.path.join(base_dir, ANISOTROPY_RESULT_FILE)
    if os.path.exists(anisotropy_path):
        anisotropy_mat = loadmat(anisotropy_path)
        r0 = _to_scalar(anisotropy_mat['r0'])
        r45 = _to_scalar(anisotropy_mat['r45'])
        r90 = _to_scalar(anisotropy_mat['r90'])
        r_bar = _to_scalar(anisotropy_mat['r_bar'])
        delta_r = _to_scalar(anisotropy_mat['delta_r'])
    else:
        r0 = direction_results['00']['r_final']
        r45 = direction_results['45']['r_final']
        r90 = direction_results['90']['r_final']
        r_bar = (r0 + 2.0 * r45 + r90) / 4.0
        delta_r = (r0 - 2.0 * r45 + r90) / 2.0

    source_files = [direction_results[d]['path'] for d in ['00', '45', '90']]
    if os.path.exists(anisotropy_path):
        source_files.append(anisotropy_path)

    return {
        'available': True,
        'description': (
            'MATLAB multi-zone result files (.mat) are authoritative for anisotropy; '
            'Python CSV recomputation is diagnostic only.'
        ),
        'source_files': source_files,
        'direction_results': direction_results,
        'r_values': {'00': r0, '45': r45, '90': r90},
        'r0': r0,
        'r45': r45,
        'r90': r90,
        'r_bar': r_bar,
        'delta_r': delta_r,
    }


def sample_direction_r_value(direction_result, rng, relative_noise=0.0):
    """Sample one directional r-value around the canonical MATLAB baseline."""
    specimens = direction_result.get('specimen_summary', [])
    if not specimens:
        return float(direction_result['r_final'])

    sampled_values = []
    sampled_weights = []
    for specimen in specimens:
        sigma = specimen['r_std']
        if sigma <= 0:
            sigma = max(0.01 * specimen['r'], 1e-4)
        sampled = rng.normal(specimen['r'], sigma)
        if relative_noise > 0:
            sampled *= 1.0 + rng.normal(0.0, relative_noise)
        sampled_values.append(max(sampled, 1e-6))
        sampled_weights.append(max(specimen['n_good'], 1))

    sampled_mean, _ = compute_weighted_direction_value(sampled_values, sampled_weights)
    if not np.isfinite(sampled_mean):
        return float(direction_result['r_final'])
    return float(max(sampled_mean, 1e-6))