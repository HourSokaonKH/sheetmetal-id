"""
material_constants.py
=====================
Single loader for SGCC JIS G 3302 material constants.

All values live in material_constants.json so that scripts share one source
of truth.  Every constant has a hard-coded fallback so the loader is safe
even if the JSON is unreadable.

Usage
-----
    from material_constants import E_MPA, NU, Q_INF, B_ISO, R0, R45, R90
    from material_constants import SIGMA_RATIO, SIGMA_0, SIGMA_45, SIGMA_90

Python 2/3 compatible (the Abaqus kernel uses Python 2.7).
"""

from __future__ import print_function
import json as _json
import os as _os

_JSON_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           'material_constants.json')

# ── Fallback values (identical to the JSON baseline) ────────────────────────
_FALLBACK = {
    'elastic': {
        'E_MPa': 200000.0,
        'nu': 0.3,
        'density_tonne_per_mm3': 7.85e-9,
    },
    'voce_isotropic_0deg': {'Q_inf_MPa': 335.16, 'b': 3.95},
    'r_values_multizone_DIC': {'r0': 0.7120, 'r45': 0.7998, 'r90': 0.7420},
    'offset_yield_ratios_pooled': {
        'sigma_0_over_sigma_0':  1.0,
        'sigma_45_over_sigma_0': 0.7152506553271111,
        'sigma_90_over_sigma_0': 1.0820253315348436,
    },
    'offset_yield_pooled_MPa': {
        'sigma_0': 352.53, 'sigma_45': 252.17, 'sigma_90': 381.47,
    },
}

SOURCE = 'fallback'
_data = _FALLBACK
try:
    with open(_JSON_PATH, 'r') as _f:
        _data = _json.load(_f)
    SOURCE = _JSON_PATH
except (IOError, OSError, ValueError):
    _data = _FALLBACK

def _get(section, key):
    return _data.get(section, _FALLBACK[section]).get(key,
                                                      _FALLBACK[section][key])

# ── Elastic ────────────────────────────────────────────────────────────────
E_MPA   = float(_get('elastic', 'E_MPa'))
NU      = float(_get('elastic', 'nu'))
DENSITY = float(_get('elastic', 'density_tonne_per_mm3'))

# ── Voce isotropic hardening (0-deg reference fit) ─────────────────────────
Q_INF = float(_get('voce_isotropic_0deg', 'Q_inf_MPa'))
B_ISO = float(_get('voce_isotropic_0deg', 'b'))

# ── Plastic anisotropy: r-values from multi-zone DIC ───────────────────────
R0  = float(_get('r_values_multizone_DIC', 'r0'))
R45 = float(_get('r_values_multizone_DIC', 'r45'))
R90 = float(_get('r_values_multizone_DIC', 'r90'))

# ── 0.2% offset yield-stress ratios (pooled) ───────────────────────────────
SIGMA_RATIO = {
    0:  float(_get('offset_yield_ratios_pooled', 'sigma_0_over_sigma_0')),
    45: float(_get('offset_yield_ratios_pooled', 'sigma_45_over_sigma_0')),
    90: float(_get('offset_yield_ratios_pooled', 'sigma_90_over_sigma_0')),
}

# ── Pooled 0.2% offset yield stresses (MPa) ────────────────────────────────
SIGMA_0  = float(_get('offset_yield_pooled_MPa', 'sigma_0'))
SIGMA_45 = float(_get('offset_yield_pooled_MPa', 'sigma_45'))
SIGMA_90 = float(_get('offset_yield_pooled_MPa', 'sigma_90'))

__all__ = [
    'E_MPA', 'NU', 'DENSITY',
    'Q_INF', 'B_ISO',
    'R0', 'R45', 'R90',
    'SIGMA_RATIO', 'SIGMA_0', 'SIGMA_45', 'SIGMA_90',
    'SOURCE',
]

if __name__ == '__main__':
    print('Loaded from:', SOURCE)
    for k in __all__:
        if k != 'SOURCE':
            print('  %-13s = %s' % (k, globals()[k]))
