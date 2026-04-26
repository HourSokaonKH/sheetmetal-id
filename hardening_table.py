"""
hardening_table.py
==================
Build the tabulated flow curve sigma_y(kappa) used by umat_yld2000_table.f.

The UMAT cutting-plane return mapping only needs a monotonically
non-decreasing scalar function sigma_y(kappa).  Any uniaxial isotropic
hardening law can be baked into the table; for the SGCC FEA identification
we use the monotonic equivalent of Voce + Chaboche(2):

    sigma_y(kappa) = sigma_0
                   + Q_inf  * (1 - exp(-b    * kappa))
                   + (C1/g1)* (1 - exp(-g1   * kappa))
                   + (C2/g2)* (1 - exp(-g2   * kappa))

This version makes (sigma_0, C1, g1, C2, g2) genuine DOFs of the FE cost
function: each Nelder-Mead trial generates a fresh table, which is written
into PROPS and reaches the UMAT unchanged.

The table must cover the plastic-strain range actually seen in the model.
Default is 0 .. eps_max=0.40 at 50 points.  The UMAT clamps linearly
beyond the last abscissa, so a reasonable margin protects the return
mapping from extrapolation surprises.

Usage
-----
    from hardening_table import build_flow_curve
    kappa, sigma_y = build_flow_curve(
        sigma_0=312.0, Q_inf=335.16, b=3.95,
        C1=502.7, gamma1=499.7, C2=100.4, gamma2=199.4,
        n_points=50, eps_max=0.40,
    )
    props = build_umat_props(E=200000.0, nu=0.3, alpha=[...8 floats...],
                             kappa=kappa, sigma_y=sigma_y)
    # props now has length 11 + 2*n_points = 111 for n_points=50
"""
from __future__ import print_function
import numpy as np


def build_flow_curve(sigma_0, Q_inf, b, C1, gamma1, C2, gamma2,
                     n_points=50, eps_max=0.40):
    """Return (kappa, sigma_y) arrays tabulating the Voce+Chaboche(2)
    monotonic flow curve at n_points abscissas on [0, eps_max].

    Parameters
    ----------
    sigma_0 : float            initial yield stress, MPa
    Q_inf, b : float           Voce amplitude (MPa) and rate
    C1, gamma1 : float         first Chaboche branch (MPa, -)
    C2, gamma2 : float         second Chaboche branch (MPa, -)
    n_points : int             number of tabulated points (>= 2)
    eps_max : float            upper abscissa (plastic strain)

    Notes
    -----
    * kappa[0] = 0 so the table covers the elastic-plastic transition.
    * If any gamma_i is tiny (numerical noise) we fall back to the
      infinitesimal limit C_i*kappa so the curve is still finite.
    """
    if n_points < 2:
        raise ValueError('n_points must be >= 2')

    kappa = np.linspace(0.0, eps_max, n_points)

    def _bs(C, g, eps):
        if abs(g) < 1e-12:
            return C * eps
        return (C / g) * (1.0 - np.exp(-g * eps))

    sigma_y = (sigma_0
               + Q_inf  * (1.0 - np.exp(-b      * kappa))
               + _bs(C1, gamma1, kappa)
               + _bs(C2, gamma2, kappa))
    return kappa, sigma_y


def build_umat_props(E, nu, alpha, kappa, sigma_y):
    """Assemble the UMAT PROPS array for umat_yld2000_table.f.

    Layout: [E, nu, alpha(1..8), NTAB, kappa(1..NTAB), sigma_y(1..NTAB)]
    Length: 11 + 2*NTAB.
    """
    alpha = list(alpha)
    if len(alpha) != 8:
        raise ValueError('alpha must have 8 entries')
    kappa = np.asarray(kappa, dtype=float)
    sigma_y = np.asarray(sigma_y, dtype=float)
    if kappa.shape != sigma_y.shape or kappa.ndim != 1:
        raise ValueError('kappa and sigma_y must be 1-D of equal length')
    if kappa.size < 2:
        raise ValueError('need >= 2 tabulated points')
    if np.any(np.diff(kappa) <= 0.0):
        raise ValueError('kappa must be strictly ascending')

    ntab = int(kappa.size)
    props = ([float(E), float(nu)]
             + [float(a) for a in alpha]
             + [float(ntab)]
             + kappa.tolist()
             + sigma_y.tolist())
    assert len(props) == 11 + 2 * ntab
    return props


def format_user_material_block(props, name='YLD2KTAB', nstatv=1):
    """Format the PROPS array as Abaqus *USER MATERIAL keyword lines.

    Abaqus accepts up to 8 values per line under *USER MATERIAL.
    """
    n = len(props)
    lines = []
    lines.append('*MATERIAL, NAME=%s' % name)
    lines.append('*USER MATERIAL, CONSTANTS=%d' % n)
    for i in range(0, n, 8):
        chunk = props[i:i + 8]
        lines.append(', '.join('%.8E' % v for v in chunk))
    lines.append('*DEPVAR')
    lines.append(' %d' % nstatv)
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Reproduce the JMPT-paper surrogate values to check assembly.
    from material_constants import E_MPA, NU, Q_INF, B_ISO  # noqa: F401

    kappa, sigma_y = build_flow_curve(
        sigma_0=312.35, Q_inf=Q_INF, b=B_ISO,
        C1=502.71, gamma1=499.72, C2=100.37, gamma2=199.44,
        n_points=50, eps_max=0.40,
    )
    print('kappa range : [%.4f, %.4f]' % (kappa[0], kappa[-1]))
    print('sigma_y(0)  : %.3f MPa' % sigma_y[0])
    print('sigma_y(max): %.3f MPa' % sigma_y[-1])

    ALPHA = [2.78216474, 1.99615102, 3.86202580, 2.58000953,
             2.87101512, 4.79371206, 1.18982067, 4.25577549]
    props = build_umat_props(E=E_MPA, nu=NU, alpha=ALPHA,
                             kappa=kappa, sigma_y=sigma_y)
    print('NPROPS      : %d' % len(props))
    assert len(props) == 111

    text = format_user_material_block(props)
    print('---- *USER MATERIAL block preview (first 6 lines) ----')
    for line in text.splitlines()[:6]:
        print(line)
    print('... (%d lines total)' % len(text.splitlines()))
