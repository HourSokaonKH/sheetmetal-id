#!/usr/bin/env python3
"""
=============================================================================
Mesh Convergence Study — Generate .inp Files for 3 Mesh Sizes
=============================================================================
Generates Abaqus .inp files with mesh sizes: 0.5, 1.0, 2.0 mm
All other parameters (material, BCs, loading) remain identical.

Transfer all 3 .inp files to the Abaqus machine and run:
  abaqus job=Tensile_mesh_0p5 interactive
  abaqus job=Tensile_mesh_1p0 interactive
  abaqus job=Tensile_mesh_2p0 interactive

Then run:  abaqus python mesh_convergence_extract.py
=============================================================================
"""

import numpy as np
import os

# ============================================================================
# FIXED MODEL PARAMETERS (identical across all mesh sizes)
# ============================================================================
WIDTH = 10.0        # half gauge width (mm)
LENGTH = 40.0       # half gauge length (mm)
THICKNESS = 1.5     # sheet thickness (mm)

E_YOUNG = 200000.0  # MPa
NU_POISSON = 0.3
DENSITY = 7.85e-9   # tonne/mm³

# Hill'48 R-values
R11, R22, R33, R12, R13, R23 = 1.0, 1.0119, 0.9347, 1.0041, 1.0, 1.0

# Combined hardening (original parameters)
SIGMA0 = 312.35
Q_INF = 335.16
B_ISO = 3.95
C1, GAMMA1 = 502.71, 499.72
C2, GAMMA2 = 100.37, 199.44

DISPLACEMENT = 25.0  # mm

# Mesh sizes to study
MESH_SIZES = [0.5, 1.0, 2.0]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def monotonic_hardening_table(sigma0, Q_inf, b, C_list, gamma_list,
                              max_strain=0.5, n_pts=50):
    eps_p = np.linspace(0, max_strain, n_pts)
    sigma = sigma0 + Q_inf * (1.0 - np.exp(-b * eps_p))
    for C_k, g_k in zip(C_list, gamma_list):
        sigma += (C_k / g_k) * (1.0 - np.exp(-g_k * eps_p))
    return list(zip(sigma, eps_p))


def write_node_list(f, node_ids, per_line=16):
    for i, nid in enumerate(node_ids):
        if i > 0 and i % per_line == 0:
            f.write('\n')
        elif i > 0:
            f.write(', ')
        f.write(f'{nid}')
    f.write('\n')


def generate_inp(mesh_size, job_name):
    """Generate a complete .inp file for a given mesh size."""
    nx = int(round(WIDTH / mesh_size))
    ny = int(round(LENGTH / mesh_size))
    dx = WIDTH / nx
    dy = LENGTH / ny

    # Nodes
    nodes = []
    nid = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append((nid, i * dx, j * dy))
            nid += 1

    # Elements (CPS4R)
    elements = []
    eid = 1
    n_per_row = nx + 1
    for j in range(ny):
        for i in range(nx):
            n1 = j * n_per_row + i + 1
            n2 = n1 + 1
            n3 = n2 + n_per_row
            n4 = n1 + n_per_row
            elements.append((eid, n1, n2, n3, n4))
            eid += 1

    # Node sets
    sym_x = [n[0] for n in nodes if abs(n[1]) < 1e-10]
    sym_y = [n[0] for n in nodes if abs(n[2]) < 1e-10]
    load_nodes = [n[0] for n in nodes if abs(n[2] - LENGTH) < 1e-10]
    corner = [n[0] for n in nodes
              if abs(n[1]) < 1e-10 and abs(n[2] - LENGTH) < 1e-10]

    # Hardening table
    hardening = monotonic_hardening_table(
        SIGMA0, Q_INF, B_ISO, [C1, C2], [GAMMA1, GAMMA2])

    filepath = os.path.join(OUTPUT_DIR, f'{job_name}.inp')
    with open(filepath, 'w') as f:
        f.write('*Heading\n')
        f.write(f'** Mesh Convergence Study — mesh size = {mesh_size} mm\n')
        f.write(f'** Mesh: {nx}x{ny} CPS4R elements ({nx*ny} total)\n')
        f.write(f'** Nodes: {len(nodes)}\n')
        f.write('**\n')
        f.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
        f.write('**\n')

        # Part
        f.write('*Part, name=SPECIMEN\n')
        f.write('*Node\n')
        for nid, x, y in nodes:
            f.write(f'{nid}, {x:.6f}, {y:.6f}\n')
        f.write('**\n')

        f.write('*Element, type=CPS4R\n')
        for eid, n1, n2, n3, n4 in elements:
            f.write(f'{eid}, {n1}, {n2}, {n3}, {n4}\n')
        f.write('**\n')

        f.write('*Elset, elset=ALL_ELEMENTS, generate\n')
        f.write(f'1, {len(elements)}, 1\n')
        f.write('**\n')

        f.write('*Nset, nset=SYM_X\n')
        write_node_list(f, sym_x)
        f.write('*Nset, nset=SYM_Y\n')
        write_node_list(f, sym_y)
        f.write('*Nset, nset=LOAD_NODES\n')
        write_node_list(f, load_nodes)
        if corner:
            f.write('*Nset, nset=CORNER\n')
            f.write(f'{corner[0]}\n')
        f.write('**\n')

        # Orientation
        f.write('*Orientation, name=ORI_ROLL, system=RECTANGULAR\n')
        f.write('1.0, 0.0, 0.0, 0.0, 1.0, 0.0\n')
        f.write('1, 0.0\n')
        f.write('**\n')

        # Section
        f.write('*Solid Section, elset=ALL_ELEMENTS, material=SGCC_STEEL, '
                'orientation=ORI_ROLL\n')
        f.write(f'{THICKNESS},\n')
        f.write('*End Part\n')
        f.write('**\n')

        # Assembly
        f.write('*Assembly, name=Assembly\n')
        f.write('*Instance, name=SPECIMEN-1, part=SPECIMEN\n')
        f.write('*End Instance\n')
        f.write('**\n')
        f.write('*Nset, nset=SYM_X, instance=SPECIMEN-1\n')
        write_node_list(f, sym_x)
        f.write('*Nset, nset=SYM_Y, instance=SPECIMEN-1\n')
        write_node_list(f, sym_y)
        f.write('*Nset, nset=LOAD_NODES, instance=SPECIMEN-1\n')
        write_node_list(f, load_nodes)
        if corner:
            f.write('*Nset, nset=CORNER, instance=SPECIMEN-1\n')
            f.write(f'{corner[0]}\n')
        f.write('*End Assembly\n')
        f.write('**\n')

        # Material
        f.write('*Material, name=SGCC_STEEL\n')
        f.write(f'*Density\n{DENSITY},\n')
        f.write(f'*Elastic\n{E_YOUNG}, {NU_POISSON}\n')
        f.write('**\n')
        f.write('*Plastic\n')
        for sigma, eps_p in hardening:
            f.write(f'{sigma:.4f}, {eps_p:.6f}\n')
        f.write('**\n')
        f.write('*Potential\n')
        f.write(f'{R11}, {R22}, {R33}, {R12}, {R13}, {R23}\n')
        f.write('**\n')

        # Step
        f.write('*Step, name=Tensile, nlgeom=YES, inc=1000\n')
        f.write('*Static\n')
        f.write('0.01, 1.0, 1e-08, 0.02\n')
        f.write('**\n')

        # BCs
        f.write('*Boundary\nSYM_X, 1, 1, 0.0\n')
        f.write('*Boundary\nSYM_Y, 2, 2, 0.0\n')
        f.write(f'*Boundary\nLOAD_NODES, 2, 2, {DISPLACEMENT}\n')
        f.write('**\n')

        # Output
        f.write('*Output, field, frequency=1\n')
        f.write('*Node Output\nU, RF\n')
        f.write('*Element Output, directions=YES\nS, E, PE, PEEQ\n')
        f.write('**\n')
        f.write('*Output, history, frequency=1\n')
        f.write('*Node Output, nset=CORNER\nU2, RF2\n')
        f.write('**\n')
        f.write('*End Step\n')

    print(f"  {job_name}.inp  —  {nx}x{ny} = {nx*ny} elements, "
          f"{len(nodes)} nodes, mesh={mesh_size}mm")
    return filepath


def main():
    print("Mesh Convergence Study — Generating .inp files\n")
    print(f"  Geometry: {WIDTH}x{LENGTH} mm quarter model")
    print(f"  Mesh sizes: {MESH_SIZES} mm\n")

    jobs = []
    for ms in MESH_SIZES:
        tag = f"{ms:.1f}".replace('.', 'p')
        job_name = f"Tensile_mesh_{tag}"
        generate_inp(ms, job_name)
        jobs.append(job_name)

    print(f"\nTransfer these files to the Abaqus machine and run:")
    for j in jobs:
        print(f"  abaqus job={j} interactive")
    print(f"\nThen run:  abaqus python mesh_convergence_extract.py")


if __name__ == '__main__':
    main()
