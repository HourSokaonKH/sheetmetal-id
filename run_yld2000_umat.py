#!/usr/bin/env python3
"""
Standalone Yld2000-2d UMAT runner for Abaqus/Standard.

This script is intended for a native x86_64 Windows or Linux Abaqus machine.
It reuses the existing UMAT/input-generation helpers from
optimize_hardening_multidir.py, but avoids entering the full optimization loop.

Typical usage on the Abaqus lab PC:

    abaqus python run_yld2000_umat.py --angles 0 45 90 --compare-exp

Windows convenience wrapper:

    run_yld2000_umat_lab_pc.cmd
"""

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import sys

import numpy as np

import optimize_hardening_multidir as md


WORK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(WORK_DIR, 'output')
DEFAULT_ALPHA_JSON = os.path.join(OUTPUT_DIR, 'yld2000_parameters.json')
DEFAULT_SUMMARY_TXT = os.path.join(OUTPUT_DIR, 'yld2000_umat_run_summary.txt')
DEFAULT_SUMMARY_JSON = os.path.join(OUTPUT_DIR, 'yld2000_umat_run_summary.json')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate and run standalone Yld2000-2d UMAT jobs in Abaqus.'
    )
    parser.add_argument(
        '--angles', nargs='+', type=int, default=[0, 45, 90],
        help='Specimen angles to run, in degrees. Default: 0 45 90.'
    )
    parser.add_argument(
        '--job-prefix', default='yld2000_umat',
        help='Prefix for Abaqus job names. Default: yld2000_umat.'
    )
    parser.add_argument(
        '--generate-only', action='store_true',
        help='Only write .inp files; do not launch Abaqus.'
    )
    parser.add_argument(
        '--compare-exp', action='store_true',
        help='Compare extracted curves against the experimental mean curves.'
    )
    parser.add_argument(
        '--cleanup-transients', action='store_true',
        help='Delete intermediate Abaqus files after a successful run but keep the ODB.'
    )
    parser.add_argument(
        '--keep-going', action='store_true',
        help='Continue to later angles if one job fails.'
    )
    parser.add_argument(
        '--timeout', type=int, default=600,
        help='Per-job timeout in seconds. Default: 600.'
    )
    parser.add_argument(
        '--displacement', type=float, default=md.DISPLACEMENT,
        help='Applied displacement in mm. Default: %.1f.' % md.DISPLACEMENT
    )
    parser.add_argument(
        '--sigma0', type=float, default=md.X0[0],
        help='Initial yield stress used in the UMAT props. Default: %.3f.' % md.X0[0]
    )
    parser.add_argument(
        '--c1', type=float, default=md.X0[1],
        help='First backstress modulus. Default: %.3f.' % md.X0[1]
    )
    parser.add_argument(
        '--gamma1', type=float, default=md.X0[2],
        help='First backstress rate. Default: %.3f.' % md.X0[2]
    )
    parser.add_argument(
        '--c2', type=float, default=md.X0[3],
        help='Second backstress modulus. Default: %.3f.' % md.X0[3]
    )
    parser.add_argument(
        '--gamma2', type=float, default=md.X0[4],
        help='Second backstress rate. Default: %.3f.' % md.X0[4]
    )
    parser.add_argument(
        '--q-inf', dest='q_inf', type=float, default=md.Q_INF,
        help='Voce saturation stress. Default: %.3f.' % md.Q_INF
    )
    parser.add_argument(
        '--b-iso', dest='b_iso', type=float, default=md.B_ISO,
        help='Voce hardening rate. Default: %.3f.' % md.B_ISO
    )
    parser.add_argument(
        '--alpha-json', default=DEFAULT_ALPHA_JSON,
        help='Optional JSON file with Yld2000 coefficients. Default: output/yld2000_parameters.json.'
    )
    return parser.parse_args()


def load_alpha_coefficients(json_path):
    if not json_path or not os.path.exists(json_path):
        return list(md.YLD2000_ALPHA), None

    with open(json_path, 'r') as f:
        data = json.load(f)

    coeffs = data.get('coefficients', {})
    alpha = [
        coeffs['alpha_1'],
        coeffs['alpha_2'],
        coeffs['alpha_3'],
        coeffs['alpha_4'],
        coeffs['alpha_5'],
        coeffs['alpha_6'],
        coeffs['alpha_7'],
        coeffs['alpha_8'],
    ]
    return alpha, json_path


def compute_nrmse(sim_curve, exp_curve):
    sim_strain, sim_stress = sim_curve
    exp_strain, exp_stress = exp_curve

    strain_min = max(float(sim_strain.min()), float(exp_strain.min()))
    strain_max = min(float(sim_strain.max()), float(exp_strain.max()))
    if strain_max <= strain_min + 0.01:
        return None

    common = np.linspace(strain_min, strain_max, 150)
    sim_interp = np.interp(common, sim_strain, sim_stress)
    exp_interp = np.interp(common, exp_strain, exp_stress)
    rmse = float(np.sqrt(np.mean((sim_interp - exp_interp) ** 2)))
    stress_range = float(exp_interp.max() - exp_interp.min())
    if stress_range <= 0.0:
        return None
    return rmse / stress_range


def write_curve_csv(job_name, curve):
    csv_path = os.path.join(OUTPUT_DIR, '%s_true_curve.csv' % job_name)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_strain', 'true_stress_mpa'])
        for strain_val, stress_val in zip(curve[0], curve[1]):
            writer.writerow(['%.10f' % strain_val, '%.10f' % stress_val])
    return csv_path


def write_summary(args, alpha_source, alpha_coeffs, jobs):
    summary = {
        'timestamp': dt.datetime.now().isoformat(timespec='seconds'),
        'runner': 'run_yld2000_umat.py',
        'umat_file': 'umat_yld2000.f',
        'generate_only': bool(args.generate_only),
        'compare_exp': bool(args.compare_exp),
        'cleanup_transients': bool(args.cleanup_transients),
        'angles': list(args.angles),
        'job_prefix': args.job_prefix,
        'timeout_s': int(args.timeout),
        'displacement_mm': float(args.displacement),
        'material': {
            'sigma0': float(args.sigma0),
            'q_inf': float(args.q_inf),
            'b_iso': float(args.b_iso),
            'c1': float(args.c1),
            'gamma1': float(args.gamma1),
            'c2': float(args.c2),
            'gamma2': float(args.gamma2),
            'alpha_source': alpha_source,
            'alpha': [float(val) for val in alpha_coeffs],
        },
        'jobs': jobs,
    }

    with open(DEFAULT_SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append('YLD2000 UMAT RUN SUMMARY')
    lines.append('=' * 72)
    lines.append('Generated: %s' % summary['timestamp'])
    lines.append('Runner: %s' % summary['runner'])
    lines.append('UMAT: %s' % summary['umat_file'])
    lines.append('Angles: %s' % ', '.join(str(val) for val in args.angles))
    lines.append('Generate only: %s' % ('yes' if args.generate_only else 'no'))
    lines.append('Compare to experimental mean: %s' % ('yes' if args.compare_exp else 'no'))
    lines.append('Cleanup transients: %s' % ('yes' if args.cleanup_transients else 'no'))
    lines.append('')
    lines.append('Material props used in *User Material:')
    lines.append('  E = %.4f MPa' % md.E_YOUNG)
    lines.append('  nu = %.6f' % md.NU)
    lines.append('  sigma0 = %.6f MPa' % args.sigma0)
    lines.append('  Q_inf = %.6f MPa' % args.q_inf)
    lines.append('  b_iso = %.6f' % args.b_iso)
    lines.append('  alpha source = %s' % (alpha_source or 'built-in constants'))
    for idx, val in enumerate(alpha_coeffs, start=1):
        lines.append('  alpha_%d = %.8f' % (idx, val))
    lines.append('')
    lines.append('Jobs:')
    for item in jobs:
        lines.append('  %s: %s (angle=%d)' % (
            item['job_name'], item['status'], item['angle_deg']))
        lines.append('    inp: %s' % item['inp_file'])
        if item.get('odb_file'):
            lines.append('    odb: %s' % item['odb_file'])
        if item.get('curve_csv'):
            lines.append('    curve: %s' % item['curve_csv'])
        if item.get('nrmse_vs_exp_mean') is not None:
            lines.append('    NRMSE vs experimental mean: %.6f' % item['nrmse_vs_exp_mean'])
        if item.get('note'):
            lines.append('    note: %s' % item['note'])

    with open(DEFAULT_SUMMARY_TXT, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    umat_path = os.path.join(WORK_DIR, 'umat_yld2000.f')
    if not os.path.exists(umat_path):
        print('ERROR: UMAT file not found: %s' % umat_path)
        return 1

    if not args.generate_only and shutil.which('abaqus') is None:
        print('ERROR: Abaqus command not found in PATH.')
        print('Run this on the lab PC from an Abaqus-enabled shell.')
        return 1

    alpha_coeffs, alpha_source = load_alpha_coefficients(args.alpha_json)
    md.MODEL_TYPE = 'yld2000'
    md.YLD2000_ALPHA = list(alpha_coeffs)
    md.Q_INF = float(args.q_inf)
    md.B_ISO = float(args.b_iso)
    md.DISPLACEMENT = float(args.displacement)

    exp_data = None
    if args.compare_exp:
        exp_data = md.load_experimental_data()

    if not args.generate_only and not md.check_yld2000_compiler_environment():
        return 1

    jobs = []
    overall_success = True

    print('=' * 72)
    print('STANDALONE YLD2000 UMAT RUNNER')
    print('=' * 72)
    print('Angles: %s' % ', '.join(str(val) for val in args.angles))
    print('Job prefix: %s' % args.job_prefix)
    print('Generate only: %s' % ('yes' if args.generate_only else 'no'))
    print('Alpha source: %s' % (alpha_source or 'built-in constants'))
    print('')

    for angle in args.angles:
        job_name = '%s_%02d' % (args.job_prefix, angle)
        print('Preparing %s...' % job_name)
        inp_path = md.generate_inp(
            job_name,
            float(args.sigma0),
            float(args.c1),
            float(args.gamma1),
            float(args.c2),
            float(args.gamma2),
            int(angle),
        )

        job_info = {
            'job_name': job_name,
            'angle_deg': int(angle),
            'inp_file': os.path.relpath(inp_path, WORK_DIR),
            'status': 'generated' if args.generate_only else 'pending',
            'command': 'abaqus job=%s user=umat_yld2000.f interactive ask_delete=OFF' % job_name,
        }

        if args.generate_only:
            print('  Generated %s' % inp_path)
            jobs.append(job_info)
            continue

        success = md.run_abaqus_job(job_name, timeout=int(args.timeout))
        job_info['status'] = 'success' if success else 'failed'
        odb_path = job_name + '.odb'
        if os.path.exists(odb_path):
            job_info['odb_file'] = os.path.relpath(odb_path, WORK_DIR)

        if success and md.HAS_ODB:
            curve = md.extract_results_odb(job_name)
            if curve is not None:
                curve_path = write_curve_csv(job_name, curve)
                job_info['curve_csv'] = os.path.relpath(curve_path, WORK_DIR)
                if exp_data is not None and angle in exp_data:
                    nrmse = compute_nrmse(curve, exp_data[angle])
                    job_info['nrmse_vs_exp_mean'] = nrmse
                    if nrmse is not None:
                        print('  %2d-deg NRMSE vs experimental mean = %.6f' % (angle, nrmse))
            else:
                job_info['note'] = 'ODB opened, but no valid true stress-strain curve was extracted.'
        elif success:
            job_info['note'] = 'Job completed, but odbAccess is unavailable in this Python session.'

        if success and args.cleanup_transients:
            md.cleanup_job(job_name, keep_odb=True)

        jobs.append(job_info)

        if not success:
            overall_success = False
            if not args.keep_going:
                print('Stopping after failure in %s.' % job_name)
                break

    write_summary(args, alpha_source, alpha_coeffs, jobs)

    print('')
    print('Summary written to: %s' % DEFAULT_SUMMARY_TXT)
    print('JSON summary written to: %s' % DEFAULT_SUMMARY_JSON)
    if args.generate_only:
        print('Input decks generated successfully.')
        return 0

    return 0 if overall_success else 1


if __name__ == '__main__':
    sys.exit(main())