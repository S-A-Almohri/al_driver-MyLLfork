#!/usr/bin/env python
"""
Compute-node entry point for D-optimality maxvol + pseudo-inverse.

Builds A_atomic from the training design matrix, drops exact-zero columns
(energy offsets in force-only A), runs plain maxvol, and writes a square
inverse matching FITENER=false candidate descriptor width:
  - A_atomic.txt
  - kept_columns.txt
  - inverse_A_subset.npy   (n_kept x n_kept)
  - maxvol_pivots.txt
  - dopt_maxvol.log

No SVD truncation: that would leave the reference in a projected basis
while candidates stay in the full force-feature space. Zero-drop alone keeps
both sides aligned by column index.
"""

from __future__ import print_function

import argparse
import gc
import os
import sys

import numpy as np

# Allow "import gen_selections" / "import helpers" when launched from any CWD.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import gen_selections  # noqa: E402


def run_maxvol(amat_path, fm_setup, traj_list_path, component, rcond, outdir):
    """
    Stream-build A_atomic, drop null columns, maxvol, write square inverse.

    Returns
    -------
    int
        Number of pivot rows selected.
    """
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    log_path = os.path.join(outdir, "dopt_maxvol.log")
    log = open(log_path, "w")

    def say(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    say("run_dopt_maxvol: amat_path={}".format(amat_path))
    say("run_dopt_maxvol: fm_setup={}".format(fm_setup))
    say("run_dopt_maxvol: component={}, rcond={}".format(component, rcond))
    say("run_dopt_maxvol: mode=zero-column-drop + plain maxvol (no SVD)")

    A_atomic_ref, n_feat = gen_selections._build_a_atomic_from_file(
        amat_path,
        fm_setup_path=fm_setup if fm_setup else None,
        traj_list_path=traj_list_path if traj_list_path else None,
        label="reference",
        component=component)

    say("run_dopt_maxvol: A_atomic_ref shape: {}".format(A_atomic_ref.shape))
    np.savetxt(os.path.join(outdir, "A_atomic.txt"), A_atomic_ref)

    try:
        from maxvolpy.maxvol import maxvol
    except ImportError:
        say("ERROR: maxvolpy not found. Install with: pip install maxvolpy")
        log.close()
        sys.exit(1)

    n_rows_atomic, n_cols_atomic = A_atomic_ref.shape
    if n_rows_atomic < n_cols_atomic:
        say("ERROR: A_atomic_ref is fat ({} rows < {} cols). "
            "maxvol requires a tall matrix.".format(n_rows_atomic, n_cols_atomic))
        log.close()
        sys.exit(1)

    # Drop exact-zero columns so A_work matches FITENER=false candidate width.
    col_norm = np.linalg.norm(A_atomic_ref, axis=0)
    keep = col_norm >= 1.0e-14
    drop_cols = np.where(~keep)[0]
    keep_cols = np.where(keep)[0]
    if len(drop_cols) > 0:
        say("run_dopt_maxvol: dropping {} exact-zero columns "
            "(typical: energy offsets): {}".format(
                len(drop_cols), drop_cols.tolist()))
    else:
        say("run_dopt_maxvol: no exact-zero columns to drop.")

    A_work = np.ascontiguousarray(A_atomic_ref[:, keep])
    del A_atomic_ref
    gc.collect()

    n_kept_cols = A_work.shape[1]
    say("run_dopt_maxvol: A_work shape after column drop: {}".format(A_work.shape))
    np.savetxt(os.path.join(outdir, "kept_columns.txt"),
               keep_cols, fmt="%d")

    if n_rows_atomic < n_kept_cols:
        say("ERROR: A_work is fat after column drop ({} rows < {} cols)."
            .format(n_rows_atomic, n_kept_cols))
        log.close()
        sys.exit(1)

    ref_rank = np.linalg.matrix_rank(A_work)
    if ref_rank < n_kept_cols:
        say("ERROR: A_work is still rank deficient after zero-drop "
            "(rank {} < {} cols). Cannot form a nonsingular square maxvol "
            "block while keeping descriptor column alignment. Inspect A_work "
            "or relax the potential / remove redundant features."
            .format(ref_rank, n_kept_cols))
        log.close()
        sys.exit(1)

    say("run_dopt_maxvol: A_work full column rank "
        "({} == {}).".format(ref_rank, n_kept_cols))

    say("run_dopt_maxvol: running maxvol on A_work ({})...".format(A_work.shape))
    try:
        piv, _ = maxvol(A_work, 1.05)
    except ValueError as exc:
        say("ERROR: maxvol failed: {}".format(exc))
        log.close()
        sys.exit(1)

    piv = np.asarray(piv, dtype=int)
    a_subset = A_work[piv]
    say("run_dopt_maxvol: pivot block shape: {}".format(a_subset.shape))
    if a_subset.shape[0] != a_subset.shape[1]:
        say("ERROR: expected square D-optimal block, got {}.".format(
            a_subset.shape))
        log.close()
        sys.exit(1)

    cond_subset = np.linalg.cond(a_subset)
    say("run_dopt_maxvol: D-optimal submatrix condition number = {:.3e}".format(
        cond_subset))
    if not np.isfinite(cond_subset) or cond_subset > 1.0e14:
        say("WARNING: D-optimal submatrix is near-singular "
            "(cond = {:.3e}).".format(cond_subset))
    elif cond_subset > 1.0e10:
        say("WARNING: D-optimal submatrix is ill-conditioned "
            "(cond = {:.3e}).".format(cond_subset))

    try:
        inverse_a_subset = np.linalg.pinv(a_subset, rcond=rcond)
    except np.linalg.LinAlgError:
        say("ERROR: pseudo-inverse of D-optimal submatrix failed.")
        log.close()
        sys.exit(1)

    say("run_dopt_maxvol: inverse_A_subset shape: {} (square={})".format(
        inverse_a_subset.shape,
        inverse_a_subset.shape[0] == inverse_a_subset.shape[1]))

    del a_subset
    del A_work
    gc.collect()

    inv_path = os.path.join(outdir, "inverse_A_subset.npy")
    np.save(inv_path, inverse_a_subset)
    np.savetxt(os.path.join(outdir, "maxvol_pivots.txt"),
               piv, fmt="%d")

    say("run_dopt_maxvol: wrote {} ({} pivot rows).".format(
        inv_path, len(piv)))
    log.close()
    return len(piv)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="D-optimality maxvol job (compute-node entry point)")
    parser.add_argument("--amat", required=True,
                        help="Path to training A.txt / A_comb.txt")
    parser.add_argument("--fm-setup", default="",
                        help="Path to GEN_FF/fm_setup.in (for FITENER layout)")
    parser.add_argument("--traj-list", default="",
                        help="Path to traj_list.dat (FITENER force-row layout)")
    parser.add_argument("--component", action="store_true",
                        help="Keep fx/fy/fz as separate rows (DO_COMPONENT)")
    parser.add_argument("--rcond", type=float, default=1.0e-12,
                        help="pinv relative singular-value cutoff")
    parser.add_argument("--outdir", default=".",
                        help="Directory for A_atomic.txt and inverse_A_subset.npy")
    args = parser.parse_args(argv)

    amat = os.path.abspath(args.amat)
    fm_setup = os.path.abspath(args.fm_setup) if args.fm_setup else ""
    traj_list = os.path.abspath(args.traj_list) if args.traj_list else ""
    outdir = os.path.abspath(args.outdir)

    if not os.path.isfile(amat):
        print("ERROR: A matrix not found:", amat)
        sys.exit(1)

    run_maxvol(amat, fm_setup, traj_list, args.component, args.rcond, outdir)


if __name__ == "__main__":
    main()
