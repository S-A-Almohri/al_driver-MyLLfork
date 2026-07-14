<p style="text-align:center;">
    <img src="./doc/ChIMES_Github_logo-2.png" alt="" width="250"/>
</p>
<hr>

ChIMES Active Learning Driver Documentation
------------------------------------------------


*Note: This documentation is under still construction.*

The Active Learning Driver is an extensible multifunction workflow tool for generating ChIMES [1] models. At its simplest, the ALD can be used for model generation via iterative refinement [2], at at its most complex, via active learning [3].

Before proceeding, the user is strongly encouraged to familiarize themselves with the ChIMES literature (See references below) and ChIMES LSQ user manual. Note that the ALD itself only contains the tools necessary to orchestrate model generation and active learning, and must be used in conjunction with the ChIMES design matrix generator, a supported MD code, and a supported quantum code. Note also that the ALD is only intended for use on high performance computing platforms and currently only supports runs via Slurm (SBATCH) schedulers. 

* [1] [**link**](https://doi.org/10.1021/acs.jctc.7b00867) R.K. Lindsey, L.E. Fried, N. Goldman, JCTC, 13, 6222 (2017)
* [2] [**link**](https://doi.org/10.1063/5.0012840) R.K. Lindsey, N. Goldman, L.E. Fried, S. Bastea, JCP, 153 054103 (2020)
* [3] [**link**](https://doi.org/10.1063/5.0021965) R.K. Lindsey, L.E. Fried, N. Goldman, S. Bastea, JCP, 153 134117 (2020)

The Active Learning Driver was developed at Lawrence Livermore National Laboratory with funding from the US Department of Energy (DOE), and is open source, distributed freely under the terms of the LGPL v3.0 License.

This work was produced under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.


<hr>

Documentation
----------------

[**Documentation**](https://https://al-driver.readthedocs.io/en/latest/) is available, but under construction.

<hr>

D-Optimality Cluster Selection
------------------------

When running cluster-based active learning (`DO_CLUSTER = True`), the driver must
decide which of the many candidate atomic clusters extracted from MD trajectories
are worth sending to expensive quantum (e.g. VASP) labeling. Two down-selection
strategies are available:

* **MC energy histogram (default):** `gen_subset()` performs a ChIMES "dumb-energy"
  calculation for each candidate (`CLUENER_CALC`) and selects clusters that flatten
  an energy histogram.
* **D-optimality (maxvol):** `gen_subset_dopt()` selects clusters by *model
  uncertainty* (extrapolation grade, gamma) using the maxvol algorithm. The
  ChIMES dumb-energy calculation is skipped entirely.

To enable the D-optimality path, set `DO_DOPT = True` (requires `DO_CLUSTER = True`).

### How it works

1. Prepare candidate-cluster inputs for a `chimes_lsq` descriptor job
   (`DOPT_DESCRIPTORS/`).
2. Submit **two Slurm jobs in parallel** (neither runs maxvol on the head/login node):
   * **Descriptor job:** `chimes_lsq` builds the candidate design matrix
     (`DOPT_DESCRIPTORS/A.txt`).
   * **Maxvol job:** `src/run_dopt_maxvol.py` stream-builds `A_atomic` from the
     current fit's reference matrix (`GEN_FF/A_comb.txt` or `A.txt`; force rows
     only; FITENER energy rows stripped), runs **maxvol**, and writes the
     **pseudo-inverse** (`pinv`, cutoff `DOPT_RCOND`) to
     `DOPT_MAXVOL/inverse_A_subset.npy`. By default each atom's three force rows
     (`fx, fy, fz`) are hstacked into one row of width `3*n_feat`; see
     `DO_COMPONENT` to keep them separate. Maxvol needs a *tall* matrix (more
     rows than features). Rank / condition diagnostics are written to
     `DOPT_MAXVOL/dopt_maxvol.log`.
3. Wait until **both** jobs finish.
4. On the driver process (lightweight), stream-score each candidate cluster:
   `gamma = max(A_candidate @ inverse_A_subset)`.
5. Keep clusters whose gamma falls in `[DOPT_GAMMA_MIN, DOPT_GAMMA_MAX]`. Below the
   minimum, a cluster is already well-represented in the training set; above the
   maximum, it is too far from the current model's domain of validity.
6. Write `all.selection.dat` / `all.xyzlist.dat` and diagnostics
   (`dopt_gamma_dist.pdf`, `dopt_cluster_gamma.txt`).

If no candidate clusters exceed the gamma threshold and `DOPT_STOP_ON_EMPTY = True`,
active learning is treated as converged: QM labeling is skipped and convergence is
recorded in `restart.dat` so subsequent driver invocations do not continue. (At
ALC-0 an empty selection is instead a fatal error, since active learning cannot
start with nothing selected.)

**Prerequisite:** `pip install maxvolpy` on the compute node that runs the maxvol job.

### Configuration flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `DO_DOPT` | bool | `False` | Use D-optimality (maxvol) for cluster down-selection instead of the MC energy histogram. Requires `DO_CLUSTER = True`. When `True`, the ChIMES dumb-energy calculation (`CLUENER_CALC`) is skipped. |
| `DOPT_GAMMA_MIN` | float | `3.0` | Lower gamma bound. Clusters below this are considered already well-represented and are excluded. |
| `DOPT_GAMMA_MAX` | float | `10.0` | Upper gamma bound. Clusters above this are considered too far from the model's domain of validity and are excluded. |
| `DOPT_STOP_ON_EMPTY` | bool | `True` | Stop active learning (convergence) when D-opt selects zero clusters above the gamma threshold. Skips QM submission and records completion in `restart.dat`. |
| `DO_COMPONENT` | bool | `False` | Controls how `A_atomic` is built. `False`: hstack each atom's `fx, fy, fz` into one row of width `3*n_feat` (maxvol/gamma operate per atom). `True`: leave the A matrix as-is — each force component row is kept separate (width `n_feat`), so each atom contributes three rows and maxvol/gamma operate per force component. FITENER energy rows are stripped in either case. |
| `DOPT_RCOND` | float | `1.0e-12` | Relative singular-value cutoff for the pseudo-inverse of the D-optimal sub-matrix. Singular values below `DOPT_RCOND × (largest singular value)` are dropped, guarding against near-singular / rank-deficient inverses. Set to `0` to recover pure `inv`-like behavior. |
| `DOPT_MAXVOL_NODES` | int | `CHIMES_BUILD_NODES` | Nodes for the maxvol Slurm job. |
| `DOPT_MAXVOL_PPN` | int | `HPC_PPN` | Processors per node for the maxvol Slurm job. |
| `DOPT_MAXVOL_TIME` | str | `CHIMES_BUILD_TIME` | Walltime for the maxvol Slurm job. |
| `DOPT_MAXVOL_QUEUE` | str | `CHIMES_BUILD_QUEUE` | Queue for the maxvol Slurm job. |
| `DOPT_MAXVOL_MODULES` | str | `CHIMES_LSQ_MODULES` | Optional `module load` string for the maxvol job. Prefer `DOPT_MAXVOL_PYTHON` for conda/`maxvolpy`. |
| `DOPT_MAXVOL_MEM` | str | `""` | Memory in GB for the maxvol job (used on UM-ARC via `--mem-per-cpu`). |
| `DOPT_MAXVOL_PYTHON` | str | `HPC_PYTHON` | Python for the **maxvol job only** (must provide `numpy` + `maxvolpy`), e.g. `.../envs/Pmini/bin/python`. Does not change `HPC_PYTHON` used by the rest of ALD. |

The `DOPT_*` and `DO_COMPONENT` flags are only read when `DO_DOPT = True`. A complete
worked example is provided in
[`examples/cluster_based_active_learning_single_statepoint-VASP-dopt/`](examples/cluster_based_active_learning_single_statepoint-VASP-dopt/).

### Numerical stability

D-optimality via maxvol assumes the design matrix is **full column rank**. ChIMES
basis sets (e.g. Chebyshev / hierarchical) are frequently near-collinear — which is
why the fit itself uses regularized regression — and that same collinearity can make
the maxvol sub-matrix singular or ill-conditioned. A naive `inv()` would not raise an
error on a near-singular matrix; it would silently return a garbage inverse and
poison every gamma score.

To guard against this, the maxvol compute job (`run_dopt_maxvol.py`):

* reports the **rank** of `A_atomic` and warns if it is rank deficient (collinear
  features);
* reports the **condition number** of the selected sub-matrix and warns when it is
  ill-conditioned (`> 1e10`) or near-singular (`> 1e14`);
* uses a **pseudo-inverse** (`np.linalg.pinv`, cutoff `DOPT_RCOND`) instead of a
  direct inverse, so degenerate singular directions are truncated rather than
  amplified.

Note that `DO_COMPONENT` changes the *shape* of the matrix but not the rank of the
underlying feature columns, so it is **not** a substitute for these guards. If the
warnings fire persistently, reduce the basis, add training data, or increase
`DOPT_RCOND`.

<hr>

Community
------------------------

Questions, discussion, and contributions (e.g. bug fixes, documentation, and extensions) are welcome. 

Additional Resources: [ChIMES Google group](https://groups.google.com/g/chimes_software).

<hr>

Contributing
------------------------

Contributions to the The ChIMES AL Driver should be made through a pull request, with ``develop`` as the destination branch. A test suite log file should be attached to the PR.  The `develop` branch has the latest contributions. Pull requests should target `develop`, and users who want the latest package versions, features, etc. can use `develop`.

<hr>


Authors
----------------

The The ChIMES AL Driver was developed by Rebecca K. Lindsey.

Contributors can be found [here](https://github.com/rk-lindsey/al_driver/graphs/contributors).

<hr>

<!--- Citing
<!--- ----------------
<!--- 
<!--- See [the documentation](https://chimes-calculator.readthedocs.io/en/latest/citing.html) for guidance on referencing ChIMES and the ChIMES calculator in <> a publication.

<hr>

License
----------------

The ChIMES AL Driver is distributed under terms of [LGPL v3.0 License](https://github.com/rk-lindsey/chimes_calculator/blob/main/LICENSE). This work was produced under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344. LLNL-CODE-839335
