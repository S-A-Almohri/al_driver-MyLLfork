# Cluster-based Active Learning with D-Optimality (maxvol) selection
# Based on: cluster_based_active_learning_single_statepoint-VASP
#
# Key difference from the MC energy-histogram example:
#   DO_DOPT = True
#   This activates gen_subset_dopt() instead of gen_subset(), which:
#     1. Prepares candidate-cluster inputs for chimes_lsq.
#     2. Submits TWO Slurm jobs in parallel:
#          (a) chimes_lsq → candidate descriptors (DOPT_DESCRIPTORS/A.txt)
#          (b) run_dopt_maxvol.py → A_atomic + maxvol + pinv
#              (DOPT_MAXVOL/inverse_A_subset.npy) on a compute node
#     3. Waits for both jobs, then scores gamma = max(A_cand @ inv_A_ref).
#     4. Selects clusters whose max gamma falls in [DOPT_GAMMA_MIN, DOPT_GAMMA_MAX].
#   The ChIMES dumb-energy calculation (CLUENER_CALC) is skipped entirely.
#
# Prerequisites:
#   A python with maxvolpy for the maxvol Slurm job — set DOPT_MAXVOL_PYTHON
#   (e.g. .../envs/Pmini/bin/python). Keep HPC_PYTHON for the rest of ALD.

################################
##### General options
################################

ATOM_TYPES = ['O', 'H']
NO_CASES = 1

DRIVER_DIR     = "/p/lustre3/lindsey11/al_driver-myLLfork/"
WORKING_DIR    = "/p/lustre3/lindsey11/al_driver-myLLfork/examples/cluster_based_active_learning_single_statepoint-VASP-dopt/"
CHIMES_SRCDIR  = "/p/lustre3/lindsey11/test_chimes_lsq-for-LL_to_ext_PR/chimes_lsq-LLfork/src/"

################################
##### General HPC options
################################

HPC_ACCOUNT = "iap"
HPC_PYTHON  = "/usr/tce/bin/python3"
HPC_SYSTEM  = "slurm"
HPC_PPN     = 56 # Ruby has 56

HPC_EMAIL   = False

################################
##### ChIMES LSQ
################################

ALC0_FILES    = WORKING_DIR + "ALL_BASE_FILES/ALC-0_BASEFILES/"
CHIMES_LSQ    = CHIMES_SRCDIR + "../build/chimes_lsq"
CHIMES_SOLVER = CHIMES_SRCDIR + "../build/chimes_lsq.py"
CHIMES_POSTPRC= CHIMES_SRCDIR + "../build/post_proc_chimes_lsq.py"

# Generic weight settings

WEIGHTS_FORCE = [ ["A"], [[1.0  ]] ]
WEIGHTS_FGAS  = [ ["A"], [[1.0  ]] ]
WEIGHTS_ENER  = [ ["A"], [[0.3  ]] ]
WEIGHTS_EGAS  = [ ["A"], [[1.0  ]] ]
WEIGHTS_STRES = [ ["A"], [[100.0]] ]

REGRESS_ALG = "dlasso"
REGRESS_VAR = "1.0E-5"
REGRESS_NRM = True

# Stress tensor settings

STRS_STYLE = "ALL"  # Options: "DIAG" or "ALL"

# chimes_lsq build job — also reused by gen_subset_dopt for the descriptor job

CHIMES_BUILD_NODES = 1
CHIMES_BUILD_QUEUE = "pdebug"
CHIMES_BUILD_TIME  = "01:00:00"

CHIMES_SOLVE_NODES = 2
CHIMES_SOLVE_QUEUE = "pdebug"
CHIMES_SOLVE_TIME  = "01:00:00"

################################
##### Cluster-based active learning with D-Optimality selection
################################

DO_CLUSTER = True
MAX_CLUATM = 10
TIGHT_CRIT = WORKING_DIR + "ALL_BASE_FILES/tight_bond_crit.dat"
LOOSE_CRIT = WORKING_DIR + "ALL_BASE_FILES/loose_bond_crit.dat"
CLU_CODE   = "/p/lustre3/lindsey11/al_driver-myLLfork/utilities/new_ts_clu.cpp"

# Use D-optimality (maxvol) for cluster downselection.
# When True, the ChIMES dumb-energy calculation (CLUENER_CALC) is skipped and
# gen_subset_dopt() runs instead of gen_subset().
DO_DOPT = True

# Clusters with max per-atom gamma outside this window are excluded:
#   gamma < DOPT_GAMMA_MIN  →  already well-represented in the training set
#   gamma > DOPT_GAMMA_MAX  →  too far from the current model's domain of validity
DOPT_GAMMA_MIN = 3.0
DOPT_GAMMA_MAX = 10.0

# Stop active learning when D-opt finds no clusters above the gamma threshold.
# Skips QM submission and records convergence in restart.dat.
DOPT_STOP_ON_EMPTY = True

# How the pseudo A matrix (A_atomic) is built before maxvol:
#   False (default) - hstack each atom's three force rows (fx, fy, fz) into a
#                     single row of width 3*n_feat; maxvol/gamma operate per atom.
#   True            - leave the A matrix as-is: each force component row is kept
#                     separate (width n_feat), so each atom contributes three
#                     rows; maxvol/gamma operate per force component.
# Energy (FITENER) rows are stripped either way.
DO_COMPONENT = False

# Numerical-stability guard for the D-optimal inverse. The pseudo-inverse (pinv)
# of the maxvol submatrix is always used; DOPT_RCOND is the relative
# singular-value cutoff (singular values below DOPT_RCOND * largest are dropped),
# which protects against near-singular / rank-deficient submatrices arising from
# collinear ChIMES feature columns. Set to 0 to recover pure inv-like behavior.
# The rank of the design matrix and the submatrix condition number are always
# reported so ill-conditioning is visible in the log.
DOPT_RCOND = 1.0e-12

# Descriptor chimes_lsq Slurm job (MPI via ibrun/srun; can be smaller than BUILD_AMAT).
# Defaults (if unset) mirror CHIMES_BUILD_* / HPC_PPN / CHIMES_LSQ_MODULES.
DOPT_DESC_NODES   = 1
DOPT_DESC_PPN     = HPC_PPN
DOPT_DESC_TIME    = "01:00:00"
DOPT_DESC_QUEUE   = "pdebug"
DOPT_DESC_MODULES = ""   # falls back to CHIMES_LSQ_MODULES if unset

# Maxvol Slurm job (runs in parallel with the descriptor chimes_lsq job).
# Defaults (if unset) mirror CHIMES_BUILD_* / HPC_PPN / CHIMES_LSQ_MODULES.
# Override these when the head/login node cannot hold A_atomic in memory.
DOPT_MAXVOL_NODES   = 1
DOPT_MAXVOL_PPN     = 1      # serial Python; maxvolpy is not MPI
DOPT_MAXVOL_TIME    = "01:00:00"
DOPT_MAXVOL_QUEUE   = "pdebug"
DOPT_MAXVOL_MODULES = ""   # optional module load; prefer DOPT_MAXVOL_PYTHON for conda
# DOPT_MAXVOL_MEM   = "128"  # GB; only applied on UM-ARC via helpers.create_and_launch_job

# Python used ONLY by the maxvol compute job (must import maxvolpy + numpy).
# Keep this separate from HPC_PYTHON so the rest of ALD can use a different env.
# Example (Pmini conda env):
#   DOPT_MAXVOL_PYTHON = "/path-to/CondaEnv/miniconda/envs/Pmini/bin/python"
DOPT_MAXVOL_PYTHON = HPC_PYTHON  # <-- set to Pmini python before running on the cluster

# MEM_ECUT is still used as a pre-filter if DO_DOPT is False; kept here for
# easy toggling between methods.
MEM_ECUT = 4000.0

################################
##### Molecular Dynamics
################################

MD_STYLE = "CHIMES"
CHIMES_MD_MPI = CHIMES_SRCDIR + "../build/chimes_md"
CHIMES_MD_SER = CHIMES_SRCDIR + "../build/chimes_md-serial"

MOLANAL  = CHIMES_SRCDIR + "../contrib/molanal/src/"
MOLANAL_SPECIES = ["H2O", "H3O", "OH"]

MD_NODES = [1] * NO_CASES
MD_QUEUE = ['pdebug'] * NO_CASES
MD_TIME  = ['00:05:00'] * NO_CASES

################################
##### QM-Specific variables (Single point calculations)
################################

QM_FILES = WORKING_DIR + "ALL_BASE_FILES/QM_BASEFILES"

VASP_EXE     = "/p/lustre3/lindsey11/vasp_std.5.4.4"
VASP_TIME    = "01:00:00"
VASP_NODES   = 1
VASP_PPN     = 56
VASP_QUEUE   = "pdebug"
VASP_MODULES = "intel-classic/19.1.2 mvapich2/2.3.6 mkl"
