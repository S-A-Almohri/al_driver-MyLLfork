# Global (python) modules

import glob # Warning: glob is unserted... set my_list = sorted(glob.glob(<str>)) if sorting needed
import os
import sys
import argparse
import numpy as np
import random
import matplotlib             # This and the next command prevents matplotlib from requiring an x-server (useful when screen is used)
matplotlib.use('Agg')

# Global (python) modules - for  matplotlib

import matplotlib.pyplot as plt
import cycler
from cycler import cycler
import math as m

# Local modules
  
import helpers


def populate_repo(my_ALC):

    """ 
    
    Updates central repository's list of selected species. 
    
    Usage: populate_repo(1)

    """

    if not os.path.isdir("../CENTRAL_REPO"):
        helpers.run_bash_cmnd("mkdir ../CENTRAL_REPO")
        
    currdir = helpers.run_bash_cmnd("pwd").rstrip()
    os.chdir("../ALC-" + repr(my_ALC))

    # Create the list of selected species for the current ALC
    
    helpers.run_bash_cmnd("rm -f ../CENTRAL_REPO/ALC-" + repr(my_ALC) + ".all_selections.xyzlist")
    
    ifstream = open("all.xyzlist.dat",'r')
    all_xyz  = ifstream.readlines()
    ifstream .close()
    
    ifstream = open("all.selection.dat",'r')
    all_sel  = ifstream.readlines()
    ifstream .close()
    
    ofstream = open("../CENTRAL_REPO/ALC-" + repr(my_ALC) + ".all_selections.xyzlist", 'w')
    
    
    for j in range(len(all_sel)):
        for i in range(len(all_xyz)):

            if i == int(all_sel[j].rstrip()):
            
                line = all_xyz[i].split()

                line = ' '.join(line[0:-1]) + " " + "../ALC-" + repr(my_ALC) + "/" + line[-1]

                ofstream.write(line + '\n')
                continue
    ofstream.close()
    
    
    # Recompile the central repo list of files

    helpers.run_bash_cmnd("rm -f ../CENTRAL_REPO/full_repo.xyzlist")

    helpers.cat_specific("../CENTRAL_REPO/full_repo.xyzlist", sorted(glob.glob("../CENTRAL_REPO/*.all_selections.xyzlist")))

    os.chdir(currdir)
    

def cleanup_repo(my_ALC):

    """ 
    
    Cleans up central repository's list of selected species. 
    
    Usage: cleanup_repo(1)
    
    Notes: Useful when AL dies mid-run by ensuring the central repsotory's
           list of selected species only corresponds to ALC's < my_ALC.
    
    """

    # check if the central repo exists
    
    if not os.path.isdir("../CENTRAL_REPO"):
        return
        
    print("Cleaning up the current CENTRAL_REPO...")
    
    currdir = helpers.run_bash_cmnd("pwd").rstrip()
    os.chdir("../CENTRAL_REPO")        
        
    # Get a list of all present 'ALC-X.all_selections.xyzlist' files
    # remove any files for ALC-X where X > my_ALC
    
    checklist = glob.glob("ALC-*.all_selections.xyzlist")
    
    for i in checklist:
    
        idx = i.split('.')[0].split('-')[1] # the "X" in ALC-X.all_selections.xyzlist
        
        if int(idx) >= my_ALC:
            
            helpers.run_bash_cmnd("rm -f " + i)
        
    os.chdir(currdir)
    
    if my_ALC > 0:
        populate_repo(my_ALC-1)
    
    print("...done.")
    
    

def GET_BIN(val,bins):

    """ 
    
    Determines a bin number for a given energy. 
    
    Usage: GET_BIN(12.2573,bins)
    
    Notes: This function is intended for use by gen_subset only.
    
    """

    for i in range(len(bins)-1):
    
        if (val >= bins[i]) and (val < bins[i+1]):
            return i
        elif val == bins[i+1]:
            return i
            
    print("PROBLEM: No bin was found for value ", val)
    print(bins[0])
    print(bins[len(bins)-1])
    exit()
            
def GEN_NORM_HIST(idx_list, ener_list, central_repo_enerlist, minval, maxval, nbins):

    """ 
    
    Normalizes a histogram.
    
    Usage: GEN_NORM_HIST(idx_list, ener_list, central_repo_enerlist, minval, maxval, nbins)
    
    Notes: This function is intended for use by gen_subset only.
    
    """

    ener_vals = []

    for i in range(len(idx_list)):
        ener_vals.append(ener_list[idx_list[i]])
    
    ener_vals += list(central_repo_enerlist)

    prob, bins = np.histogram(ener_vals, bins=nbins, range=(minval,maxval), density=False)

    if(float(np.sum(prob)) == 0.0):
        print("Problem, found a zero sum:")
        print(prob)
        print() 
        exit()
    
    prob = prob/float(np.sum(prob))

    return prob, bins    
        
    
def SET_CONDITIONS(nsweeps):

    """ 
    
    Determines which sweeps to print MC stats for.
    
    Usage: SET_CONDITIONS(nsweeps)
    
    Notes: This function is intended for use by gen_subset only.
    
    """
    
    arr = []

    for i in range(5):
        arr.append(m.exp(float(i)*2))
        
    for i in range(5):
        arr[i] = m.ceil((arr[i] - arr[0])/(arr[4] - arr[0])*(nsweeps-1))
        
    print("Stats will be printed for sweeps: ",arr)
    return arr


def gen_subset(**kwargs): # time python gen_subset.py  all.energies_normed $SELECTIONS $SWEEPS 0 # Last 2 args: # to select, # sweeps, (optional:) E-cutoff

    """ 
    
    Selects a subset of species to add to the central repository.
    
    Usage: gen_subset(<arguments>)
    
    Notes: See function definition in helpers.py for a full list of options. 
           Expects to be run from the ALC-X directory
           Expects "all.energies_normed" in the ALC-X directory
           Generates a plot of histogram and RMS evolution
           Generates a list of selected species
    
    """
    
    ################################
    # 0. Set up an argument parser
    ################################    
    
    default_keys   = [""]*7
    default_values = [""]*7        
    
    # Cluster specific controls
    
    default_keys[0 ] = "energies"     ; default_values[0 ] = 'all.energies_normed' # List of energies to select from  
    default_keys[1 ] = "nsel"     ; default_values[1 ] = '400'                     # Number of selections to make     
    default_keys[2 ] = "nsweep"     ; default_values[2 ] = '200000'                # Number of MC sqeeps         
    default_keys[3 ] = "nbins"     ; default_values[3 ] = '20'                     # Number of histogram bins     
    default_keys[4 ] = "ecut"     ; default_values[4 ] = '1.0E10'                  # Maximum energy to consider     
    default_keys[5 ] = "repo"     ; default_values[5 ] = ''                        # Location of central repo energies
    default_keys[6 ] = "seed"     ; default_values[6 ] = 1                         # Seed for random number generator

        
    args = dict(list(zip(default_keys, default_values)))
    args.update(kwargs)
    
    ENER    =       args["energies"]
    NSELECT =   int(args["nsel"    ])
    NSWEEPS =   int(args["nsweep"  ]) 
    NBINS   =   int(args["nbins"   ]) 
    ECUTOFF = float(args["ecut"    ])
    REPENER =       args["repo"    ]    
    
    print("Parsed arguments: ")
    print("energies: ",ENER)   
    print("nsel:     ",NSELECT)
    print("ncyc:     ",NSWEEPS)
    print("nbins:    ",NBINS)
    print("ecut:     ",ECUTOFF)
    print("repo:     ",REPENER)    
    

    ################################
    # 1. Read the energy file(s), Identify max/min values
    ################################

    # Energies to select from:
    
    ENER    = np.loadtxt(ENER)

    if NSELECT > ENER.shape[0]:
        print("ERROR: NSELECT is larger than the number of available energies: ")
        print("NSELECT:   ", NSELECT)
        print("NENERGIES: ", ENER.shape[0])
        exit()
        
    # Find the max and min energy values
    
    MAX_VAL = ENER[0]
    MIN_VAL = ENER[0]
    
    MIN_FROM_MAIN = True # Assume max/min values are coming from ENER and not REPENER
    MAX_FROM_MAIN = True
    
    MAX_IDX = 0
    MIN_IDX = 0

    PAST_CUT_VAL = []
    PAST_CUT_IDX = []

    for i in range(len(ENER)):
    
        if abs(ENER[i]) >= ECUTOFF:
            PAST_CUT_VAL.append(ENER[i])
            PAST_CUT_IDX.append(i)    
        else:    

            if MAX_VAL < ENER[i]:
                MAX_VAL = ENER[i]
                MAX_IDX = i

            if MIN_VAL > ENER[i]:
                MIN_VAL = ENER[i]
                MIN_IDX = i
            
    
    # Central repository energies
    
    IGNORE_REPO_GTMAX = 0
    IGNORE_REPO_LTMIN = 0
    
    if REPENER != '':
    
        REPENER = np.loadtxt(REPENER)    
        
        REPO_POP_LIST = []
        
        for i in range(len(REPENER)):
            if abs(REPENER[i]) >= ECUTOFF:
                print("WARNING: Repo energy is outside ECUTOFF!")
                REPO_POP_LIST.append(i)
                
            if REPENER[i] > MAX_VAL and (i not in REPO_POP_LIST): # modified because i might repeat
                #print "WARNING: Found central repo energy larger than current iteration\'s ... ignoring."
                IGNORE_REPO_GTMAX += 1
                REPO_POP_LIST.append(i)
            elif REPENER[i] < MIN_VAL and (i not in REPO_POP_LIST): # modified
                #print "WARNING: Found central repo energy smaller than current iteration\'s ... ignoring."
                IGNORE_REPO_LTMIN += 1
                REPO_POP_LIST.append(i)
                
        if len(    REPO_POP_LIST ) >  0:
        
            TMP = list(REPENER)
            
            REPO_POP_LIST.sort(reverse=True)
            
            for i in range(len(REPO_POP_LIST)):
            
                TMP.pop(REPO_POP_LIST[i])
            
            REPENER = np.array(TMP)
            
        
        for i in range(len(REPENER)):

            if MAX_VAL < REPENER[i]:
                MAX_VAL = REPENER[i]
                MAX_IDX = i
                MAX_FROM_MAIN = False

            if MIN_VAL > REPENER[i]:
                MIN_VAL = REPENER[i]
                MIN_IDX = i
                MIN_FROM_MAIN = False
    else:
        REPENER = []
        
    #PAST_CUT_VAL.sort(reverse=True)    
    #PAST_CUT_IDX.sort(reverse=True)
                
    print("FOUND MIN/MAX VALS (from ENER?): ", MIN_VAL, MAX_VAL, MIN_FROM_MAIN, MAX_FROM_MAIN) 
    print("FOUND MIN/MAX IDXS             : ", MIN_IDX, MAX_IDX)    
    print("Ignored repo vals (<min)       : ", IGNORE_REPO_LTMIN)
    print("Ignored repo vals (>max)       : ", IGNORE_REPO_GTMAX)
                
    ########################
    # Generate the probability distribution for the energy list
    # (a normalized numpy histogram; NBINS bins; NOT including REPENER
    # values except max/min)
    ########################
    
    # Generate

    PROB, BINS = np.histogram(ENER, bins=NBINS, range=(MIN_VAL,MAX_VAL), density=False)
    
    if(float(np.sum(PROB)) == 0.0):
        print("Problem, found a zero sum:")
        print(PROB)
        print() 
        exit()
        
    print("Total number of configurations: ",sum(PROB))
    print("Number of configurations in each bin: ",PROB)
    
    NSWEEPS *= len(ENER)
        
    PROB = PROB/float(np.sum(PROB))
    


    # Plot Original random selection results ( and set color scheme)
    
    colors = plt.cm.OrRd(np.linspace(0, 1, 10)[::-1]) # See https://matplotlib.org/tutorials/colors/colormaps.html for options
    plt.rcParams['axes.prop_cycle'] = cycler('color', colors)    

    MID = (BINS[:-1] + BINS[1:]) / 2.0    # Plot (use bin midpoints)
    plt.plot(MID, PROB, marker='o', fillstyle='none', label='all current iteration configs')


    ########################
    # Generate the initial sub-selection
    ########################
    
    random.seed(args["seed"])
    
    # Step 0: Combine the new (ENER) and old (REPENER) energies into REPO. If
    #         min/max are from ENER, remove (pop off) the from REPO b/c they 
    #         will always be in the selected subset then we can always pick from 
    #         the first len(ENER)-NPOPPED configuations
    
    NENER = ENER.shape[0]
    
    #COMBINED_ENERS = np.concatenate((ENER, REPENER)) # ENER + REPENER # First len(ENER) entries are ALWAYS from ENER    
    
    REPO = list(range(NENER))

    NPOPPED = 0
    NCUTOFF = 0
    
    POP_LIST = list(PAST_CUT_IDX)

    NCUTOFF = len(PAST_CUT_IDX)
    
    if MAX_FROM_MAIN and (MAX_IDX not in POP_LIST): # modified
        POP_LIST.append(MAX_IDX)
        NPOPPED += 1

    if MIN_FROM_MAIN and (MIN_IDX not in POP_LIST): # modified
        POP_LIST.append(MIN_IDX)
        NPOPPED += 1
        
    POP_LIST.sort(reverse=True)
    
    for i in range(len(POP_LIST)):
        REPO.pop(REPO.index(POP_LIST[i]))
    
    print("Values above energy cutoff: ", NCUTOFF)
        
    # Step 1: Select NSELECT-NPOPPED random energies only from SELE elemnts from ENER! (repetition not allowed)
        
    if (NENER-NPOPPED-NCUTOFF) < (NSELECT-NPOPPED):
        print("ERROR: not enough candidate clusters to select from. Try increasing energy cutoff.")
        exit(0)    

    SELE = random.sample(REPO[0:(NENER-NPOPPED-NCUTOFF)],NSELECT-NPOPPED)     # random sample returns *VALUES* of REPO to SELE... so SELE[i] = energy list index

    # Need to figure out the corresponding REPO index for the values stored in SELE, before popping them off

    for i in range(len(SELE)):
        REPO.pop(REPO.index(SELE[i]))
        
    # Determine the number of elements in SELE arising from ENER, and in REPO arising from ENER
    
    N_ENER_SELE = len(SELE)
    N_ENER_REPO = NENER - NPOPPED - N_ENER_SELE - NCUTOFF
    
    
    ########################
    # Do a MC sweep to update selections
    ########################
    #
    # Our scheme: accept if P_OLD - P_NEW > rand(0,1)
    # This will bias toward new configurations that were 
    # lower in probability, and should flatten our hist
    
    
    # Set up conditions for printing current histogram
    
    CONDITIONS = SET_CONDITIONS(NSWEEPS)

    # Generate the probability histogram for the MC run... histogram is over selected new energies (ENER) and all old energies (REPENER)

    HIST_SELE = SELE[:]

    if MIN_FROM_MAIN:
        HIST_SELE += [MIN_IDX]
    if MAX_FROM_MAIN:
        HIST_SELE += [MAX_IDX]
        
    REMAINING =  REPO[N_ENER_REPO:] # These are values from the central repository
        
    HIST_SELE += REMAINING

    SELE_PROB, SELE_BINS = GEN_NORM_HIST(HIST_SELE, ENER, REPENER, MIN_VAL, MAX_VAL, NBINS)

    # Do MC sweeps

    HIST_SELE = []
    SSQR_LIST = []
    SSQR      = "not yet calculated"

    if NSELECT == ENER.shape[0]:
        NSWEEPS = 1
        HIST_SELE = SELE[:]
        
        if MIN_FROM_MAIN:
            HIST_SELE += [MIN_IDX]
        if MAX_FROM_MAIN:
            HIST_SELE += [MAX_IDX]
            
        HIST_SELE += REMAINING

        SELE_PROB, SELE_BINS = GEN_NORM_HIST(HIST_SELE, ENER, REPENER, MIN_VAL, MAX_VAL)

    
    for sweep in range(NSWEEPS):

        if sweep in CONDITIONS: #if ((NSWEEPS/5>0) and (sweep+1)%(NSWEEPS/5)==0):
            print("Running MC sweep " + repr(sweep) + " of " + repr(NSWEEPS) + " ... SSQR: " +  repr(SSQR)) 
        
        NONE_ACC = True

        for i in range(NSELECT-NPOPPED):
        
            # print "    Running MC step " + `i` + " of " + `NSELECT-2`

            if NSELECT < ENER.shape[0]:        

                # Select an existing and new energy (OLD and NEW = index)

                
                OLD = random.randint(0,N_ENER_SELE-1) # provides a SELE index           # One from the current subset
                NEW = random.randint(0,N_ENER_REPO-1) # Provides a REPO index           # One from the possible repo of configs


                if (OLD >len(ENER)) or (NEW >len(ENER)):
                    print("++++")                
                    print("ERROR: ", OLD, NEW)
                    
                    print(len(ENER))
                    ENER.sort(reverse=True)
                    print(ENER[0])
                    print("++++")
                    
                    exit()

                # Apply the acceptance criteria

                P_OLD = SELE_PROB[GET_BIN(ENER[SELE[OLD]],SELE_BINS)]
                P_NEW = SELE_PROB[GET_BIN(ENER[REPO[NEW]],SELE_BINS)]
                        
                
                RAND  = random.random()
                CRIT  = 0.5*(P_NEW-P_OLD) + 1.0
                #CRIT  = (P_NEW-P_OLD) #  + 1.0    
                #CRIT = 1.0 + (P_OLD - P_NEW)*0.5
                
                if False:            
                
                    print("")
                    print("           old:  ", P_OLD)
                    print("           new:  ", P_NEW)
                    print("           rand: ", RAND)
                    print("           crit: ", CRIT)
            
                if ( CRIT > RAND): # This is a high prob cfg... we want to bias against it
                    continue
    
                NONE_ACC = False
                    
                # We've accepted the move... update the selected and stored respositories and re-compute the probabilities

                SELE_VAL = SELE.pop(OLD)
                REPO_VAL = REPO.pop(NEW) # .pop returns REPO[NEW], and removes element [NEW] from REPO
            
                SELE.append(REPO_VAL)
                REPO.append(SELE_VAL)

            
            
            if any( x>len(ENER) for x in SELE):
                
                SELE.sort(reverse=True)
            
                print("Warning: Selected value too large: ")
                print(len(SELE))
                print(SELE[0])
                print(len(ENER))
                print(ENER)            
                exit()
            
            
            HIST_SELE = SELE[:]
            
            if MIN_FROM_MAIN:
                HIST_SELE += [MIN_IDX]
            if MAX_FROM_MAIN:
                HIST_SELE += [MAX_IDX]
                
            HIST_SELE += REMAINING

            SELE_PROB, SELE_BINS = GEN_NORM_HIST(HIST_SELE, ENER, REPENER, MIN_VAL, MAX_VAL, NBINS)
            

            if NSELECT == ENER.shape[0]:
                break
            
            
        # Compute sum of squared residuals (our "equilibration" criteria)    
            
        if NONE_ACC:
            if len(SSQR_LIST) == 0:
                SSQR_LIST.append(-1)
                
            SSQR = SSQR_LIST[len(SSQR_LIST)-1]
            SSQR_LIST.append(SSQR)
        else:
            TARGET = 1/float(NBINS) # 1/nbins
            SSQR   = 0.0
            
            for i in range(len(SELE_PROB)):
        
                #SSQR += (TARGET - ENER[HIST_SELE[i]])**2.0
                SSQR += (TARGET - SELE_PROB[i])**2.0
                
            try: 
                SSQR = m.sqrt( SSQR / len(HIST_SELE)) 
            except: 
                print(SELE)
                print(HIST_SELE)
        
            SSQR_LIST.append(SSQR)    
        
        
        # Plot current selection results 

        SELE_MID = (SELE_BINS[:-1] + SELE_BINS[1:]) / 2.0
        LABEL    = "sweep " + repr(sweep+1)

        if sweep in CONDITIONS: # if ((NSWEEPS/5>0) and (sweep+1)%(NSWEEPS/5)==0):
            plt.plot(SELE_MID, SELE_PROB, marker='x', label=LABEL)

    
    TMP_SELE = SELE[:]

    if MIN_FROM_MAIN:
        TMP_SELE += [MIN_IDX]
    if MAX_FROM_MAIN:
        TMP_SELE += [MAX_IDX]    
        
    # Save resulting selection

    np.savetxt("selection.dat",np.array(TMP_SELE).astype(int),fmt='%5d')

    #print "Probabilites for final selection: "
    #print SELE_PROB

    # Plot results


    #plt.legend(loc='upper left')
    plt.legend_ = None
    plt.savefig('energy_hist.pdf')
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(list(range(1,len(SSQR_LIST)+1)), SSQR_LIST)
    plt.xscale('log')
    plt.savefig('residuals.pdf')
    plt.clf
    plt.cla
    plt.close()    

    helpers.run_bash_cmnd("mv selection.dat all.selection.dat")


################################
# D-Optimality cluster selection
################################

def _read_xyzlist(filename):
    """
    Reads a xyzlist.dat or ts_xyzlist.dat file.

    Returns a list of (n_atoms, filepath) tuples; returns [] if file is absent or empty.
    """
    clusters = []
    if not os.path.exists(filename):
        return clusters
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            n_atoms  = int(parts[0])
            filepath = parts[-1]
            clusters.append((n_atoms, filepath))
    return clusters


def _cluster_xyz_to_xyzf(src_xyz, dst_xyzf):
    """
    Converts a cluster .wrap.xyz file to ChIMES xyzf format with zero forces/energy.

    The wrap.xyz comment line is 'Lx Ly Lz' (orthorhombic box dims).
    The xyzf comment line becomes 'NON_ORTHO Lx 0 0  0 Ly 0  0 0 Lz  0 0 0 0 0 0  0'
    (box matrix, then six stress zeros, then one energy zero).
    """
    with open(src_xyz, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    box_parts = lines[1].strip().split()
    Lx, Ly, Lz = box_parts[0], box_parts[1], box_parts[2]

    box_line = ("NON_ORTHO  {Lx} 0.0 0.0  0.0 {Ly} 0.0  0.0 0.0 {Lz}"
                "  0.0 0.0 0.0 0.0 0.0 0.0  0.0\n").format(Lx=Lx, Ly=Ly, Lz=Lz)

    with open(dst_xyzf, 'w') as f:
        f.write(lines[0])
        f.write(box_line)
        for line in lines[2:2 + n_atoms]:
            parts = line.strip().split()
            f.write("{} {} {} {}  0.0 0.0 0.0\n".format(
                parts[0], parts[1], parts[2], parts[3]))


def _modify_fm_setup_for_descriptors(src_fm_setup, dst_fm_setup, n_total_frames):
    """
    Writes a modified copy of fm_setup.in suitable for descriptor-only generation.

    Changes made relative to the source:
    - TRJFILE  → MULTI candidate_traj_list.dat
    - WRAPTRJ  → false   (clusters already wrapped)
    - NFRAMES  → n_total_frames
    - FITSTRS  → false
    - FITENER  → false
    - SPLITFI  → false
    - HIERARC  → false
    """
    with open(src_fm_setup, 'r') as f:
        lines = f.readlines()

    flag_trjfile = False
    flag_wraptrj = False
    flag_nframes = False
    flag_fitstrs = False
    flag_fitener = False
    flag_splitfi = False
    flag_hierarc = False

    out = []
    for line in lines:
        if flag_trjfile:
            out.append('\tMULTI candidate_traj_list.dat\n')
            flag_trjfile = False
        elif flag_wraptrj:
            out.append('\tfalse\n')
            flag_wraptrj = False
        elif flag_nframes:
            out.append('\t{}\n'.format(n_total_frames))
            flag_nframes = False
        elif flag_fitstrs:
            out.append('\tfalse\n')
            flag_fitstrs = False
        elif flag_fitener:
            out.append('\tfalse\n')
            flag_fitener = False
        elif flag_splitfi:
            out.append('\tfalse\n')
            flag_splitfi = False
        elif flag_hierarc:
            out.append('\tfalse\n')
            flag_hierarc = False
        else:
            out.append(line)
            if   "TRJFILE" in line: flag_trjfile = True
            elif "WRAPTRJ" in line: flag_wraptrj = True
            elif "NFRAMES" in line: flag_nframes = True
            elif "FITSTRS" in line: flag_fitstrs = True
            elif "FITENER" in line: flag_fitener = True
            elif "SPLITFI" in line: flag_splitfi = True
            elif "HIERARC" in line: flag_hierarc = True

    with open(dst_fm_setup, 'w') as f:
        f.writelines(out)


def gen_subset_dopt(**kwargs):
    """
    Selects candidate clusters using D-optimality (maxvol algorithm).

    Usage: gen_subset_dopt(<arguments>)

    Algorithm:
        1. Build an atomic design matrix A_atomic from the reference GEN_FF/A.txt
           (or GEN_FF/A_comb.txt) by hstacking every three consecutive force rows
           (fx_i, fy_i, fz_i) into one row per atom.
        2. Run maxvol on A_atomic to find the D-optimal subset (indices piv) and
           compute the inverse of the square submatrix A_atomic[piv].
        3. Submit a chimes_lsq job to generate descriptors for the candidate clusters
           listed in xyzlist.dat and ts_xyzlist.dat.
        4. Build A_atomic for the candidate clusters from the resulting A.txt.
        5. Compute per-atom gamma = max row of (A_atomic_cand @ inverse_A_subset).
        6. Aggregate to per-cluster max gamma, then keep clusters with gamma in
           [gamma_min, gamma_max].
        7. Write all.xyzlist.dat and all.selection.dat; save diagnostic PDF.

    Returns:
        int: Number of clusters selected for DFT labeling.

    Notes:
        - Requires maxvolpy: pip install maxvolpy
        - Expects xyzlist.dat and ts_xyzlist.dat in the CWD (from cluster.list_clusters)
        - Expects GEN_FF/A.txt (or GEN_FF/A_comb.txt) from the current ALC's fit
        - Expects GEN_FF/fm_setup.in to exist (used as template for descriptor job)
        - All file I/O is relative to the CWD (the ALC-N directory)

    """

    ################################
    # 0. Set up argument parser
    ################################

    default_keys   = [""]*16
    default_values = [""]*16

    default_keys[0 ] = "gamma_min"     ; default_values[0 ] = 3.0
    default_keys[1 ] = "gamma_max"     ; default_values[1 ] = 10.0
    default_keys[2 ] = "amat_path"     ; default_values[2 ] = ""              # auto-detect GEN_FF/A_comb.txt or A.txt
    default_keys[3 ] = "xyzlist"       ; default_values[3 ] = "xyzlist.dat"
    default_keys[4 ] = "ts_xyzlist"    ; default_values[4 ] = "ts_xyzlist.dat"
    default_keys[5 ] = "fm_setup"      ; default_values[5 ] = "GEN_FF/fm_setup.in"
    default_keys[6 ] = "job_name"      ; default_values[6 ] = "dopt_desc"
    default_keys[7 ] = "job_nodes"     ; default_values[7 ] = "1"
    default_keys[8 ] = "job_ppn"       ; default_values[8 ] = "36"
    default_keys[9 ] = "job_walltime"  ; default_values[9 ] = "01:00:00"
    default_keys[10] = "job_queue"     ; default_values[10] = "pdebug"
    default_keys[11] = "job_account"   ; default_values[11] = ""
    default_keys[12] = "job_system"    ; default_values[12] = "slurm"
    default_keys[13] = "job_email"     ; default_values[13] = True
    default_keys[14] = "job_modules"   ; default_values[14] = ""
    default_keys[15] = "job_executable"; default_values[15] = ""              # path to chimes_lsq

    args = dict(list(zip(default_keys, default_values)))
    args.update(kwargs)

    GAMMA_MIN = float(args["gamma_min"])
    GAMMA_MAX = float(args["gamma_max"])

    print("gen_subset_dopt: gamma_min={}, gamma_max={}".format(GAMMA_MIN, GAMMA_MAX))

    ################################
    # 1. Read candidate cluster lists
    ################################

    tight_clusters = _read_xyzlist(args["xyzlist"])
    ts_clusters    = _read_xyzlist(args["ts_xyzlist"])
    all_clusters   = tight_clusters + ts_clusters

    if len(all_clusters) == 0:
        print("ERROR (gen_subset_dopt): No candidate clusters found in {} or {}.".format(
            args["xyzlist"], args["ts_xyzlist"]))
        exit()

    print("gen_subset_dopt: {} tight + {} ts = {} total candidate clusters".format(
        len(tight_clusters), len(ts_clusters), len(all_clusters)))

    ################################
    # 2. Build all.xyzlist.dat
    #    (required by populate_repo; normally created by get_repo_energies)
    ################################

    helpers.cat_specific("all.xyzlist.dat",
        [f for f in [args["xyzlist"], args["ts_xyzlist"]]
         if os.path.exists(f) and os.path.getsize(f) > 0])

    ################################
    # 3. Prepare DOPT_DESCRIPTORS work directory
    ################################

    WORK_DIR = "DOPT_DESCRIPTORS"
    helpers.run_bash_cmnd("rm -rf " + WORK_DIR)
    helpers.run_bash_cmnd("mkdir  " + WORK_DIR)

    curr_dir = helpers.run_bash_cmnd("pwd").rstrip()

    # Convert each cluster xyz → xyzf (with zero forces/energy)
    traj_list_lines = [str(len(all_clusters)) + "\n"]

    for idx, (n_atoms, src_xyz) in enumerate(all_clusters):
        tag      = "{:05d}".format(idx)
        dst_xyzf = os.path.join(WORK_DIR, "candidate_{}.xyzf".format(tag))

        src_path = src_xyz if os.path.isabs(src_xyz) else os.path.join(curr_dir, src_xyz)
        _cluster_xyz_to_xyzf(src_path, dst_xyzf)
        traj_list_lines.append("1  candidate_{}.xyzf  0\n".format(tag))

    with open(os.path.join(WORK_DIR, "candidate_traj_list.dat"), 'w') as f:
        f.writelines(traj_list_lines)

    ################################
    # 4. Write modified fm_setup.in for descriptor generation
    ################################

    _modify_fm_setup_for_descriptors(
        args["fm_setup"],
        os.path.join(WORK_DIR, "fm_setup.in"),
        len(all_clusters))

    ################################
    # 5. Submit chimes_lsq descriptor job, wait
    ################################

    os.chdir(WORK_DIR)

    job_cmd = args["job_executable"] + " fm_setup.in | tee fm_setup.log"

    job_id = helpers.create_and_launch_job(
        job_name       = args["job_name"],
        job_nodes      = str(args["job_nodes"]),
        job_ppn        = str(args["job_ppn"]),
        job_walltime   = str(args["job_walltime"]),
        job_queue      = args["job_queue"],
        job_account    = args["job_account"],
        job_system     = args["job_system"],
        job_email      = args["job_email"],
        job_executable = job_cmd,
        job_modules    = args["job_modules"],
    )

    os.chdir(curr_dir)

    helpers.wait_for_jobs(
        [job_id],
        job_system = args["job_system"],
        verbose    = True,
        job_name   = "dopt_descriptors")

    cand_amat_path = os.path.join(WORK_DIR, "A.txt")
    if not os.path.exists(cand_amat_path):
        print("ERROR (gen_subset_dopt): Descriptor job did not produce {}.".format(cand_amat_path))
        print("       Check {}/fm_setup.log for errors.".format(WORK_DIR))
        exit()

    ################################
    # 6. Load reference A matrix, build A_atomic
    ################################

    if args["amat_path"]:
        ref_amat_path = args["amat_path"]
    elif os.path.exists("GEN_FF/A_comb.txt"):
        ref_amat_path = "GEN_FF/A_comb.txt"
    else:
        ref_amat_path = "GEN_FF/A.txt"

    print("gen_subset_dopt: loading reference A matrix from", ref_amat_path)

    A_ref = np.loadtxt(ref_amat_path)

    if A_ref.ndim == 1:
        A_ref = A_ref.reshape(-1, 1)

    n_ref_rows, n_feat = A_ref.shape

    if n_ref_rows % 3 != 0:
        print("ERROR (gen_subset_dopt): Reference A matrix has {} rows, not divisible by 3.".format(n_ref_rows))
        print("       Ensure GEN_FF/A.txt is from a force-only fit.")
        exit()

    n_atoms_ref   = n_ref_rows // 3
    A_atomic_ref  = np.array([np.hstack(A_ref[3*i : 3*(i+1), :]) for i in range(n_atoms_ref)])

    print("gen_subset_dopt: A_atomic_ref shape:", A_atomic_ref.shape)
    np.savetxt("A_atomic.txt", A_atomic_ref)

    ################################
    # 7. D-optimal subset via maxvol, compute inverse
    ################################

    try:
        from maxvolpy.maxvol import maxvol
    except ImportError:
        print("ERROR (gen_subset_dopt): maxvolpy not found. Install with: pip install maxvolpy")
        exit()

    n_rows_atomic, n_cols_atomic = A_atomic_ref.shape
    if n_rows_atomic < n_cols_atomic:
        print("ERROR (gen_subset_dopt): A_atomic_ref is fat ({} rows < {} cols).".format(
            n_rows_atomic, n_cols_atomic))
        print("       maxvol requires a tall matrix. Add more training data.")
        exit()

    print("gen_subset_dopt: running maxvol...")
    piv, _ = maxvol(A_atomic_ref, 1.0)

    a_subset = A_atomic_ref[piv]   # shape: (n_cols_atomic, n_cols_atomic) — square

    try:
        inverse_a_subset = np.linalg.inv(a_subset)
    except np.linalg.LinAlgError:
        print("ERROR (gen_subset_dopt): D-optimal submatrix is singular.")
        exit()

    np.save("inverse_A_subset.npy", inverse_a_subset)
    print("gen_subset_dopt: maxvol complete. {} pivot rows selected.".format(len(piv)))

    ################################
    # 8. Load candidate A matrix, build A_atomic
    ################################

    A_cand = np.loadtxt(cand_amat_path)

    if A_cand.ndim == 1:
        A_cand = A_cand.reshape(-1, 1)

    n_cand_rows, n_feat_cand = A_cand.shape

    if n_feat_cand != n_feat:
        print("ERROR (gen_subset_dopt): Candidate A matrix has {} features but reference has {}.".format(
            n_feat_cand, n_feat))
        print("       Ensure the same fm_setup.in topology/cutoffs are used.")
        exit()

    if n_cand_rows % 3 != 0:
        print("ERROR (gen_subset_dopt): Candidate A matrix has {} rows, not divisible by 3.".format(n_cand_rows))
        exit()

    n_atoms_cand   = n_cand_rows // 3
    A_atomic_cand  = np.array([np.hstack(A_cand[3*i : 3*(i+1), :]) for i in range(n_atoms_cand)])

    print("gen_subset_dopt: A_atomic_cand shape:", A_atomic_cand.shape)

    ################################
    # 9. Compute per-atom gamma, aggregate to per-cluster max gamma
    ################################

    dot_products   = A_atomic_cand @ inverse_a_subset       # (n_atoms_cand, n_cols_atomic)
    gamma_per_atom = np.max(dot_products, axis=1)

    cluster_gamma = []
    atom_offset   = 0

    for n_atoms, _ in all_clusters:
        block = gamma_per_atom[atom_offset : atom_offset + n_atoms]
        cluster_gamma.append(float(np.max(block)))
        atom_offset += n_atoms

    cluster_gamma = np.array(cluster_gamma)

    if atom_offset != n_atoms_cand:
        print("WARNING (gen_subset_dopt): atom count mismatch: xyzlist says {} atoms but A.txt has {}.".format(
            atom_offset, n_atoms_cand))

    ################################
    # 10. Select clusters in [gamma_min, gamma_max]
    ################################

    selected = np.where((cluster_gamma >= GAMMA_MIN) & (cluster_gamma <= GAMMA_MAX))[0]

    print("gen_subset_dopt: {} / {} clusters selected (gamma in [{}, {}])".format(
        len(selected), len(all_clusters), GAMMA_MIN, GAMMA_MAX))

    if len(selected) == 0:
        print("WARNING (gen_subset_dopt): No clusters satisfy the gamma threshold. "
              "Consider adjusting DOPT_GAMMA_MIN / DOPT_GAMMA_MAX.")

    np.savetxt("all.selection.dat", selected.astype(int), fmt='%5d')
    n_selected = len(selected)

    ################################
    # 11. Diagnostic plots
    ################################

    plt.figure(figsize=(10, 6))
    plt.hist(cluster_gamma, bins=min(50, len(cluster_gamma)),
             alpha=0.7, color='steelblue', label='Cluster max gamma')
    plt.axvline(x=GAMMA_MIN, color='green',  linestyle='--', label='gamma_min = {}'.format(GAMMA_MIN))
    plt.axvline(x=GAMMA_MAX, color='red',    linestyle='--', label='gamma_max = {}'.format(GAMMA_MAX))
    plt.xlabel('Max Gamma per Cluster')
    plt.ylabel('Count')
    plt.title('D-Optimality Cluster Selection: Gamma Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig('dopt_gamma_dist.pdf')
    plt.clf()
    plt.cla()
    plt.close()

    np.savetxt("dopt_cluster_gamma.txt", cluster_gamma)

    print("gen_subset_dopt: wrote all.selection.dat, dopt_gamma_dist.pdf, dopt_cluster_gamma.txt")

    return n_selected


def finish_dopt_al_convergence(restart_controller, THIS_ALC, email_add, driver_dir):
    """
    Gracefully terminate active learning when D-opt finds no clusters above
    the gamma threshold. Skips QM labeling for the current cycle and records
    convergence in restart.dat so later driver invocations do not continue.
    """

    print("")
    print("=" * 72)
    print("D-optimality convergence reached at ALC-{}".format(THIS_ALC))
    print("No candidate clusters exceed the gamma threshold.")
    print("Skipping QM calculations and stopping active learning.")
    print("=" * 72)
    print("")

    restart_controller.DOPT_AL_COMPLETE = True
    restart_controller.update_file("DOPT_AL_COMPLETE: TRUE" + '\n')
    restart_controller.update_file("THIS_ALC: COMPLETE" + '\n')

    os.chdir("..")

    print("ALC-", THIS_ALC, "stopped: D-opt convergence (no clusters above gamma threshold)")

    helpers.email_user(
        driver_dir,
        email_add,
        "ALC-" + str(THIS_ALC) + " status: D-opt convergence — active learning complete",
    )

    return True








    
    
    
    
