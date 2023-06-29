#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#  Copyright (C) 2016-2019 Ambroise van Roekeghem <ambroise.vanroekeghem@gmail.com>
#  Copyright (C) 2016-2019 Jesús Carrete Montaña <jcarrete@gmail.com>
#  Copyright (C) 2016-2019 Natalio Mingo Bisquert <natalio.mingo@cea.fr>
#
#  This file is part of qSCAILD.
#
#  qSCAILD is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  qSCAILD is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with qSCAILD.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import os.path
import shutil
import subprocess
import glob
import pprint
import datetime
import logging
import actions
import time
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def str2bool(v):
    return v.lower().strip() in ("yes", "true", "t", "1")


if (not os.path.isfile("POTCAR")) or (not os.path.isfile("INCAR")) or (
        not os.path.isfile("POSCAR")) or (not os.path.isfile("SPOSCAR")):
    with open("finished", "w") as file:
        file.write("finished: error\n")
    print("some file is missing: check POSCAR, SPOSCAR, POTCAR, INCAR")
    comm.Abort()
    sys.exit(1)

#Temperature in K
T = 300.
#Number of configurations for each cycle
nconf = 10
#Number of cycles
nfits = 20
#Size of the supercell
n0 = 3
n1 = 3
n2 = 3
#Cutoff for the third order force constants
cutoff = -3
#If third order force constants are calculated
third = False
#If pressure is computed
use_pressure = "False"
#Values of the pressure on the diagonal of the tensor
pressure = np.array([0., 0., 0.])
#If atomic positions are optimized
optimize_positions = False
#Whether small displacements are used
use_smalldisp = False
#Whether symmetries are computed
calc_symm = True
#Whether symmetries include acoustic sum rule
symm_acoustic = True
# Value of the frequency to which imaginary modes are put, in THz.
# -1 means that the frequencies are "reflected" from negative to positive.
# All frequencies smaller than 1e-4 are ignored, so 0 would just ignore all
# imaginary frequencies.
imaginary_freq = 1.0
# Empirically enforce the acoustic sum rule by adding a translated supercell
# with zero force (not advised, prefer using symmetries).
enforce_acoustic = False
# Grid for the calculation of the mean thermal displacement matrix and of the
# gruneisen parameters
grid = 9
# Tolerance for the convergence criterion (maximum absolute difference in the
# force constants)
tolerance = 1e-2
# Pressure tolerance for the convergence criterion (maximum absolute difference
# in the stress tensor in kbar)
pdiff = 1.0
# Memory factor to accumulate configurations: all configurations starting from
# floor(iteration*(1.0-memory)) are taken into account in the fit
memory = 0.3
# Mixing between fcs in differents iterations
mixing = 0.

# Read input file
if rank == 0:
    with open("parameters", 'r') as f:
        for line in f.readlines():
            if 'T_K' in line:
                T = float(line.split("=")[1])
            if 'nconf' in line:
                nconf = int(line.split("=")[1])
            if 'nfits' in line:
                nfits = int(line.split("=")[1])
            if 'n0 ' in line:
                n0 = int(line.split("=")[1])
            if 'n1 ' in line:
                n1 = int(line.split("=")[1])
            if 'n2 ' in line:
                n2 = int(line.split("=")[1])
            if 'cutoff' in line:
                cutoff = float(line.split("=")[1])
            if 'third' in line:
                third = str2bool(line.split("=")[1])
            if 'use_pressure' in line:
                use_pressure = line.split("=")[1].strip()
            if 'pressure_diag' in line:
                p = line.split("=")[1].split(',')
                pressure = np.array([float(p[0]), float(p[1]), float(p[2])])
            if 'optimize_positions' in line:
                optimize_positions = str2bool(line.split("=")[1].strip())
            if 'use_smalldisp' in line:
                use_smalldisp = str2bool(line.split("=")[1])
            if 'calc_symm' in line:
                calc_symm = str2bool(line.split("=")[1])
            if 'symm_acoustic' in line:
                symm_acoustic = str2bool(line.split("=")[1])
            if 'imaginary_freq' in line:
                imaginary_freq = float(line.split("=")[1])
            if 'tolerance' in line:
                tolerance = float(line.split("=")[1])
            if 'pdiff' in line:
                pdiff = float(line.split("=")[1])
            if 'memory' in line:
                memory = float(line.split("=")[1])
            if 'enforce_acoustic' in line:
                enforce_acoustic = str2bool(line.split("=")[1])
            if 'mixing' in line:
                mixing = float(line.split("=")[1])
            if 'grid' in line:
                grid = int(line.split("=")[1])
    print("T = " + str(T) + " K")
    print("nconf = " + str(nconf))
    print("nfits = " + str(nfits))
    print("supercell = " + str(n0) + " " + str(n1) + " " + str(n2))
    print("cutoff = " + str(cutoff))
    print("third order = " + str(third))
    print("use pressure = " + use_pressure)
    print("pressure = " + str(pressure))
    print("use small displacements = " + str(use_smalldisp))
    print("calculate symmetries = " + str(calc_symm))
    print("symmetries enforcing acoustic sum rule= " + str(symm_acoustic))
    print("imaginary frequencies are set to " + str(imaginary_freq) + " THz")
    print("absolute tolerance in fcs for convergence = " + str(tolerance))
    print("pressure tolerance in kbar for convergence = " + str(pdiff))
    print("memory factor to accumulate configurations = " + str(memory))
    print("empirically enforce acoustic sum rule = " + str(enforce_acoustic))
    print("grid for the calculation of the phonon quantities = " + str(grid))
    print("mixing = " + str(mixing))
    print("optimize positions = " + str(optimize_positions))
T = comm.bcast(T, root=0)
nconf = comm.bcast(nconf, root=0)
nfits = comm.bcast(nfits, root=0)
n0 = comm.bcast(n0, root=0)
n1 = comm.bcast(n1, root=0)
n2 = comm.bcast(n2, root=0)
cutoff = comm.bcast(cutoff, root=0)
third = comm.bcast(third, root=0)
use_pressure = comm.bcast(use_pressure, root=0)
pressure = comm.bcast(pressure, root=0)
optimize_positions = comm.bcast(optimize_positions, root=0)
use_smalldisp = comm.bcast(use_smalldisp, root=0)
calc_symm = comm.bcast(calc_symm, root=0)
symm_acoustic = comm.bcast(symm_acoustic, root=0)
imaginary_freq = comm.bcast(imaginary_freq, root=0)
tolerance = comm.bcast(tolerance, root=0)
pdiff = comm.bcast(pdiff, root=0)
memory = comm.bcast(memory, root=0)
enforce_acoustic = comm.bcast(enforce_acoustic, root=0)
grid = comm.bcast(grid, root=0)
mixing = comm.bcast(mixing, root=0)

n = [n0, n1, n2]

if (not os.path.isfile("FORCE_CONSTANTS")) and (not use_smalldisp):
    with open("finished", "w") as file:
        file.write("finished: error\n")
    print("some file is missing: check FORCE_CONSTANTS")
    comm.Abort()
    sys.exit(1)

os.sync()
comm.Barrier()

actions.fit_force_constants(nconf, nfits, T, n, cutoff, third, use_pressure,
                            pressure, optimize_positions, use_smalldisp, calc_symm, symm_acoustic,
                            imaginary_freq, enforce_acoustic, grid, tolerance,
                            pdiff, memory, mixing)
sys.exit(0)
