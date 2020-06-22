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
import copy
import json
import itertools
import io
import hashlib
import sqlite3
import logging
import glob
import shutil
import time

import numpy as np
import scipy as sp
import scipy.linalg
import scipy.stats
import scipy.constants as codata

import phonopy
import thermal_disp
import gradient

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def read_POSCAR(filename):
    """
    Return all the relevant information contained in a POSCAR file.
    """
    nruter = dict()
    nruter["lattvec"] = np.empty((3, 3))
    f = open(filename, "r")
    firstline = next(f)
    factor = .1 * float(next(f).strip())
    for i in range(3):
        nruter["lattvec"][:, i] = [float(j) for j in next(f).split()]
    nruter["lattvec"] *= factor
    line = next(f)
    fields = next(f).split()
    old = False
    try:
        int(fields[0])
    except ValueError:
        old = True
    if old:
        nruter["elements"] = firstline.split()
        nruter["numbers"] = np.array([int(i) for i in line.split()])
        typeline = "".join(fields)
    else:
        nruter["elements"] = line.split()
        nruter["numbers"] = np.array([int(i) for i in fields], dtype=np.intc)
        typeline = next(f)
    natoms = nruter["numbers"].sum()
    nruter["positions"] = np.empty((3, natoms))
    for i in range(natoms):
        nruter["positions"][:, i] = [float(j) for j in next(f).split()]
    f.close()
    nruter["types"] = []
    for i in range(len(nruter["numbers"])):
        nruter["types"] += [i] * nruter["numbers"][i]
    if typeline[0] == "C":
        nruter["positions"] = sp.linalg.solve(nruter["lattvec"],
                                              nruter["positions"] * factor)
    return nruter


def distort_POSCAR(poscar, distortions):
    """
    Return a modified POSCAR dictionary with its coordinates displaced
    according to distortions.
    """
    cd = distortions.reshape((-1, 3))
    dd = sp.linalg.solve(poscar["lattvec"], cd.T)
    nruter = copy.deepcopy(poscar)
    nruter["positions"] += dd
    return nruter


def write_POSCAR(poscar, filename):
    """
    Write the contents of poscar to filename.
    """
    f = io.StringIO()
    f.write("1.0\n")
    for i in range(3):
        f.write("{0[0]:>20.15f} {0[1]:>20.15f} {0[2]:>20.15f}\n".format(
            (poscar["lattvec"][:, i] * 10.).tolist()))
    f.write("{0}\n".format(" ".join(poscar["elements"])))
    f.write("{0}\n".format(" ".join([str(i) for i in poscar["numbers"]])))
    f.write("Direct\n")
    for i in range(poscar["positions"].shape[1]):
        f.write("{0[0]:>20.15f} {0[1]:>20.15f} {0[2]:>20.15f}\n".format(
            poscar["positions"][:, i].tolist()))
    header = hashlib.sha1(f.getvalue().encode('utf-8')).hexdigest()
    with open(filename, "w") as finalf:
        finalf.write("{0}\n".format(header))
        finalf.write(f.getvalue())
    f.close()


def generate(nconfig, iteration, poscar_file, sposcar_file, fcs_file, T, n,
             use_smalldisp, imaginary_freq, grid):
    """
    Generate displaced configurations.
    """

    # The thermal displacement matrix is sampled in the whole Brillouin zone
    # with the chosen grid. If grid = 0, it is sampled only at the gamma point
    # of the supercell.
    if grid > 0:
        matrix_config = thermal_disp.write_displacement_matrix(
            poscar_file, fcs_file, T, n, use_smalldisp, imaginary_freq, grid)

    if rank == 0:

        if grid == 0:
            matrix_config = thermal_disp.write_displacement_matrix_gamma(
                sposcar_file, fcs_file, T, n, use_smalldisp, imaginary_freq)

        cov = np.array(matrix_config)
        distr = sp.stats.multivariate_normal(cov=cov, allow_singular=True)
        displacements = distr.rvs(nconfig)

        conn = sqlite3.connect("QSCAILD.db")
        cur = conn.cursor()

        mean = 0.
        absmean = 0.
        absmean_full = 0.

        cur.execute("""SELECT MAX(id) FROM configurations""")
        already_calc = cur.fetchall()[0][0]
        if already_calc is None:
            already_calc = 0

        conn.commit()

        sposcar = read_POSCAR(sposcar_file)
        shutil.copy(poscar_file, "POSCAR_" + str(iteration))
        shutil.copy(sposcar_file, "SPOSCAR_" + str(iteration))

        for isample in range(nconfig):
            mean += displacements[isample, :]
            absmean += sum(abs(displacements[isample, :])) / len(
                displacements[isample, :])
            absmean_full += abs(displacements[isample, :])
            cur.execute(
                "INSERT INTO configurations VALUES "
                "(?,?,?,?,?,null,null,null,null)",
                (isample + already_calc + 1, iteration,
                 json.dumps(displacements[isample, :].tolist()),
                 distr.logpdf(displacements[isample, :]),
                 distr.logpdf(displacements[isample, :])))

            newposcar = distort_POSCAR(sposcar, displacements[isample, :])
            filename = "SPOSCAR.config.{0}".format(isample + already_calc + 1)
            write_POSCAR(newposcar, filename)
            print(filename, "written", flush=True)

        mean = mean / nconfig
        absmean = absmean / nconfig
        absmean_full = absmean_full / nconfig
        with open("out_disp", 'a') as file:
            file.write("iteration: " + str(iteration) + "\n")
            file.write("mean absolute displacement in nm: " +
                       str(absmean.tolist()) + "\n")
            file.write("mean absolute displacement in nm for every atom: " +
                       str(absmean_full.tolist()) + "\n")
            file.write("mean displacement in nm for every atom: " +
                       str(mean.tolist()) + "\n")

        conn.commit()

        # Compute the current probability of old configurations for reweighting
        # purpose
        cur.execute(
            "SELECT id, iteration, displacements"
            " FROM configurations WHERE iteration <?", (iteration, ))
        config = cur.fetchall()
        conn.commit()
        for c in config:
            sposcar_old = read_POSCAR("SPOSCAR_" + str(c[1]))
            newdisp = np.array(json.loads(c[2])) + np.ravel(
                np.dot(sposcar_old["lattvec"], sposcar_old["positions"]) -
                np.dot(sposcar["lattvec"], sposcar["positions"]))
            cur.execute(
                "UPDATE configurations SET current_proba = ? WHERE id = ?",
                (distr.logpdf(newdisp), c[0]))

        conn.commit()
        conn.close()
    return


def prepare_conf(nconfig, iteration, poscar_file, sposcar_file, fcs_file, T, n,
                 use_smalldisp, imaginary_freq, grid):
    """
    Prepare a set of directories to calculate the forces and energy of the
    configurations. Return a tuple with their names.
    """

    generate(nconfig, iteration, poscar_file, sposcar_file, fcs_file, T, n,
             use_smalldisp, imaginary_freq, grid)

    if rank == 0:

        results = sorted(glob.glob("SPOSCAR.config.*"))
        nruter = []
        for r in results:
            postfix = r.split(".")[2]
            dirname = "config-" + postfix
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            shutil.copy("POTCAR", dirname)
            shutil.copy("INCAR", dirname)
            shutil.copy("KPOINTS", dirname)
            shutil.move(r, os.path.join(dirname, "POSCAR"))
            os.sync()
            nruter.append(os.path.abspath(dirname))
    else:
        nruter = None
    comm.Barrier()
    os.sync()

    nruter = comm.bcast(nruter, root=0)
    return nruter
