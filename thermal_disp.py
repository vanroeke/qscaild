#!/usr/bin/env python
# -*- encoding:utf-8 -*-
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
import os.path
import sys
import json
import itertools
import multiprocessing
import binascii

import numpy as np
import scipy as sp
import scipy.constants as codata

import phonopy
import phonopy.interface
import phonopy.file_IO
import io
import hashlib
import subprocess
import generate_conf
from mpi4py import MPI


def harmonic_free_energy(f, T):
    """
    Return the free energy depending of T, force constants and structure
    """

    f = f[f > 1.e-4]
    print("frequencies used for the calculation of harmonic free energy", f)
    free_ener_har = np.sum(
        0.5 * codata.h * f * 1e12 +
        codata.k * T * np.log1p(-np.exp(-codata.h * f * 1e12 / codata.k / T)))

    free_ener_har /= 1.60217657 * 1e-19  ##put it in eV
    print("harmonic free energy in eV: ", free_ener_har)

    return free_ener_har


def fBE(f, T):
    """
    Bose-Einstein distribution with mu=0.. f is expected to be in THz,
    T in K.
    """
    if (T == 0.):
        return 0.
    elif (codata.h * abs(f) * 1e12 / (codata.k * T) < 2):
        return 1. / np.expm1(codata.h * abs(f) * 1e12 / (codata.k * T))
    else:
        return -np.exp(-codata.h * abs(f) * 1e12 / (codata.k * T)) / np.expm1(
            -codata.h * abs(f) * 1e12 / (codata.k * T))


def write_displacement_matrix(poscar_file, fcs_file, T, n, use_smalldisp,
                              imaginary_freq, grid):
    """
    Write the displacement matrix depending of T, force constants and
    structure, integrated over the Brillouin zone.
    """

    SYMPREC = 1e-5
    NQ = grid

    if not os.path.isfile(poscar_file):
        sys.exit("The specified POSCAR file does not exist.")
    na, nb, nc = [int(i) for i in n]
    if min(na, nb, nc) < 1:
        sys.exit("All dimensions must be positive integers")

    poscar = generate_conf.read_POSCAR(poscar_file)
    natoms = poscar["numbers"].sum()
    ntot = na * nb * nc * natoms
    nmodes = 3 * natoms
    ncells = na * nb * nc
    matrix = np.zeros((ncells * nmodes, ncells * nmodes), dtype=np.complex128)
    local_matrix = np.zeros((ncells * nmodes, ncells * nmodes),
                            dtype=np.complex128)

    if use_smalldisp:
        for idiag in range(ncells * nmodes):
            matrix[idiag, idiag] = 0.000001
        return matrix

    if not os.path.isfile(fcs_file):
        sys.exit("The specified FORCE_CONSTANTS file does not exist.")

    supercell_matrix = np.diag([na, nb, nc])

    structure = phonopy.interface.read_crystal_structure(poscar_file,
                                                         "vasp")[0]
    fc = phonopy.file_IO.parse_FORCE_CONSTANTS(fcs_file)

    phonon = phonopy.Phonopy(
        structure,
        supercell_matrix,
        primitive_matrix=None,
        factor=phonopy.units.VaspToTHz,
        dynamical_matrix_decimals=None,
        force_constants_decimals=None,
        symprec=SYMPREC,
        is_symmetry=True,
        log_level=0)
    if os.path.isfile("BORN"):
        nac_params = phonopy.file_IO.get_born_parameters(
            open("BORN"), phonon.get_primitive(),
            phonon.get_primitive_symmetry())
        phonon.set_nac_params(nac_params=nac_params)

    phonon.set_force_constants(fc)

    masses = phonon.get_supercell().get_masses(
    ) * codata.physical_constants["atomic mass constant"][0]

    qpoints = create_mesh(NQ)
    nqpoints = qpoints.shape[0]

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD  # get MPI communicator object
    size = comm.size  # total number of processes
    rank = comm.rank  # rank of this process

    full_qlist = list(range(nqpoints))
    qlist = full_qlist[rank::size]

    for qpt in qlist:
        local_matrix += qpoint_worker((qpoints[qpt, :], phonon, T, na, nb, nc,
                                       imaginary_freq, poscar["positions"]))

    # Reduce all qpoints
    comm.Reduce(local_matrix, matrix, op=MPI.SUM, root=0)

    if rank == 0:
        matrix *= codata.hbar / 2. / nqpoints / 2. / np.pi / 1e12  # m**2
        for i, j in itertools.product(
                range(ncells * nmodes), range(ncells * nmodes)):
            matrix[i, j] /= np.sqrt(masses[i // 3] * masses[j // 3])

        matrix = 1e18 * matrix.real  # nm**2
    comm.Barrier()
    return matrix


def write_displacement_matrix_gamma(sposcar_file, fcs_file, T, n,
                                    use_smalldisp, imaginary_freq):
    """
    Writes the displacement matrix depending of T, force constants and
    structure, using only the gamma-point of the supercell.
    """

    SYMPREC = 1e-5

    if not os.path.isfile(sposcar_file):
        sys.exit("The specified POSCAR file does not exist.")
    na, nb, nc = [1, 1, 1]

    poscar = generate_conf.read_POSCAR(sposcar_file)
    natoms = poscar["numbers"].sum()
    ntot = na * nb * nc * natoms
    nmodes = 3 * natoms
    ncells = na * nb * nc
    matrix = np.zeros((ncells * nmodes, ncells * nmodes), dtype=np.complex128)

    if use_smalldisp:
        for idiag in range(ncells * nmodes):
            matrix[idiag, idiag] = 0.000001
        return matrix

    if not os.path.isfile(fcs_file):
        sys.exit("The specified FORCE_CONSTANTS file does not exist.")

    supercell_matrix = np.diag([na, nb, nc])

    structure = phonopy.interface.read_crystal_structure(sposcar_file,
                                                         "vasp")[0]
    fc = phonopy.file_IO.parse_FORCE_CONSTANTS(fcs_file)

    phonon = phonopy.Phonopy(
        structure,
        supercell_matrix,
        primitive_matrix=None,
        factor=phonopy.units.VaspToTHz,
        dynamical_matrix_decimals=None,
        force_constants_decimals=None,
        symprec=SYMPREC,
        is_symmetry=True,
        log_level=0)
    if os.path.isfile("BORN"):
        nac_params = phonopy.file_IO.get_born_parameters(
            open("BORN"), phonon.get_primitive(),
            phonon.get_primitive_symmetry())
        phonon.set_nac_params(nac_params=nac_params)

    phonon.set_force_constants(fc)

    masses = phonon.get_supercell().get_masses(
    ) * codata.physical_constants["atomic mass constant"][0]

    nqpoints = 1
    print("nqpoints: " + repr(nqpoints))

    matrix = qpoint_worker(([0., 0., 0.], phonon, T, na, nb, nc,
                            imaginary_freq, poscar["positions"]))

    matrix *= codata.hbar / 2. / nqpoints / 2. / np.pi / 1e12  # m**2
    for i, j in itertools.product(
            range(ncells * nmodes), range(ncells * nmodes)):
        matrix[i, j] /= np.sqrt(masses[i // 3] * masses[j // 3])

    matrix = 1e18 * matrix.real  # nm**2
    return matrix


def qpoint_worker(args):
    """
    Computes the unnormalized contribution to the thermal
    displacement matrix from a single q point.
    """
    q, phonon, T, na, nb, nc, imaginary_freq, positions = args
    natoms = phonon.get_primitive().get_number_of_atoms()
    nmodes = 3 * natoms
    ncells = na * nb * nc
    matrix = np.zeros((ncells * nmodes, ncells * nmodes), dtype=np.complex128)
    f, psi = phonon.get_frequencies_with_eigenvectors(q)

    for ifreq in range(len(f)):
        if (f[ifreq] < -1.e-4):
            print("ATTENTION: IMAGINARY FREQUENCIES ->"
                  " CONVERTED TO POSITIVE VALUE")
            if imaginary_freq == -1:
                f[ifreq] = abs(f[ifreq])
            else:
                f[ifreq] = imaginary_freq
    n_B = np.array([fBE(fq, T) for fq in f])
    r = range(nmodes)
    factors = np.empty(ncells, dtype=np.complex128)
    for pos, (k, j, i) in enumerate(
            itertools.product(range(nc), range(nb), range(na))):
        factors[pos] = np.exp(2j * np.pi * np.dot(q, [i, j, k]))
    for im in r:
        if (f[im] > 1.e-4):
            lpsi = np.empty(ncells * nmodes, dtype=np.complex128)
            for ia in range(natoms):
                for pos, (k, j, i) in enumerate(
                        itertools.product(range(nc), range(nb), range(na))):
                    for ic in range(3):
                        jm = np.ravel_multi_index((ia, ic), (natoms, 3))
                        li = np.ravel_multi_index((ia, k, j, i, ic),
                                                  (natoms, nc, nb, na, 3))
                        lpsi[li] = psi[jm, im] * factors[pos] * np.exp(
                            2j * np.pi * np.dot(q, positions[:, ia]))
            matrix += np.outer(lpsi, lpsi.conj()) * (1. + 2. * n_B[im]) / f[im]
    return matrix


def create_mesh(nq):
    """
    Create a MP mesh with nq q points along each
    direction. Return the direct coordinates of each point.
    """
    nruter = []
    for i, j, k in itertools.product(range(nq), range(nq), range(nq)):
        nruter.append([float(i) / nq, float(j) / nq, float(k) / nq])
    nruter = np.array(nruter)
    center = np.mean(nruter, axis=0)
    for i in range(nruter.shape[0]):
        nruter[i, :] -= center
    return nruter
