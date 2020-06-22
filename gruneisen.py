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
import shutil
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
import symmetry
import gradient
import thirdorder_common
import thirdorder_save
from sklearn import linear_model
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def read_3rd_fcs_asinfile(fcs_file, poscar_file):
    """
    Read a set of anharmonic force constants from a file.
    """
    poscar = generate_conf.read_POSCAR(poscar_file)
    lattvec = poscar["lattvec"] * 10.
    fcs_3rd = []
    atom_info = []
    R_j = []
    R_k = []
    abc = []
    with open(fcs_file, "r") as f:
        nblocks = int(next(f).strip())
        tmp = np.empty((3, 2))
        for iblock in range(nblocks):
            next(f)
            next(f)
            tmp[:, 0] = [float(i) for i in next(f).split()]
            tmp[:, 1] = [float(i) for i in next(f).split()]
            Rj = np.round(sp.linalg.solve(lattvec, tmp[:, 0]))
            Rk = np.round(sp.linalg.solve(lattvec, tmp[:, 1]))
            ijk = [int(i) - 1 for i in next(f).split()]
            for il in range(27):
                fields = next(f).split()
                c1, c2, c3 = [int(i) - 1 for i in fields[:3]]
                atom_info.append(ijk)
                R_j.append(Rj)
                R_k.append(Rk)
                abc.append([c1, c2, c3])
                fcs_3rd.append(float(fields[3]))
    return np.array(fcs_3rd), np.array(atom_info), np.array(R_j), np.array(
        R_k), np.array(abc)


def mode_gruneisen(f, psii, psij, massesi, massesj, cartesian_positions,
                   fcs_3rd, factorj):
    """
    Computes the value of the gruneisen parameter for a given mode
    """

    sumijk = np.sum(
        (fcs_3rd * cartesian_positions * factorj * psii * psij / np.sqrt(
            massesi * massesj)).reshape(-1, 3),
        axis=0)
    mode_gruneisen = -(
        sumijk * 1.e-24 * 1.e20 * 1.6e-19 / 2. / 4. / np.pi / np.pi / f / f /
        codata.physical_constants["atomic mass constant"][0])
    return mode_gruneisen


def har_cv(f, T):
    """
    Return the specific heat depending of T and frequency (f in THz, T in K)
    """

    if (T == 0.):
        return 1.
    else:
        x = codata.h * f * 1e12 / (2. * codata.k * T)
        return codata.k * x**2 / np.sinh(x)**2


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


def write_mode_gruneisen(poscar_file, n, fcs_file, fcs_3rd_file,
                         imaginary_freq, grid, filename):
    """
    Write the gruneisen parameters depending of T, force constants and
    structure, in the whole Brillouin zone.
    """
    SYMPREC = 1e-5

    if not os.path.isfile(poscar_file):
        sys.exit("Gruneisen: the specified POSCAR file does not exist.")

    poscar = generate_conf.read_POSCAR(poscar_file)
    ncells = n[0] * n[1] * n[2]
    natoms = poscar["numbers"].sum()
    nmodes = 3 * natoms

    supercell_matrix = np.diag([n[0], n[1], n[2]])

    if not os.path.isfile(fcs_file):
        sys.exit("The specified FORCE_CONSTANTS file does not exist.")
    if not os.path.isfile(fcs_3rd_file):
        sys.exit("The specified FORCE_CONSTANTS_3RD file does not exist.")

    fcs_3rd, atom_info, R_j, R_k, abc = read_3rd_fcs_asinfile(
        fcs_3rd_file, poscar_file)
    nblocks = len(fcs_3rd)
    poscar_positions = sp.dot(poscar["lattvec"], poscar["positions"]).T * 10.
    cartesian_positions = np.array([
        sp.dot(poscar["lattvec"], R_k[iblock])[abc[iblock, 2]] * 10. +
        poscar_positions[atom_info[iblock, 2], abc[iblock, 2]]
        for iblock in range(nblocks)
    ])

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

    masses = phonon.get_unitcell().get_masses()
    massesi = masses[atom_info[:, 0]]
    massesj = masses[atom_info[:, 1]]

    NQ = grid
    mesh = create_mesh(NQ)
    nqpoints = mesh.shape[0]

    m_grun = []
    f_grun = []
    r = range(nmodes)

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD  # get MPI communicator object
    size = comm.size  # total number of processes
    rank = comm.rank  # rank of this process

    # Scatter qpoints across cores.
    full_qlist = list(range(nqpoints))
    qlist = full_qlist[rank::size]

    for qpt in qlist:
        q = mesh[qpt, :]
        f, psi = phonon.get_frequencies_with_eigenvectors(q)
        factor = np.exp(2j * np.pi * np.dot(q, poscar["positions"]))
        factorj = np.exp(2j * np.pi * np.dot(R_j, q))
        for im in r:
            if (f[im] < -1.e-4):
                print("ATTENTION: IMAGINARY FREQUENCIES ->"
                      " CONVERTED TO POSITIVE VALUE")
                if imaginary_freq == -1:
                    f[im] = abs(f[im])
                else:
                    f[im] = imaginary_freq
            if (f[im] > 1.e-4):
                lpsi = psi[:, im].reshape(-1, 3) * factor[:, np.newaxis]
                psii = np.conj(
                    np.array([
                        lpsi[atom_info[iblock, 0], abc[iblock, 0]]
                        for iblock in range(nblocks)
                    ]))
                psij = np.array([
                    lpsi[atom_info[iblock, 1], abc[iblock, 1]]
                    for iblock in range(nblocks)
                ])
                m_gruni = mode_gruneisen(f[im], psii, psij, massesi, massesj,
                                         cartesian_positions, fcs_3rd,
                                         factorj).real
                m_grun.append(m_gruni)
                f_grun.append(f[im])
            else:
                m_grun.append(np.array([0., 0., 0.]))
                f_grun.append(f[im])
    m_grun = np.array(m_grun)
    f_grun = np.array(f_grun)

    # Gather all qpoints
    m_grun = comm.gather(m_grun, root=0)
    f_grun = comm.gather(f_grun, root=0)

    if rank == 0:
        m_grun = np.concatenate(m_grun)
        f_grun = np.concatenate(f_grun)
        with open(filename, 'w') as f:
            for i in range(f_grun.shape[0]):
                f.write(
                    str(f_grun[i]) + '  ' + str(m_grun[i][0]) + '  ' +
                    str(m_grun[i][1]) + '  ' + str(m_grun[i][2]) + '\n')

    m_grun = comm.bcast(m_grun, root=0)
    f_grun = comm.bcast(f_grun, root=0)

    return f_grun, m_grun


def write_weighted_gruneisen(f_grun, m_grun, Tlist, filename):
    w_grun = []
    for T in Tlist:
        cv = np.array([har_cv(f, T) for f in f_grun])
        w_grun.append(
            [T, np.sum(cv[:, np.newaxis] * m_grun, axis=0) / np.sum(cv)])
    with open(filename, 'w') as f:
        for T, wgruni in w_grun:
            f.write(
                str(T) + '  ' + str(wgruni[0]) + '  ' + str(wgruni[1]) + '  ' +
                str(wgruni[2]) + '\n')
    return w_grun


###BEWARE: NOT TESTED YET AND PROBABLY NOT WORKING
def write_mode_gruneisen_gamma(poscar_file, sposcar_file, n, fcs_file,
                               fcs_3rd_file, imaginary_freq, filename):
    """
    Write the gruneisen parameters depending of T, force constants and
    structure, using only the gamma point of the supercell.
    """

    SYMPREC = 1e-5
    NQ = 1

    if not os.path.isfile(poscar_file):
        sys.exit("Gruneisen: the specified POSCAR file does not exist.")

    poscar = generate_conf.read_POSCAR(poscar_file)
    sposcar = generate_conf.read_POSCAR(sposcar_file)
    corresp = symmetry.calc_corresp(poscar, sposcar, n)
    ncells = n[0] * n[1] * n[2]
    natoms = poscar["numbers"].sum()
    nmodes = 3 * natoms * ncells

    supercell_matrix = np.diag([1, 1, 1])

    if not os.path.isfile(fcs_file):
        sys.exit("The specified FORCE_CONSTANTS file does not exist.")
    if not os.path.isfile(fcs_3rd_file):
        sys.exit("The specified FORCE_CONSTANTS_3RD file does not exist.")

    fcs_3rd, atom_info, R_j, R_k, abc = read_3rd_fcs_asinfile(
        fcs_3rd_file, poscar_file)
    nblocks = len(fcs_3rd)
    poscar_positions = sp.dot(poscar["lattvec"], poscar["positions"]).T * 10.
    cartesian_positions = np.array([
        sp.dot(poscar["lattvec"], R_k[iblock])[abc[iblock, 2]] * 10. +
        poscar_positions[atom_info[iblock, 2], abc[iblock, 2]]
        for iblock in range(nblocks)
    ])

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

    masses = phonon.get_unitcell().get_masses()
    massesi = np.array([
        masses[corresp.index([atom_info[iblock, 0], 0, 0, 0])]
        for iblock in range(nblocks)
    ])
    massesj = np.array([
        masses[corresp.index([atom_info[iblock, 1], 0, 0, 0])]
        for iblock in range(nblocks)
    ])

    print("nqpoints: " + str(NQ))
    mesh = [[0., 0., 0.]]

    w_grun = []
    m_grun = []
    f_grun = []
    tot_cv = 0.
    r = range(nmodes)
    for q in mesh:
        f, psi = phonon.get_frequencies_with_eigenvectors(q)
        factorj = 1.0
        for im in r:
            if (f[im] < -1.e-4):
                print("ATTENTION: IMAGINARY FREQUENCIES ->"
                      " CONVERTED TO POSITIVE VALUE")
                if imaginary_freq == -1:
                    f[im] = abs(f[im])
                else:
                    f[im] = imaginary_freq
            if (f[im] > 1.e-4):
                lpsi = psi[:, im].reshape(-1, 3)
                psii = np.conj(
                    np.array([
                        lpsi[corresp.index([atom_info[iblock, 0], 0, 0, 0]
                                           ), abc[iblock, 0]]
                        for iblock in range(nblocks)
                    ]))
                psij = np.array([
                    lpsi[corresp.index([
                        atom_info[iblock, 1], R_j[iblock, 0] %
                        n[0], R_j[iblock, 1] % n[1], R_j[iblock, 2] % n[2]
                    ]), abc[iblock, 1]] for iblock in range(nblocks)
                ])
                m_gruni = mode_gruneisen(f[im], psii, psij, massesi, massesj,
                                         cartesian_positions, fcs_3rd,
                                         factorj).real * ncells
                m_grun.append(m_gruni)
                f_grun.append(f[im])
            else:
                m_grun.append(np.array([0., 0., 0.]))
                f_grun.append(f[im])

    with open(filename, 'w') as f:
        for i in range(len(f_grun)):
            f.write(
                str(f_grun[i]) + '  ' + str(m_grun[i][0]) + '  ' +
                str(m_grun[i][1]) + '  ' + str(m_grun[i][2]) + '\n')

    m_grun = np.array(m_grun)
    f_grun = np.array(f_grun)
    return f_grun, m_grun


#Returns effective pressure in kbar
def calc_gruneisen_pressure(f_grun, m_grun, T, poscar_file):
    """
    Return the pressure estimated from the Gruneisen parameters.
    """
    poscar = generate_conf.read_POSCAR(poscar_file)
    nBE = np.array([fBE(f, T) + 0.5 for f in f_grun])
    nmodes = poscar["numbers"].sum() * 3.
    volume = abs(np.linalg.det(poscar["lattvec"] * 10.))
    return -nmodes * np.mean(
        codata.h * f_grun[:, np.newaxis] * 1e12 * m_grun * nBE[:, np.newaxis],
        axis=0) / volume * 1e30 * 1e-8


def calc_energy_pressure(f_grun, m_grun, T, poscar_file):
    """
    Calculate the energy associated to the pressure estimated from the
    Gruneisen parameters.
    """
    poscar = generate_conf.read_POSCAR(poscar_file)
    nBE = np.array([fBE(f, T) + 0.5 for f in f_grun])
    nmodes = poscar["numbers"].sum() * 3.
    volume = abs(np.linalg.det(poscar["lattvec"] * 10.))
    return nmodes * np.mean(codata.h * m_grun * 1e12 *
                            nBE[:, np.newaxis]) / volume * 1e30 * 1e-8 / 3.


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
