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
import xml.etree.ElementTree as ElementTree
import sqlite3

import numpy as np
import scipy as sp
import scipy.constants as codata

import phonopy
import phonopy.interface
import phonopy.file_IO

import thermal_disp
import generate_conf
import symmetry
import time


def read_FORCE_CONSTANTS(sposcar_file, fcs_file):
    """
    Read and return the information contained in a FORCE_CONSTANTS
    file, which must correspond to the provided sposcar object.
    """

    #in phonopy comments:
    #force_constants[ i, j, a, b ]
    #  i: Atom index of finitely displaced atom.
    #  j: Atom index at which force on the atom is measured.
    #  a, b: Cartesian direction indices = (0, 1, 2) for i and j, respectively

    sposcar = generate_conf.read_POSCAR(sposcar_file)
    with open(fcs_file, "r") as f:
        n = int(next(f).strip())
        nsatoms = len(sposcar["types"])
        if n != nsatoms:
            raise ValueError(
                "file {} does not match the provided SPOSCAR object".format(
                    fcs_file))
        fullmatrix = np.empty((nsatoms, nsatoms, 3, 3))
        for iat1 in range(nsatoms):
            for iat2 in range(nsatoms):
                p1, p2 = [int(i) for i in next(f).split()]
                if p1 != iat1 + 1 or p2 != iat2 + 1:
                    raise ValueError(
                        "invalid index in file {}".format(fcs_file))
                for icoord in range(3):
                    fullmatrix[iat1, iat2, icoord, :] = [
                        float(j) for j in next(f).split()
                    ]
    return fullmatrix


def print_FORCE_CONSTANTS(fullmatrix, out_fcs_file):
    """
    Print FORCE_CONSTANTS file from a matrix object
    """
    with open(out_fcs_file, "w") as f:
        nsatoms = len(fullmatrix)
        f.write(" %3i\n" % nsatoms)
        for i in range(nsatoms):
            for j in range(nsatoms):
                f.write(" %3i %3i\n" % (i + 1, j + 1))
                for alpha in range(3):
                    f.write(" %21.15f %21.15f %21.15f\n" % (
                        fullmatrix[i, j, alpha, 0], fullmatrix[i, j, alpha, 1],
                        fullmatrix[i, j, alpha, 2]))
    return


def _check_file(filename):
    """
   Check if a calculation has finished successfully.
   """
    if not os.path.isfile(filename):
        return False
    if "forces" not in open(filename, "r").read():
        return False
    try:
        xml_tree = ElementTree.parse(filename)
    except ElementTree.ParseError as e:
        return False

    return True


def read_vasp_forces(filename):
    """
    Read a set of forces on atoms from filename, presumably in vasprun.xml
    format.
    """
    if not os.path.isfile(filename):
        sys.exit("The specified vasprun.xml file does not exist.")

    xml_tree = ElementTree.parse(filename)
    calculation = xml_tree.find("calculation")
    for a in calculation.findall("varray"):
        if a.attrib["name"] == "forces":
            break
    nruter = []
    for i in a.iter(tag="v"):
        nruter.append([float(j) for j in i.text.split()])
    nruter = np.array(nruter, dtype=np.double)

    return nruter


def read_vasp_stress(filename):
    """
    Read the stress tensor from filename, presumably in vasprun.xml format.
    """
    if not os.path.isfile(filename):
        sys.exit("The specified vasprun.xml file does not exist.")

    xml_tree = ElementTree.parse(filename)
    calculation = xml_tree.find("calculation")
    for a in calculation.findall("varray"):
        if a.attrib["name"] == "stress":
            break
    nruter = []
    for i in a.iter(tag="v"):
        nruter.append([float(j) for j in i.text.split()])
    nruter = np.array(nruter, dtype=np.double)
    #print("stress tensor: " + str(nruter.tolist()) + "\n")
    return nruter


def calc_mean_stress(iteration_min):
    """
    Calculate the mean stress tensor as an average over all configurations.
    """
    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()
    cur.execute("""SELECT id FROM configurations WHERE iteration >=?""",
                (iteration_min, ))
    config = cur.fetchall()
    conn.commit()
    stress = []
    for c in config:
        filename = os.path.join("config-" + str(c[0]), "vasprun.xml")
        if not _check_file(filename):
            print("problem with file " + filename + ", remove configuration")
            cur.execute("""DELETE FROM configurations WHERE id=?""", (c[0], ))
        print("read " + filename)
        stress.append(read_vasp_stress(filename))
    conn.close()
    return np.mean(np.array(stress), axis=0)


def calc_mean_stress_weights(iteration_min, weights):
    """
    Calculate the mean stress tensor as an average over all configurations.
    """
    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()
    cur.execute("""SELECT id FROM configurations WHERE iteration >=?""",
                (iteration_min, ))
    config = cur.fetchall()
    conn.commit()
    stress = []
    for c in config:
        filename = os.path.join("config-" + str(c[0]), "vasprun.xml")
        if not _check_file(filename):
            print("problem with file " + filename + ", remove configuration")
            cur.execute("""DELETE FROM configurations WHERE id=?""", (c[0], ))
        print("read " + filename)
        stress.append(read_vasp_stress(filename))
    conn.close()
    newweights = weights.reshape((len(config), -1))[:, 0]
    nruter = np.sum(
        np.array(stress) * newweights[:, np.newaxis, np.newaxis],
        axis=0) / np.sum(newweights)
    return nruter


def read_vasp_energy(filename):
    """
    Read energy from filename, presumably in vasprun.xml format.
    """
    if not os.path.isfile(filename):
        sys.exit("The specified vasprun.xml file does not exist.")

    xml_tree = ElementTree.parse(filename)
    calculation = xml_tree.find("calculation")
    for a in calculation.findall("energy"):
        nruter = 0.
        for i in a.iter(tag="i"):
            if i.attrib["name"] == "e_fr_energy":
                nruter = float(i.text)

    with open('out_energy', 'a') as file:
        file.write(filename + ' energy: ' + str(nruter) + '\n')
    return nruter


def read_vasp_fermi(filename):
    """
    Read fermi level from filename, presumably in vasprun.xml format.
    """
    if not os.path.isfile(filename):
        sys.exit("The specified vasprun.xml file does not exist.")

    xml_tree = ElementTree.parse(filename)
    calculation = xml_tree.find("calculation")
    for a in calculation.findall("dos"):
        nruter = 0.
        for i in a.iter(tag='i'):
            if i.attrib["name"] == "efermi":
                nruter = float(i.text)
                break
    with open('out_fermi', 'a') as file:
        file.write(filename + ' Fermi level: ' + str(nruter) + '\n')
    return nruter


def read_vasp_eigenvalues(filename):
    """
    Read eigenvalues from filename, presumably in vasprun.xml format.
    The number of eigenvalues to read is hardcoded.
    """
    if not os.path.isfile(filename):
        sys.exit("The specified vasprun.xml file does not exist.")

    nruter = []
    xml_tree = ElementTree.parse(filename)
    calculation = xml_tree.find("calculation")
    for a in calculation.findall("eigenvalues"):
        for i in a.getchildren()[0].getchildren()[-1].getchildren(
        )[0].getchildren()[0].getchildren():
            nruter.append([float(j) for j in i.text.split()])
    nruter = np.array(nruter, dtype=np.double)
    occupied = nruter[nruter[:, 1] >= 0.5, 0]
    empty = nruter[nruter[:, 1] < 0.5, 0]
    with open('out_eigenvalues', 'a') as file:
        file.write('3 highest occupied: ' + str(occupied[-3:].tolist()) + '\n')
        file.write('3 lowest empty: ' + str(empty[:3].tolist()) + '\n')
    return nruter, occupied[-3:], empty[:3]


def store_vasp_forces_energy(iteration):
    """
    Store forces and energy from VASP for all configurations
    """
    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()

    cur.execute("""SELECT id FROM configurations WHERE iteration=?""",
                (iteration, ))
    config = cur.fetchall()
    conn.commit()

    vasp_forces_energy = []
    for c in config:
        filename = os.path.join("config-" + str(c[0]), "vasprun.xml")
        if not _check_file(filename):
            print("problem with file " + filename + ", remove configuration")
            cur.execute("""DELETE FROM configurations WHERE id=?""", (c[0], ))
        else:
            print("read " + filename)
            forces = json.dumps(read_vasp_forces(filename).tolist())
            energy = read_vasp_energy(filename)
            vasp_forces_energy.append([forces, energy])
            cur.execute(
                """UPDATE configurations SET forces=?, energy=? WHERE id=?""",
                (forces, energy, c[0]))
    conn.commit()
    conn.close()
    return np.array(vasp_forces_energy)


def calc_3rd_forces(fcs_3rd_1cell, M, N, displacements):
    """
    Calculate the forces from the 3rd order force constants and the atomic
    displacements.
    """
    #BEWARE: NEED TO HAVE THE MATRIX IN (n,3,ntot,3,ntot,3).ravel() shape
    ntot3 = len(displacements)
    ncells = M.shape[0]
    fcs_3rd_coo = fcs_3rd_1cell.tocoo()
    rowi, coli = fcs_3rd_coo.nonzero()
    rowinew, colinew = np.unravel_index(coli, (ntot3 // ncells, ntot3 * ntot3))
    fcs_3rd_coo = sp.sparse.coo_matrix((fcs_3rd_coo.data, (rowinew, colinew)),
                                       shape=(ntot3 // ncells, ntot3 * ntot3))
    disp = np.array(displacements).reshape(-1, 3)

    forces = np.array([])
    for icell in range(M.shape[0]):
        disp_new = np.dot(M[icell], disp)
        disp2 = np.outer(disp_new.ravel(), disp_new.ravel()).ravel() * 100.
        forces = np.concatenate((forces, -fcs_3rd_coo.dot(disp2) / 2.))
    forces = np.dot(N, forces.reshape(-1, 3)).ravel()

    return forces


def calc_energy_3rd(fcs_2nd, fcs_3rd, M, N, displacements):
    """"
    Calculate the energy from the 2nd order and from the 3rd order part of the
    Hamiltonian for a given displaced configuration.
    """
    disp = np.array(displacements).reshape(-1, 3) * 10.
    forces_2nd = -np.sum(np.einsum('ij,ikjl->ikl', disp, fcs_2nd), axis=0)
    energy_2nd = -np.sum(forces_2nd * disp) / 2.

    forces_3rd = calc_3rd_forces(fcs_3rd, M, N, displacements)
    energy_3rd = -np.sum(forces_3rd * np.array(displacements) * 10.) / 3.

    return energy_2nd, energy_3rd


def calc_energy_deviation(fit_2nd, fit_3rd, mat_rec_ac, mat_rec_ac_3rd, M, N):
    """
    Calculate the difference between the energy from DFT and from the effective
    Hamiltonian.
    """
    fcs_2nd = np.einsum('i,ijklm->jklm', fit_2nd, mat_rec_ac)
    fcs_3rd = np.sum(
        [mat_rec_ac_3rd[k] * fit_3rd[k] for k in range(len(fit_3rd))], axis=0)

    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()

    cur.execute("""SELECT id, displacements, energy FROM configurations""")
    config = cur.fetchall()
    conn.commit()
    conn.close()

    energy_vasp = []
    energy_model = []
    deviation_2nd = []
    deviation = []
    for c in config:
        evasp = c[2]
        emodel2nd, emodel3rd = calc_energy_3rd(fcs_2nd, fcs_3rd, M, N,
                                               json.loads(c[1]))
        energy_vasp.append(evasp)
        energy_model.append(emodel2nd + emodel3rd)
        deviation_2nd.append(emodel2nd - evasp)
        deviation.append(emodel2nd + emodel3rd - evasp)

    return energy_vasp, energy_model, deviation_2nd, deviation


def calc_forces_energy(fcs, displacements):
    """
    Calculate the harmonic forces and energies for one configuration from the
    current force constants and lattice parameter.
    """
    disp = np.array(displacements).reshape(-1, 3)

    disp *= 10.  ##Put displacements in Angstroms
    natoms = len(disp)
    forces = -np.sum(np.einsum('ij,ikjl->ikl', disp, fcs), axis=0)
    energy = -np.sum(forces * disp)
    #print "harmonic forces",forces
    energy *= 0.5
    return [forces, energy]


def prepare_fit(mat_rec_ac, enforce_acoustic, iteration_min):
    """
    Prepare the input for the fit for a 2nd order effective Hamiltonian.
    """

    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()

    cur.execute(
        "SELECT id, displacements, forces "
        "FROM configurations WHERE iteration >=?", (iteration_min, ))
    config = cur.fetchall()
    conn.commit()
    conn.close()

    #ydata = []
    #xdata = []
    natoms = len(json.loads(config[0][1])) // 3
    xdata=np.empty(len((mat_rec_ac),len(config)*3*natoms))
    for k in range(len(mat_rec_ac)):
        xdata_int = np.empty((len(config),3*natoms))
        for c in range(len(config)):
            disp = 10. * np.array(json.loads(
                config[c][1]))  ##Put displacements in Angstroms
            xdata_int[c]=(-mat_rec_ac[k].dot(disp))
        if enforce_acoustic:
            for alpha in range(3):
                disp = np.zeros((natoms, 3))
                disp[:, alpha] = 0.01 * np.ones(natoms).T
                xdata_int.append(-mat_rec_ac[k].dot(np.ravel(disp)))
        xdata[k]=(np.ravel(xdata_int))
    xdata = np.transpose(np.array(xdata))
    ydata=np.empty(np.array(len(config),3*natoms))
    for c in range(len(config)):
        ydata[c]=(np.ravel(np.array(json.loads(c[2]))))
    if enforce_acoustic:
        for alpha in range(3):
            ydata.append(np.zeros(natoms * 3))
    ydata = np.ravel(ydata)

    return [xdata, ydata]


def prepare_fit_weights(mat_rec_ac, enforce_acoustic, iteration_min):
    """
    Prepare the input for the fit for a 2nd order effective Hamiltonian.
    """

    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT id, displacements, forces, probability, current_proba,"
        " iteration FROM configurations WHERE iteration >=?",
        (iteration_min, ))
    config = cur.fetchall()
    conn.commit()
    conn.close()

    ydata = []
    xdata = []
    weights = []
    natoms = int(len(json.loads(config[0][1])) / 3)

    sposcar = generate_conf.read_POSCAR("SPOSCAR_CURRENT")
    xdata=np.empty((len(mat_rec_ac),len(config)*3*natoms))
    for k in range(len(mat_rec_ac)):
        xdata_int = np.empty((len(config),3*natoms))
        for c in range(len(config)):
            sposcar_old = generate_conf.read_POSCAR("SPOSCAR_" + str(config[c][5]))
            newdisp = np.array(json.loads(config[c][1])) + np.ravel(
                np.dot(sposcar_old["lattvec"], sposcar_old["positions"]) -
                np.dot(sposcar["lattvec"], sposcar["positions"]))
            disp = 10. * newdisp  ##Put displacements in Angstroms
            xdata_int[c]=(-mat_rec_ac[k].dot(disp))
        if enforce_acoustic:
            for alpha in range(3):
                disp = np.zeros((natoms, 3))
                disp[:, alpha] = 0.01 * np.ones(natoms).T
                xdata_int.append(-mat_rec_ac[k].dot(np.ravel(disp)))
        xdata[k]=(np.ravel(xdata_int))
    xdata = np.transpose(np.array(xdata))
    ydata = np.empty((len(config),len(np.ravel(np.array(json.loads(config[0][2]))))))
    weights=np.empty((len(config),natoms*3))
    for c in range(len(config)):
        #print(len(json.loads(config[c][2])))
        ydata[c]=np.ravel(np.array(json.loads(config[c][2])))
        print("weight of config "+str(config[c][0])+": "+str(np.exp(config[c][4]-config[c][3])))
        weights[c]=(np.exp(config[c][4]-config[c][3])*np.ones(natoms*3))
    if enforce_acoustic:
        for alpha in range(3):
            ydata.append(np.zeros(natoms * 3))
            weights.append(10. * np.ones(natoms * 3))

    ydata = np.array(ydata)
    weights = np.array(weights)

#   remove mean force where it is not zero by symmetry
    if os.path.isfile("POSCAR_PARAM") and os.path.isfile("SPOSCAR_PARAM"):
        sposcar_param=generate_conf.read_POSCAR("SPOSCAR_PARAM")
        cartesian_positions=np.ravel(sp.dot(sposcar_param["lattvec"],sposcar_param["positions"]).T*10.)
        mean_forces = np.sum(ydata*weights,axis=0)/np.sum(weights,axis=0)
        delta_Ep = np.mean(mean_forces*cartesian_positions)
        with np.errstate(divide='ignore', invalid='ignore'):
            symm_mean_force = np.divide(delta_Ep, cartesian_positions)
            symm_mean_force[ ~ np.isfinite(symm_mean_force)] = 0

        with open("out_fit","a") as file:
            file.write("mean_forces: "+str(mean_forces.tolist())+"\n")
            file.write("symm_mean_force: "+str(symm_mean_force.tolist())+"\n")
        #ydata -= symm_mean_force[np.newaxis,:]

    ydata = np.ravel(ydata)
    weights = np.ravel(weights)

    return [xdata, ydata, weights]


def prepare_fit_3rd(mat_rec_ac, mat_rec_ac_3rd, M, N, enforce_acoustic,
                    iteration_min):
    """
    Prepare the input for the fit for a 2nd and 3rd order effective
    Hamiltonian.
    """
    xdata, ydata = prepare_fit(mat_rec_ac, enforce_acoustic, iteration_min)
    print("finished preparing 2nd order part of the fit")

    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()
    cur.execute(
        """SELECT id, displacements FROM configurations WHERE iteration >=?""",
        (iteration_min, ))
    config = cur.fetchall()
    conn.commit()
    conn.close()

    xdata_3rd = np.empty((range(len(mat_rec_ac_3rd.shape[0]))))
    natoms = len(json.loads(config[0][1])) // 3
    for k in range(mat_rec_ac_3rd.shape[0]):
        print("preparing data number " + str(k))
        fcs_3rd_full = mat_rec_ac_3rd[k]
        #print "full fcs matrix has been calculated"
        xdata_int = np.empty((range(len(config)),len(calc_3rd_forces(fcs_3rd_full,M,N,json.loads(config[0][1])))))
        for c in range(len(config)):
            xdata_int[c]=(
                calc_3rd_forces(fcs_3rd_full, M, N, json.loads(c[1])))

        if enforce_acoustic:
            for alpha in range(3):
                disp = np.zeros((natoms, 3))
                disp[:, alpha] = 0.01 * np.ones(natoms).T
                xdata_int.append(
                    calc_3rd_forces(fcs_3rd_full, M, N, np.ravel(disp)))

        xdata_3rd[k]=(np.ravel(xdata_int))
    xdata_3rd = np.concatenate((xdata, np.transpose(np.array(xdata_3rd))),
                               axis=1)

    return [xdata_3rd, ydata]

def prep_prepare_fit_3rd_weights(mat_rec_ac, mat_rec_ac_3rd, M, N, enforce_acoustic,
                            iteration_min):
    """
    Function to  prepare the parallel calculation of the weights
    """

    xdata, ydata, weights = prepare_fit_weights(mat_rec_ac, enforce_acoustic,
                                                iteration_min)
    print("finished preparing 2nd order part of the fit")

    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT id, displacements, iteration "
        "FROM configurations WHERE iteration >=?", (iteration_min, ))
    config = cur.fetchall()
    conn.commit()
    conn.close()

    sposcar = generate_conf.read_POSCAR("SPOSCAR_CURRENT")

    return xdata,ydata,weights,sposcar,config

  
    
  

def parallel_loop(k, mat_rec_ac_3rd, M, N, enforce_acoustic,
                            iteration_min,sposcar,config):

    #for k in range(mat_rec_ac_3rd.shape[0]):
    print("preparing data number " + str(k) + " of "+  str(len(range(mat_rec_ac_3rd.shape[0]))))
    fcs_3rd_full = mat_rec_ac_3rd[k]
    xdata_int = np.empty((len(config),len(np.array(calc_3rd_forces(fcs_3rd_full,M,N,json.loads(config[0][1]))))))
    natoms=int(len(json.loads(config[0][1]))/3)
    for c in range(len(config)):
        sposcar_old = generate_conf.read_POSCAR("SPOSCAR_" + str(config[c][2]))
        newdisp = np.array(json.loads(config[c][1])) + np.ravel(
                np.dot(sposcar_old["lattvec"], sposcar_old["positions"]) -
                np.dot(sposcar["lattvec"], sposcar["positions"]))
        xdata_int[c]=(calc_3rd_forces(fcs_3rd_full, M, N, newdisp))

    if enforce_acoustic:
        for alpha in range(3):
            disp = np.zeros((natoms, 3))
            disp[:, alpha] = 0.01 * np.ones(natoms).T
            xdata_int.append(
            calc_3rd_forces(fcs_3rd_full, M, N, np.ravel(disp)))
    
    return (np.ravel(xdata_int))



def calc_kinetic_term(iteration_min, weights):
    """
    Computes the kinetic contribution to pressure from a virial-like expression
    """
    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT id, displacements, forces, iteration FROM configurations"
        " WHERE iteration >=?", (iteration_min, ))
    config = cur.fetchall()
    conn.commit()
    conn.close()
    sposcar = generate_conf.read_POSCAR("SPOSCAR_CURRENT")
    newweights = weights.reshape((len(config), -1))[:, 0]
    newweights = newweights / np.sum(newweights)
    nruter = -np.sum(
        np.array([
            np.sum(
                10. * (np.array(json.loads(c[1])) + np.ravel(
                    np.dot(
                        generate_conf.read_POSCAR("SPOSCAR_" +
                                                  str(c[3]))["lattvec"],
                        generate_conf.read_POSCAR("SPOSCAR_" +
                                                  str(c[3]))["positions"]) -
                    np.dot(sposcar["lattvec"], sposcar["positions"]))).reshape(
                        -1, 3) * (np.array(json.loads(c[2])).reshape(-1, 3)),
                axis=0) for c in config
        ]) * newweights[:, np.newaxis],
        axis=0)
    return nruter / abs(np.linalg.det(
        sposcar["lattvec"] * 10.)) * codata.e * 1e30 * 1e-8


def store_current_forces_energy(fcs):
    """
    Store the harmonic forces and energies for all configurations from the
    current force constants
    """
    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()

    cur.execute("""SELECT id, displacements FROM configurations""")
    config = cur.fetchall()
    conn.commit()
    conn.close()

    forces = []
    energy = []
    for c in config:
        forcesc, energyc = calc_forces_energy(fcs, json.loads(c[1]))
        forces.append(forcesc)
        energy.append(energyc)
    return [np.array(forces), np.array(energy)]


def calc_mean_vasp_energy():
    """
    Calculate the averaged DFT energy on all configurations.
    """
    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()

    cur.execute("""SELECT id, energy FROM configurations""")
    config = cur.fetchall()

    conn.commit()
    conn.close()
    energy = np.array([c[1] for c in config])

    return np.mean(energy)


def calc_param_grad(fcs, param_sposcar, iteration_min):
    """
    Calculate and return the gradient of energy with respect to a given
    structural parameter. The harmonic part from the effective Hamiltonian is
    removed to reduce statistical noise.
    """
    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()

    cur.execute(
        "SELECT id, displacements, forces "
        "FROM configurations WHERE iteration >=?", (iteration_min, ))
    config = cur.fetchall()

    conn.commit()
    conn.close()

    forces = np.array([json.loads(c[2]) for c in config])
    print("forces: " + str(forces.tolist()))

    grad = np.mean(-np.mean(forces, axis=0) * param_sposcar["positions"].T)
    print("parameter gradient: " + str(grad))

    # gradient is in eV/A
    return grad


def calc_delta_Ep(fcs, sposcar_param, iteration_min):
    """
    Calculate and return the difference of potential energy with respect to a
    given structural parameter.
    """
    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()

    cur.execute(
        "SELECT id, displacements, forces "
        "FROM configurations WHERE iteration >=?", (iteration_min, ))
    config = cur.fetchall()

    conn.commit()
    conn.close()

    forces = np.array([json.loads(c[2]) for c in config])
    print("forces: " + str(forces.tolist()))

    cartesian_positions = sp.dot(sposcar_param["lattvec"],
                                 sposcar_param["positions"]).T * 10.
    delta_Ep = np.mean(-np.mean(forces, axis=0) * cartesian_positions)
    print("delta Ep: " + str(delta_Ep))

    #delta_Ep is in eV
    return delta_Ep


def calc_delta_Ep_weights(fcs, sposcar_param, iteration_min, weights):
    """
    Calculate and return the difference of potential energy with respect to a
    given structural parameter.
    """
    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()

    cur.execute(
        "SELECT id, displacements, forces "
        "FROM configurations WHERE iteration >=?", (iteration_min, ))
    config = cur.fetchall()

    conn.commit()
    conn.close()

    forces = np.array([json.loads(c[2]) for c in config])
    print("forces: " + str(forces.tolist()))

    newweights = weights.reshape((len(config), -1))[:, 0]
    mean_forces = np.sum(
        forces * newweights[:, np.newaxis, np.newaxis],
        axis=0) / np.sum(newweights)

    cartesian_positions = sp.dot(sposcar_param["lattvec"],
                                 sposcar_param["positions"]).T * 10.
    delta_Ep = np.mean(-mean_forces * cartesian_positions)
    print("delta Ep: " + str(delta_Ep))

    #delta_Ep is in eV
    return delta_Ep

def disp_optimize_positions_weights(fcs, sposcar_file, iteration_min, weights):
    """
    Calculates and returns the displacement that optimizes atomic positions
    """
    conn = sqlite3.connect("QSCAILD.db")
    cur = conn.cursor()

    cur.execute(
        "SELECT id, displacements, forces "
        "FROM configurations WHERE iteration >=?", (iteration_min, ))
    config = cur.fetchall()

    conn.commit()
    conn.close()
    
    sym_force_prec = 1e-2

    forces = np.array([json.loads(c[2]) for c in config])

    newweights = weights.reshape((len(config), -1))[:, 0]
    mean_forces = np.sum(
        forces * newweights[:, np.newaxis, np.newaxis],
        axis=0) / np.sum(newweights)

    symm_forces = symmetry.symmetrize_forces(sposcar_file, mean_forces)

    #with open("out_atomic_positions","a") as file:
    #    file.write("mean_forces: "+str(mean_forces.tolist())+"\n")
    #    file.write("symm_forces: "+str(symm_forces.tolist())+"\n")

    flat_forces = np.ravel(symm_forces)
    flat_fcs = np.swapaxes(fcs,1,2).reshape((flat_forces.shape[0],flat_forces.shape[0]))
    disp = sp.linalg.solve(-flat_fcs.T,flat_forces).reshape(-1,3)*0.1
    np.savetxt('current_forces.txt', flat_forces)
    np.savetxt('current_fcs.txt', flat_fcs)
    #Resymmetrize displacement
    symm_disp = symmetry.symmetrize_forces(sposcar_file, disp)
    #Removing drift:
    symm_disp=np.ravel(symm_disp.reshape(-1,3)-np.multiply(np.ones(symm_disp.reshape(-1,3).shape),np.mean(symm_disp.reshape(-1,3),axis=0)))
    #with open("out_atomic_positions","a") as file:
    #    file.write("non_symmetrized displacements:"+str(disp.tolist())+"\n")
    #    file.write("symmetrized displacements:"+str(symm_disp.tolist())+"\n")
    
    #Add threshold to avoid huge displacements
    max_disp = np.amax(np.abs(symm_disp))
    scale_disp=0.2

    with open("out_atomic_positions","a") as file:
        file.write("check correspondence:\n")
        file.write("max harmonic forces:"+str(np.max(np.abs(symm_forces - calc_forces_energy(fcs, disp)[0])))+"\n")
        file.write("max delta sym displacement:"+str(np.max(np.abs(symm_forces - calc_forces_energy(fcs, symm_disp)[0])))+"\n")
        file.write("max force:"+str(np.max(np.abs(symm_forces)))+"\n")
        file.write("max disp:"+str(max_disp)+"\n")
        file.write("Removed drift:"+str(np.mean(symm_disp.reshape(-1,3),axis=0)))

    if (max_disp > 0.01):
        symm_disp*=0.01/max_disp
    if np.max(np.abs(symm_forces)) < sym_force_prec:
        print("Max force lower than threshold, skipping ionic optimization")
        symm_disp*=0
    if max_disp > 5.:
        print("Max disp unrealistically large -> FCs not converged enough, skipping ionic optimization")
        symm_disp*=0

    return (-1)*scale_disp*np.ravel(symm_disp)
