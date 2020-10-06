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
import gc
import sqlite3
import logging

import numpy as np
import scipy as sp
import scipy.constants as codata

import phonopy
import phonopy.interface
import phonopy.file_IO
import phonopy.structure
import phonopy.structure.symmetry

import thermal_disp
import generate_conf
import gradient
import thirdorder_core
import thirdorder_save
import thirdorder_common


def get_symmetry_information(sposcar_file):
    """
    Obtain the symmetry information useful to compute the 2nd order force
    constants irreducible elements.
    """

    logging.basicConfig(level=0)
    SYMPREC = 1e-5

    if not os.path.isfile(sposcar_file):
        sys.exit("The specified SPOSCAR file does not exist.")

    supercell_matrix = np.diag([1, 1, 1])

    structure = phonopy.interface.calculator.read_crystal_structure(sposcar_file,
                                                         "vasp")[0]
    natoms = structure.get_number_of_atoms()

    symmetry = phonopy.structure.symmetry.Symmetry(structure, symprec=SYMPREC)

    dataset = symmetry.get_dataset()
    print("Space group {0} ({1}) detected".format(dataset["international"],
                                                  dataset["number"]))
    crotations = get_crotations(structure, symmetry)

    operations = symmetry.get_symmetry_operations()
    translations = operations["translations"]
    rotations = operations["rotations"]

    logging.debug("About to classify atom pairs")
    permutations = find_correspondences(structure, symmetry, SYMPREC)
    print("permutations: " + str(permutations))
    equivalences = classify_pairs(structure, permutations)
    logging.debug("{0} equivalence classes found".format(len(equivalences)))

    for i, eq in enumerate(equivalences):
        logging.debug("Equivalence class #{0}".format(i))
        for il, l in enumerate(eq):
            logging.debug("\t{0}. {1}".format(il + 1, l))

    kernels = []
    irreducible = []
    for i, eq in enumerate(equivalences):
        logging.debug("Representative of equivalence class #{0}".format(i))
        invariances = find_invariances(eq[0][0], permutations)
        logging.debug("is left invariant by {0} operations".format(
            len(invariances)))
        for iinv, inv in enumerate(invariances):
            logging.debug("\t{0}. {1}".format(iinv, inv))
        logging.debug("About to build the constraint matrix")
        coefficients = get_constraints(
            [crotations[inv] for inv in invariances])
        rank = np.linalg.matrix_rank(coefficients)
        logging.debug("{0} constraints".format(coefficients.shape[0]))
        logging.debug(
            "{0} independent components in this equivalence class:".format(
                9 - rank))
        if (rank > 0):
            v = sp.linalg.svd(coefficients, full_matrices=False)[2].T
            kernels.append(v[:, rank - 9:])
            irreducible.append(complete_constraints(coefficients))
        else:
            kernels.append(np.identity(9))
            irreducible.append([0, 1, 2, 3, 4, 5, 6, 7, 8])
        for iirr, irr in enumerate(irreducible[-1]):
            c1, c2 = np.unravel_index(irr, (3, 3))
            logging.debug("\t{0}. {1}{2},{3}{4}".format(
                iirr, eq[0][0][0], "xyz" [c1], eq[0][0][1], "xyz" [c2]))
    irreducible = tuplify(irreducible)

    todo = []
    for eq, irr in zip(equivalences, irreducible):
        a1, a2 = eq[0][0]
        for i in irr:
            print("a1", a1, "a2", a2, "i", np.unravel_index(i, (3, 3)))
        todo += [dof2dof(a1, a2, i, natoms) for i in irr]
    todo = tuplify(todo)

    with open('out_sym', 'a') as file:
        file.write("2nd order todo list before acoustic sum rule: " +
                   str(todo) + "\n")
        file.write("number of irreducible elements before acoustic sum rule: "
                   + str(len(todo)) + "\n")
    return [natoms, crotations, equivalences, kernels, irreducible, todo]


def reconstruct_fc(natoms, crotations, equivalences, kernels, irreducible,
                   irr_fc):
    """
    Reconstruct a full 2nd order force constants matrix from the irreducible
    elements.
    """
    dataset = np.empty((natoms * 3, natoms * 3), dtype=np.float64)

    first = 0
    last = 0
    pairfc = np.empty(9, dtype=np.float64)
    for iclass, (eq, ker, irr) in enumerate(
            zip(equivalences, kernels, irreducible)):
        last += len(irr)
        logging.debug(
            "About to reconstruct the whole FCS for atom pair {0}".format(
                eq[0][0]))
        indep = irr_fc[first:last]
        if len(irr) == 0:
            pairfc = np.zeros(9)
        else:
            pairfc = np.dot(ker, sp.linalg.solve(ker[irr, :], indep))
            pairfc = pairfc * (np.fabs(pairfc) >= 1e-12)
        a1, a2 = eq[0][0]
        for iel in range(9):
            dof1, dof2 = dof2dof(a1, a2, iel, natoms)
            dataset[dof1, dof2] = pairfc[iel]
        logging.debug(
            "About to reconstruct all FCS for equivalence class #{0}".format(
                iclass))
        for triplet in eq[1:]:
            a1, a2 = triplet[0]
            r = crotations[triplet[1]]
            tmpfc = pairfc.reshape((3, 3))
            tmpfc = np.dot(r.T, np.dot(tmpfc, r))
            tmpfc = tmpfc * (np.fabs(tmpfc) >= 1e-12)
            if triplet[2]:
                tmpfc = tmpfc.T
            tmpfc = tmpfc.ravel()
            for iel in range(9):
                dof1, dof2 = dof2dof(a1, a2, iel, natoms)
                dataset[dof1, dof2] = tmpfc[iel]
        first = last

    mat = dataset.reshape(natoms, 3, natoms, 3).swapaxes(1, 2)

    return mat


def reconstruct_fc_sparse(natoms, crotations, equivalences, kernels,
                          irreducible, irr_fc):
    """
    Reconstruct a sparse 2nd order force constants matrix from the irreducible
    elements.
    """
    dataset = np.empty((natoms * 3, natoms * 3), dtype=np.float64)

    first = 0
    last = 0
    pairfc = np.empty(9, dtype=np.float64)
    for iclass, (eq, ker, irr) in enumerate(
            zip(equivalences, kernels, irreducible)):
        last += len(irr)
        logging.debug(
            "About to reconstruct the whole FCS for atom pair {0}".format(
                eq[0][0]))
        indep = irr_fc[first:last]
        if len(irr) == 0:
            pairfc = np.zeros(9)
        else:
            pairfc = np.dot(ker, sp.linalg.solve(ker[irr, :], indep))
            pairfc = pairfc * (np.fabs(pairfc) >= 1e-12)
        a1, a2 = eq[0][0]
        for iel in range(9):
            dof1, dof2 = dof2dof(a1, a2, iel, natoms)
            dataset[dof1, dof2] = pairfc[iel]
        logging.debug(
            "About to reconstruct all FCS for equivalence class #{0}".format(
                iclass))
        for triplet in eq[1:]:
            a1, a2 = triplet[0]
            r = crotations[triplet[1]]
            tmpfc = pairfc.reshape((3, 3))
            tmpfc = np.dot(r.T, np.dot(tmpfc, r))
            tmpfc = tmpfc * (np.fabs(tmpfc) >= 1e-12)
            if triplet[2]:
                tmpfc = tmpfc.T
            tmpfc = tmpfc.ravel()
            for iel in range(9):
                dof1, dof2 = dof2dof(a1, a2, iel, natoms)
                dataset[dof1, dof2] = tmpfc[iel]
        first = last

    return sp.sparse.csr_matrix(dataset)


def acoustic_sum_rule(natoms, crotations, equivalences, kernels, irreducible,
                      todo):
    """
    Obtains the constraints imposed by the 2nd order acoustic sum rule.
    """
    print("calculate 2nd order acoustic sum")
    nirr = len(todo)
    sum_rule = np.empty((natoms * 9, nirr))
    for i in range(nirr):
        print("element " + str(i))
        phi = np.zeros(nirr)
        phi[i] = 1.
        sum_rule[:, i] = np.ravel(
            np.sum(
                reconstruct_fc(natoms, crotations, equivalences, kernels,
                               irreducible, phi),
                axis=1))
    print("acoustic sum rule constraint begins")
    print("calculate rank")
    rank = np.linalg.matrix_rank(sum_rule)
    print("rank: " + str(rank))
    print("calculate kernel")
    v = sp.linalg.svd(sum_rule)[2].T
    kernel = v[:, rank - nirr:]
    print("acoustic sum rule constraint finished")
    with open('out_sym', 'a') as file:
        file.write(
            "2nd order number of irreducible elements after acoustic sum rule: "
            + str(nirr - rank) + "\n")
    return kernel, nirr - rank


def reconstruct_fc_acoustic(natoms,
                            ker_ac,
                            irr_fc_ac,
                            crotations,
                            equivalences,
                            kernels,
                            irreducible,
                            enforce_acoustic=True):
    """
    Reconstructs a sparse 2nd order force constants matrix from the irreducible
    elements including the acoustic sum rule constraints.
    """
    if enforce_acoustic:
        fcg_rec = np.dot(ker_ac, irr_fc_ac)
    else:
        fcg_rec = irr_fc_ac
    fc = reconstruct_fc_sparse(natoms, crotations, equivalences, kernels,
                               irreducible, fcg_rec)
    return fc


def reconstruct_fc_acoustic_frommat(mat_rec_ac, irr_fc_ac):
    """
    Reconstructs a full 2nd order force constants matrix from the symmetry
    matrix and the irreducible elements including the acoustic sum rule
    constraints.
    """
    fc = sum(mat_rec_ac * irr_fc_ac).todense()
    natoms = len(fc) // 3
    mat = np.empty((natoms, natoms, 3, 3), dtype=np.float64)
    for d1 in range(len(fc)):
        for d2 in range(len(fc)):
            mat[np.unravel_index(d1, (natoms, 3))[0],
                np.unravel_index(d2, (natoms, 3))[0],
                np.unravel_index(d1, (natoms, 3))[1],
                np.unravel_index(d2, (natoms, 3))[1]] = fc[d1, d2]
    return mat


def calc_corresp(poscar, sposcar, n):
    """
    Calculates the correspondence between the atoms in a POSCAR and in an
    SPOSCAR file.
    """
    poscar_positions = sp.dot(poscar["lattvec"], poscar["positions"])
    sposcar_positions = sp.dot(sposcar["lattvec"], sposcar["positions"])
    corresp = []
    for iatom in range(sposcar["numbers"].sum()):
        for na in range(n[0]):
            for nb in range(n[1]):
                for nc in range(n[2]):
                    for jatom in range(poscar["numbers"].sum()):
                        if np.allclose(
                                sposcar_positions[:, iatom] - sp.dot(
                                    poscar["lattvec"], [na, nb, nc]),
                                poscar_positions[:, jatom]):
                            corresp.append([jatom, na, nb, nc])
    return corresp


def calc_mat_rec_ac_3rd(poscar,
                        sposcar,
                        ker_ac_3rd,
                        nirr_ac_3rd,
                        wedge,
                        list4,
                        n3rdorder,
                        enforce_acoustic=True):
    """
    Calculates the 3rd order symmetry matrix including the acoustic sum rule.
    """
    datanew = np.array([])
    colinew = np.array([])
    rowinew = np.array([])
    natoms = poscar["numbers"].sum()
    ncells = n3rdorder[0] * n3rdorder[1] * n3rdorder[2]

    for k in range(nirr_ac_3rd):
        print("preparing data number " + str(k))
        phi = np.zeros(nirr_ac_3rd)
        phi[k] = 1.
        fcs_3rd_1cell = sp.sparse.coo_matrix(
            np.ravel(
                reconstruct_3rd_fcs(poscar, sposcar, ker_ac_3rd, phi, wedge,
                                    list4, enforce_acoustic)))
        data = fcs_3rd_1cell.data
        rowi, coli = fcs_3rd_1cell.nonzero()
        fullindex = np.unravel_index(
            coli, (3, 3, 3, natoms, natoms * ncells, natoms * ncells))
        coli = np.ravel_multi_index(
            (fullindex[3], fullindex[0], fullindex[4], fullindex[1],
             fullindex[5], fullindex[2]),
            (natoms, 3, natoms * ncells, 3, natoms * ncells, 3))
        print("fcs matrix has been calculated")
        datanew = np.concatenate((datanew, data))
        colinew = np.concatenate((colinew, coli))
        rowinew = np.concatenate((rowinew,
                                  np.array([k for nnz in range(len(coli))])))

    mat_rec_ac_3rd = sp.sparse.coo_matrix(
        (datanew, (rowinew, colinew)),
        shape=(nirr_ac_3rd,
               natoms * natoms * natoms * ncells * ncells * 27)).tocsr()
    return mat_rec_ac_3rd


def reconstruct_3rd_fcs(poscar,
                        sposcar,
                        ker_ac_3rd,
                        irr_fc_ac_3rd,
                        wedge,
                        list4,
                        enforce_acoustic=True):
    """
    Reconstructs the 3rd order force constants from the irreducible elements.
    """
    if enforce_acoustic:
        phi = np.dot(ker_ac_3rd, irr_fc_ac_3rd)
    else:
        phi = irr_fc_ac_3rd
    fcs_3rd_1cell = thirdorder_core.reconstruct_ifcs_philist(
        phi, wedge, list4, poscar, sposcar)
    return fcs_3rd_1cell


def save_symmetry_information_3rd(n3rdorder, third, symm_acoustic=True):
    """
    Computes and saves the 2nd and 3rd order symmetry matrices.
    """
    (natoms, crotations, equivalences, kernels, irreducible,
     todo) = get_symmetry_information("SPOSCAR")
    if (symm_acoustic):
        ker_ac, nirr_ac = acoustic_sum_rule(natoms, crotations, equivalences,
                                            kernels, irreducible, todo)
    else:
        nirr_ac = len(todo)
        ker_ac = np.identity(nirr_ac)
    mat_rec_ac = [
        reconstruct_fc_acoustic(
            natoms, ker_ac, np.array([int(j == k) for j in range(nirr_ac)]),
            crotations, equivalences, kernels, irreducible, symm_acoustic)
        for k in range(nirr_ac)
    ]
    np.save(
        "../mat_rec_ac_2nd_" + str(n3rdorder[0]) + "x" + str(n3rdorder[1]) +
        "x" + str(n3rdorder[2]) + ".npy", mat_rec_ac)
    if not third:
        return
    poscar = generate_conf.read_POSCAR("POSCAR")
    sposcar = thirdorder_common.gen_SPOSCAR(poscar, n3rdorder[0], n3rdorder[1],
                                            n3rdorder[2])
    if (symm_acoustic):
        (ker_ac_3rd, nirr_ac_3rd, wedge, list4,
         dmin, nequi, shifts, frange) = thirdorder_save.save(
             "save_sparse", n3rdorder[0], n3rdorder[1], n3rdorder[2],
             n3rdorder[3])
        with open('out_sym', 'a') as file:
            file.write("3rd order number of irreducible elements after " +
                       "acoustic sum rule: " + str(nirr_ac_3rd) + "\n")
        np.save(
            "../ker_ac_3rd_" + str(n3rdorder[0]) + "x" + str(n3rdorder[1]) +
            "x" + str(n3rdorder[2]) + "_" + str(n3rdorder[3]) + ".npy",
            ker_ac_3rd)
    else:
        wedge, list4, dmin, nequi, shifts, frange = thirdorder_save.save(
            "return", n3rdorder[0], n3rdorder[1], n3rdorder[2], n3rdorder[3])
        nirr_ac_3rd = 0
        for ii in range(wedge.nlist):
            print("nindependentbasis: " + str(wedge.nindependentbasis[ii]))
            nirr_ac_3rd += wedge.nindependentbasis[ii]
        with open('out_sym', 'a') as file:
            file.write("3rd order number of irreducible elements without" +
                       " acoustic sum rule: " + str(nirr_ac_3rd) + "\n")
        ker_ac_3rd = np.identity(nirr_ac_3rd)
    mat_rec_ac_3rd = calc_mat_rec_ac_3rd(poscar, sposcar, ker_ac_3rd,
                                         nirr_ac_3rd, wedge, list4, n3rdorder,
                                         symm_acoustic)
    np.save(
        "../mat_rec_ac_3rd_" + str(n3rdorder[0]) + "x" + str(n3rdorder[1]) +
        "x" + str(n3rdorder[2]) + "_" + str(n3rdorder[3]) + ".npy",
        mat_rec_ac_3rd)
    return


# M is the matrix making the correspondence between one atom in sposcar2
# (with respect to a given unit cell) and its counterpart in sposcar
# (with respect to the first unit cell)
# N is the matrix making the correspondence between one atom when the unit
# cells are considered successively and its counterpart in sposcar2
def calc_cells_dispmats(n):
    """
    Computes the correspondence between the SPOSCAR in the thirdorder
    convention and the one in the generate_conf convention.
    """
    poscar = generate_conf.read_POSCAR("POSCAR")
    sposcar = thirdorder_common.gen_SPOSCAR(poscar, n[0], n[1], n[2])
    sposcar2 = generate_conf.read_POSCAR("SPOSCAR")
    natoms = poscar["numbers"].sum()
    ncells = n[0] * n[1] * n[2]
    corresp = calc_corresp(poscar, sposcar, [n[0], n[1], n[2]])
    corresp2 = calc_corresp(poscar, sposcar2, [n[0], n[1], n[2]])
    print("corresp: " + str(corresp))
    print("corresp2: " + str(corresp2))
    M = np.zeros((ncells, natoms * ncells, natoms * ncells))
    for icell in range(ncells):
        for iatom in range(natoms * ncells):
            nacell, nbcell, nccell = np.unravel_index(icell,
                                                      (n[0], n[1], n[2]))
            i1cell, na1cell, nb1cell, nc1cell = corresp[iatom]
            jatom = corresp2.index([
                i1cell, (na1cell + nacell) % n[0], (nb1cell + nbcell) % n[1],
                (nc1cell + nccell) % n[2]
            ])
            M[icell, iatom, jatom] = 1.
    np.set_printoptions(threshold=sys.maxsize)
    N = np.zeros((ncells * natoms, ncells * natoms))
    for iatom in range(natoms * ncells):
        i1cell, na1cell, nb1cell, nc1cell = corresp2[iatom]
        icell = np.ravel_multi_index((na1cell, nb1cell, nc1cell),
                                     (n[0], n[1], n[2]))
        N[iatom, icell * natoms + i1cell] = 1.
    return M, N


def compute_irr_fc(sposcar_file,
                   fcs_file,
                   mat_rec_ac_file,
                   mat_rec_ac_reshaped_file="mat_rec_ac_2nd_reshaped.npy",
                   calc_reshape=True):
    """
    Computes the 2nd order irreducible elements from a force constants file
    (correspondence not tested in this latest version).
    """
    fcs_to_fit = gradient.read_FORCE_CONSTANTS(sposcar_file, fcs_file)
    ntot = len(fcs_to_fit)
    mat_rec_ac = np.load(mat_rec_ac_file, allow_pickle = True)
    if calc_reshape:
        reshape_mat_rec_ac(mat_rec_ac, mat_rec_ac_reshaped_file, ntot)
    mat_rec_ac_new = np.load(mat_rec_ac_reshaped_file, allow_pickle = True)[()]
    print("computing irreducible elements")
    fit = sp.sparse.linalg.lsqr(mat_rec_ac_new,
                                np.rollaxis(fcs_to_fit, 2, 1).ravel())
    irr_fc_ac = fit[0]
    print("printing force constants to file")
    force_constants = reconstruct_fc_acoustic_frommat(mat_rec_ac, irr_fc_ac)
    gradient.print_FORCE_CONSTANTS(force_constants, "FORCE_CONSTANTS_REC")
    return irr_fc_ac


def reshape_mat_rec_ac(mat_rec_ac, mat_rec_ac_reshaped_file, ntot):
    """
    Reshapes the 2nd order symmetry matrix for use in the compute_irr_fc
    function.
    """
    print("reshaping matrix")
    nirr_ac = len(mat_rec_ac)
    print(str(nirr_ac) + " irreducible elements")
    data_new = []
    colinew = []
    rowinew = []
    for k in range(nirr_ac):
        print("element " + str(k))
        fcs_matrix = mat_rec_ac[k].tocoo()
        data = fcs_matrix.data
        rowi, coli = fcs_matrix.nonzero()
        data_new = np.concatenate((data_new, data))
        rowinew = np.concatenate((rowinew,
                                  np.ravel_multi_index((rowi, coli),
                                                       (ntot * 3, ntot * 3))))
        colinew = np.concatenate((colinew, [k for i in range(len(data))]))
    mat_rec_ac_new = sp.sparse.coo_matrix((data_new, (rowinew, colinew)),
                                          shape=(ntot * ntot * 9,
                                                 nirr_ac)).tocsr()
    print("saving matrix")
    np.save(mat_rec_ac_reshaped_file, mat_rec_ac_new)
    return


def dof2str(dof):
    """
    Return a short string describing what a given degree of freedom
    means.
    """
    a, c = divmod(dof, 3)
    a += 1
    c = "xyz" [c]
    return "{0}{1}".format(a, c)


def dof2dof(a1, a2, i, natoms):
    """
    Translate indices for 9-vectors into indices for ndof x ndof
    matrices.
    """
    c1, c2 = np.unravel_index(i, (3, 3))
    dof1 = np.ravel_multi_index((a1, c1), (natoms, 3))
    dof2 = np.ravel_multi_index((a2, c2), (natoms, 3))
    return (dof1, dof2)


def get_crotations(structure, symmetry):
    """
    Return the matrices representing the rotations in Cartesian
    coordinates.
    """
    lattvec = structure.cell.T
    operations = symmetry.get_symmetry_operations()
    rotations = operations["rotations"]
    nruter = []
    for r in rotations:
        crotation = (np.dot(np.linalg.solve(lattvec.T, r.T), lattvec.T)).T
        nruter.append(crotation * (np.fabs(crotation) >= 1e-12))
    return nruter


def complete_constraints(coefficients):
    """
    Find an irreducible set of coefficients of a block of the Green's
    function given a set of constraints.
    """
    nruter = []
    coeff = np.copy(coefficients)
    maxrank = len(coeff[0])
    todo = maxrank - np.linalg.matrix_rank(coeff)
    for i in range(maxrank):
        row = np.zeros(maxrank)
        row[i] = 1.
        newcoeff = np.vstack((coeff, row))
        newtodo = maxrank - np.linalg.matrix_rank(newcoeff)
        if newtodo < todo:
            todo = newtodo
            coeff = newcoeff
            nruter.append(i)
        if todo == 0:
            break
    return nruter


def get_constraints(crotations):
    """
    Build the coefficient matrix describing the constraints imposed by
    rotations on a block of the Green's function.
    """
    nruter = []
    for r in crotations:
        for (i, l) in itertools.product(range(3), repeat=2):
            row = []
            for (j, k) in itertools.product(range(3), repeat=2):
                tmp = r[i, j] * r[l, k]
                if i == j and l == k:
                    tmp -= 1.
                row.append(tmp)
            nruter.append(row)
    return np.array(nruter)


def find_invariances(pair, permutations):
    """
    Return the indices of all permutations that map a pair onto
    itself. The identity is not included.
    """
    nruter = []
    for iperm, perm in enumerate(permutations):
        newpair = tuple(perm[i] for i in pair)
        if newpair == pair:
            nruter.append(iperm)
    return nruter[1:]


def tuplify(arg):
    """
    Take an arbitrarily nested list structure and convert all lists to
    tuples.
    """
    if isinstance(arg, (list, tuple)):
        return tuple(tuplify(i) for i in arg)
    else:
        return arg


def find_correspondences(structure, symmetry, tolerance):
    """
    Map each symmetry operation to a permutation over the atoms in
    structure.
    """
    natoms = structure.get_number_of_atoms()
    operations = symmetry.get_symmetry_operations()
    translations = operations["translations"]
    rotations = operations["rotations"]

    correspondences = []
    positions = structure.get_scaled_positions()
    nsym = len(translations)
    for isym in range(nsym):
        remaining = list(range(natoms))
        perm = []
        for iatom in range(natoms):
            dest = np.dot(rotations[isym, :, :],
                          positions[iatom, :]) + translations[isym, :]
            dest %= 1.
            for jatom in remaining:
                delta = dest - positions[jatom, :]
                delta -= np.round(delta)
                if np.allclose(delta, 0., atol=tolerance):
                    remaining.remove(jatom)
                    perm.append(jatom)
                    break
            else:
                raise ValueError(
                    "the structure must have the specified symmetry")
        correspondences.append(perm)
    return tuplify(correspondences)


def classify_pairs(structure, permutations):
    """
    Split the pairs of atoms in structure among equivalence classes
    defined by the permutation operations. Each equivalence class is
    represented by a tuple of tuples:

    (pair, operation, exchange)

    where operation is the index of the permutation that maps pair
    onto the first pair in the class, and exchange is a boolean
    variable taking a value of True when an additional exchange of
    indices is necessary.
    """
    natoms = structure.get_number_of_atoms()
    nruter = []

    for pair in itertools.product(range(natoms), repeat=2):
        for iperm, perm in enumerate(permutations):
            newpair = tuple(perm[i] for i in pair)
            for eq in nruter:
                if newpair == eq[0][0]:
                    eq.append((pair, iperm, False))
                    break
                elif newpair[::-1] == eq[0][0]:
                    eq.append((pair, iperm, True))
                    break
            else:
                continue
            break
        else:
            # We rely on operation 0 being the identity.
            nruter.append([(pair, 0, False)])
    return tuplify(nruter)
