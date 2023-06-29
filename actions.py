#! /usr/bin/env python
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

import sys
import os
import os.path
import shutil
import sqlite3
import subprocess
import glob
import pprint
import datetime
import time
import logging
try:
    import pickle as pickle
except ImportError:
    import pickle
import numpy as np

import generate_conf
import thermal_disp
import gradient
import symmetry
import gruneisen
import json

import math
import scipy as sp
import phonopy
import phonopy.interface
import phonopy.file_IO
import phonopy.structure
import phonopy.structure.symmetry
import io
import scipy.constants as codata
from sklearn import linear_model
import thirdorder_common
import thirdorder_save
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def fit_force_constants(nconf, nfits, T, n, cutoff, third, use_pressure,
                        pressure, optimize_positions, use_smalldisp, calc_symm, symm_acoustic,
                        imaginary_freq, enforce_acoustic, grid, tolerance,
                        pdiff, memory, mixing):
    """
    Main function that monitors the self-consistency loop.
    """
    if rank==0:
        tic=time.perf_counter()
    iteration = 0
    write_gruneisen = False

    if not os.path.isfile("iteration"):
        iteration = 1
        if os.path.isfile("QSCAILD.db"):
            with open("finished", "w") as file:
                file.write("finished: error\n")
            comm.Abort()
            sys.exit(
                "Problem: no previous iteration but database already present,"
                " remove file QSCAILD.db")
        else:
            if rank == 0:
                shutil.copy("POSCAR", "POSCAR_CURRENT")
                shutil.copy("SPOSCAR", "SPOSCAR_CURRENT")
                print("Create table in database")
                conn = sqlite3.connect("QSCAILD.db")
                cur = conn.cursor()
                cur.execute(
                    "CREATE TABLE configurations (id integer, iteration"
                    " integer, displacements text, probability real,"
                    " current_proba real, forces text, energy real,"
                    " har_forces text, har_energy real)")
                conn.commit()
                conn.close()
                if calc_symm:
                    symmetry.save_symmetry_information_3rd(
                        [n[0], n[1], n[2], cutoff], third, symm_acoustic)
            calc_dirs=renew_configurations(nconf, T, n, iteration, "POSCAR", "SPOSCAR",
                                 "FORCE_CONSTANTS", use_smalldisp,
                                 imaginary_freq, grid)
            comm.Barrier()
            if rank == 0:
                with open("iteration", "w") as f:
                    f.write(str(iteration) + "\n")
        return
    comm.Barrier()
    mat_rec_ac_3rd_shape=None
    mat_rec_ac_shape=None    
    if rank == 0:
        with open("iteration", "r") as f:
            iteration = int(f.readline().split()[0])
        if not os.path.isfile("QSCAILD.db"):
            with open("finished", "w") as file:
                file.write("finished: error\n")
            comm.Abort()
            sys.exit("Problem: previous iterations but no database is present,"
                     " remove file iteration")
        with open("out_energy", 'a') as file:
            file.write("iteration: " + str(iteration) + "\n")
        with open("out_fit", 'a') as file:
            file.write("iteration: " + str(iteration) + "\n")
        gradient.store_vasp_forces_energy(iteration)

        iteration_min = int(math.floor(iteration * (1.0 - memory)))
        if os.path.isfile("FORCE_CONSTANTS_CURRENT"):
            shutil.copy("FORCE_CONSTANTS_CURRENT", "FORCE_CONSTANTS_PREVIOUS")

        print("Load 2nd order symmetry information")
        mat_rec_ac = np.load("../mat_rec_ac_2nd_" + str(n[0]) + "x" +
                             str(n[1]) + "x" + str(n[2]) + ".npy", allow_pickle=True)
        mat_rec_ac_shape=mat_rec_ac.shape
        nirr_ac = len(mat_rec_ac)
        poscar = generate_conf.read_POSCAR("POSCAR")

        natoms = poscar["numbers"].sum() * n[0] * n[1] * n[2]
        if third:
            sposcar = thirdorder_common.gen_SPOSCAR(poscar, n[0], n[1], n[2])
            print("Load 3rd order symmetry information")
            wedge, list4, dmin, nequi, shifts, frange = thirdorder_save.save(
                "return", n[0], n[1], n[2], cutoff)
            mat_rec_ac_3rd = np.load("../mat_rec_ac_3rd_" + str(n[0]) + "x" +
                                     str(n[1]) + "x" + str(n[2]) + "_" +
                                     str(cutoff) + ".npy", allow_pickle=True)[()]
            print(("shape: " + str(mat_rec_ac_3rd.shape)))
            mat_rec_ac_3rd_shape=mat_rec_ac_3rd.shape
            if (symm_acoustic):
                ker_ac_3rd = np.load("../ker_ac_3rd_" + str(n[0]) + "x" +
                                     str(n[1]) + "x" + str(n[2]) + "_" +
                                     str(cutoff) + ".npy", allow_pickle=True)
            else:
                ker_ac_3rd = np.identity(mat_rec_ac_3rd.shape[0])
    
    if third and rank != 0:
    #    mat_rec_ac = np.load("../mat_rec_ac_2nd_" + str(n[0]) + "x" +
    #                         str(n[1]) + "x" + str(n[2]) + ".npy", allow_pickle=True)
        mat_rec_ac_3rd = np.load("../mat_rec_ac_3rd_" + str(n[0]) + "x" +
                                    str(n[1]) + "x" + str(n[2]) + "_" +
                                    str(cutoff) + ".npy", allow_pickle=True)[()]


    comm.Barrier()
    #Parallel computation of the displacement matrices
    M, N = symmetry.calc_cells_dispmats_paral(n,rank,size)
    
    rcv_buf_M=np.empty((M.shape[0],M.shape[1],M.shape[2]))
    rcv_buf_N=np.empty((N.shape[0],N.shape[1]))
    comm.Barrier()
    comm.Allreduce(N,rcv_buf_N, op=MPI.SUM)
    comm.Allreduce(M,rcv_buf_M, op=MPI.SUM)
    N=rcv_buf_N
    M=rcv_buf_M
    os.sync()
    comm.Barrier()
    if rank==0:
        toc=time.perf_counter()
        print(f"Symmetry related parts calculated in {toc - tic:0.4f} seconds")
        tic=time.perf_counter()

    if third: # Parallel computation of third order forces
        if rank==0:    
            print("Calculating 3rd order part")
            x_data, y_data, weights,sposcar,config = gradient.prep_prepare_fit_3rd_weights(
                mat_rec_ac, mat_rec_ac_3rd, M, N, enforce_acoustic, iteration_min)
            toc=time.perf_counter()
            print(f"Weight of previous configs calculated in {toc - tic:0.4f} seconds")
            tic=time.perf_counter()
            #function arguments for third order FCs
            args=[enforce_acoustic,
                            iteration_min,sposcar,config]
            xdata_shape=(mat_rec_ac_3rd.shape[0],len(config)*len(gradient.calc_3rd_forces(mat_rec_ac_3rd,M,N,json.loads(config[0][1]))))
            krange=mat_rec_ac_3rd.shape[0]
        
        else:
            xdata_shape=None
            args=None
            krange=None
        os.sync()
        comm.Barrier()
        #Transfer to all nodes
        args=comm.bcast(args,root=0)
        xdata_shape=comm.bcast(xdata_shape,root=0)
        xdata=np.zeros(xdata_shape)
        krange=comm.bcast(krange,root=0)
        comm.Barrier()
        #Distribution of the calculation
        for k in range(rank,krange,size):
            xdata[k]=gradient.parallel_loop(k, mat_rec_ac_3rd, M, N, *args)
        
        #Transfering back to rank 0
        recv_buf=None 
        if rank==0:
            recv_buf=np.empty([xdata.shape[0],xdata.shape[1]])
        os.sync()
        comm.Barrier()
        comm.Reduce(xdata,recv_buf,op=MPI.SUM,root=0)
        comm.Barrier()
        if rank==0:
            xdata_3rd=recv_buf
            x_data = np.concatenate((x_data, np.transpose(np.array(xdata_3rd))),axis=1)
            #np.savetxt("test.xdata",x_data)
        
    else:
        if rank==0:
            x_data, y_data, weights = gradient.prepare_fit_weights(mat_rec_ac, enforce_acoustic, iteration_min)

    if rank==0:
        toc=time.perf_counter()
        print(f"Preparation of forces in {toc - tic:0.4f} seconds")
        tic=time.perf_counter()

    if rank==0:   
        clf = linear_model.LinearRegression(fit_intercept=False, n_jobs=-1)
        clf.fit(x_data, y_data, weights)

        if third:
            coef_2nd = clf.coef_[:nirr_ac]
            coef_3rd = clf.coef_[nirr_ac:]
        else:
            coef_2nd = clf.coef_

        with open("out_fit", 'a') as file:
            file.write("fit 2nd: " + str(coef_2nd.tolist()) + "\n")
            if third:
                file.write("fit 3rd: " + str(coef_3rd.tolist()) + "\n")
            file.write("score: "+str(clf.score(x_data,y_data,weights))+"\n")
        force_constants = symmetry.reconstruct_fc_acoustic_frommat(
            mat_rec_ac, np.array(coef_2nd))
        gradient.print_FORCE_CONSTANTS(force_constants,
                                       "FORCE_CONSTANTS_FIT_" + str(iteration))
        if iteration == 1 and not use_smalldisp:
            force_constants = (
                1.0 - mixing
            ) * force_constants + mixing * gradient.read_FORCE_CONSTANTS(
                "SPOSCAR", "FORCE_CONSTANTS")
        elif not use_smalldisp:
            force_constants = (
                1.0 - mixing
            ) * force_constants + mixing * gradient.read_FORCE_CONSTANTS(
                "SPOSCAR_CURRENT", "FORCE_CONSTANTS_CURRENT")
        gradient.print_FORCE_CONSTANTS(force_constants,
                                       "FORCE_CONSTANTS_CURRENT")


        if third:
            phifull = symmetry.reconstruct_3rd_fcs(poscar, sposcar, ker_ac_3rd,
                                                   coef_3rd, wedge, list4,
                                                   symm_acoustic)
            thirdorder_common.write_ifcs(
                phifull, poscar, sposcar, dmin, nequi, shifts, frange,
                "FORCE_CONSTANTS_FIT_3RD_" + str(iteration))
            thirdorder_common.write_ifcs(phifull, poscar, sposcar, dmin, nequi,
                                         shifts, frange,
                                         "FORCE_CONSTANTS_CURRENT_3RD")

        poscar_current = generate_conf.read_POSCAR("POSCAR_CURRENT")
        sposcar_current = generate_conf.read_POSCAR("SPOSCAR_CURRENT")
        factor = np.array([1.0, 1.0, 1.0])

    os.sync()
    comm.Barrier()
    if rank==0:
        toc=time.perf_counter()
        print(f"Fitting forces in {toc - tic:0.4f} seconds")
        tic=time.perf_counter()

    #Compute external pressure and updates lattice parameter if necessary
    if use_pressure in ['cubic', 'tetragonal', 'orthorhombic','fixab']:

        if os.path.isfile("POSCAR_PARAM") and os.path.isfile(
                "SPOSCAR_PARAM") and rank == 0:
            poscar_param = generate_conf.read_POSCAR("POSCAR_PARAM")
            poscar_param["lattvec"] = poscar_current["lattvec"]
            generate_conf.write_POSCAR(poscar_param, "POSCAR_PARAM")
            sposcar_param = generate_conf.read_POSCAR("SPOSCAR_PARAM")
            sposcar_param["lattvec"] = sposcar_current["lattvec"]
            generate_conf.write_POSCAR(sposcar_param, "SPOSCAR_PARAM")

        if grid > 0 and write_gruneisen and third:
            f_grun, m_grun = gruneisen.write_mode_gruneisen(
                "POSCAR_CURRENT", n, "FORCE_CONSTANTS_CURRENT",
                "FORCE_CONSTANTS_CURRENT_3RD", imaginary_freq, grid,
                "mode_gruneisen")

        if rank == 0:

            if grid == 0 and write_gruneisen and third:
                f_grun, m_grun = gruneisen.write_mode_gruneisen_gamma(
                    "POSCAR_CURRENT", "SPOSCAR_CURRENT", n,
                    "FORCE_CONSTANTS_CURRENT", "FORCE_CONSTANTS_CURRENT_3RD",
                    imaginary_freq, "mode_gruneisen")

            if write_gruneisen and third:
                gruneisen.write_weighted_gruneisen(
                    f_grun, m_grun, [Ti * 100. for Ti in range(1, 16)],
                    "weighted_gruneisen")

            potential_pressure = np.diag(
                gradient.calc_mean_stress_weights(iteration_min, weights))
            kinetic_pressure = gradient.calc_kinetic_term(
                iteration_min, weights)
            mean_pressure = potential_pressure + kinetic_pressure

            for i in range(3):
                if (mean_pressure[i] > pressure[i] + pdiff / 2.):
                    factor[i] = 1. + 0.001 * (min(
                        (mean_pressure[i] - pressure[i]) / 10., 4.))
                elif (mean_pressure[i] < pressure[i] - pdiff / 2.):
                    factor[i] = 1. - 0.001 * (min(
                        (pressure[i] - mean_pressure[i]) / 10., 4.))

            #Symmetrize stress tensor
            if use_pressure == 'cubic':
                factor = np.array(
                    [np.mean(factor),
                     np.mean(factor),
                     np.mean(factor)])

            if use_pressure == 'tetragonal':
                factor = np.array([
                    0.5 * factor[0] + 0.5 * factor[1],
                    0.5 * factor[0] + 0.5 * factor[1], factor[2]
                ])

            if use_pressure == 'fixab':
                factor = np.array([1,1, factor[2]])


            if use_pressure == 'cubic':
                sposcar_current[
                    "lattvec"] = sposcar_current["lattvec"] * factor[0]
                poscar_current[
                    "lattvec"] = poscar_current["lattvec"] * factor[0]
            else:
                # Can only handle diagonal lattice vectors if the unit cell
                # is non-cubic
                for i in range(3):
                    sposcar_current["lattvec"][
                        i, i] = sposcar_current["lattvec"][i, i] * factor[i]
                    poscar_current["lattvec"][
                        i, i] = poscar_current["lattvec"][i, i] * factor[i]



            # Update atomic positions based on the symmetrized forces and on force constants
            if optimize_positions:
                disp = gradient.disp_optimize_positions_weights(
                    force_constants, 'SPOSCAR', iteration_min, weights)

                corresp = symmetry.calc_corresp(poscar_current, sposcar_current, n)
                folded_disp = np.zeros(poscar["numbers"].sum()*3)
                for iatom in range(len(corresp)):
                    for direction in range(3):
                        folded_disp[corresp[iatom][0]*3+direction] = disp[iatom*3+direction]
                drift=np.sum(folded_disp)

                with open("out_atomic_positions", 'a') as file:
                    file.write("iteration: " + str(iteration) + "\n")
                    file.write("displacement in nm: " + str(disp.tolist()) + "\n")
                    file.write("folded displacement in nm: "
                                + str(folded_disp.tolist()) + "\n")
                    #file.write("total drift: " + str(drift.tolist()) + "\n")

                sposcar_current = generate_conf.distort_POSCAR(sposcar_current, disp)
                poscar_current = generate_conf.distort_POSCAR(poscar_current, folded_disp)

            generate_conf.write_POSCAR(poscar_current, "POSCAR_CURRENT")
            generate_conf.write_POSCAR(sposcar_current, "SPOSCAR_CURRENT")
            if use_pressure in ['cubic', 'tetragonal', 'orthorombic']:
                with open("out_volume", 'a') as file:
                    file.write("iteration: " + str(iteration) + "\n")
                    file.write("kinetic pressure: " +
                               str(kinetic_pressure.tolist()) + "\n")
                    file.write("potential pressure: " +
                               str(potential_pressure.tolist()) + "\n")
                    file.write("mean pressure: " +
                               str(mean_pressure.tolist()) + "\n")
                    file.write("lattice vectors: " +
                               str(poscar_current["lattvec"].tolist()) + "\n")
        else:
            mean_pressure = None
        mean_pressure = comm.bcast(mean_pressure, root=0)

    os.sync()
    iteration = comm.bcast(iteration, root=0)
    comm.Barrier()
    if rank==0:
        toc=time.perf_counter()
        print(f"Structural optimization  in {toc - tic:0.4f} seconds")
        tic=time.perf_counter()

    if iteration >= nfits:
        if rank == 0:
            with open("finished", "w") as file:
                file.write("finished: maximum iteration number\n")
        return

    if test_convergence(iteration, tolerance):
        if not use_pressure in ['cubic', 'tetragonal', 'orthorombic']:
            if rank == 0:
                with open("finished", "w") as file:
                    file.write("finished: obtained convergence\n")
            return
        elif np.amax(np.abs(mean_pressure - pressure)) < pdiff:
            if rank == 0:
                with open("finished", "w") as file:
                    file.write("finished: obtained convergence\n")
            return

    iteration += 1

    calc_dirs=renew_configurations(nconf, T, n, iteration, "POSCAR_CURRENT",
                         "SPOSCAR_CURRENT", "FORCE_CONSTANTS_CURRENT",
                         use_smalldisp, imaginary_freq, grid)
    if rank == 0:
        with open("iteration", "w") as f:
            f.write(str(iteration) + "\n")
    os.sync()
    if rank==0:
        toc=time.perf_counter()
        print(f"Renew configurations  in {toc - tic:0.4f} seconds")
        tic=time.perf_counter()
    
    return 


def renew_configurations(nconf, T, n, iteration, poscar_file, sposcar_file,
                         fcs_file, use_smalldisp, imaginary_freq, grid):
    """
    Generates a new set of configurations and submit the DFT jobs.
    """

    if rank == 0:

        print("Generate new set of configurations")

    dirs = generate_conf.prepare_conf(nconf, iteration, poscar_file,
                                      sposcar_file, fcs_file, T, n,
                                      use_smalldisp, imaginary_freq, grid)
    if rank == 0:
        with open("to_calc", "w") as file:
            for r in dirs:
                file.write(r + "\n")

    return


def test_convergence(iteration, tolerance):
    """
    Test convergence of the fitted force constants.
    """
    if iteration < 2:
        return False
    if rank == 0:
        fcs_current = gradient.read_FORCE_CONSTANTS("SPOSCAR_CURRENT",
                                                    "FORCE_CONSTANTS_CURRENT")
        fcs_previous = gradient.read_FORCE_CONSTANTS(
            "SPOSCAR_CURRENT", "FORCE_CONSTANTS_PREVIOUS")
        with open("out_convergence", 'a') as file:
            file.write("iteration: " + str(iteration) + "\n")
            file.write("fcs max absolute difference: " +
                       str(np.amax(np.absolute(fcs_current - fcs_previous))) +
                       "\n")
            file.write("fcs mean absolute difference: " +
                       str(np.mean(np.absolute(fcs_current - fcs_previous))) +
                       "\n")
            file.write("fcs max relative difference: " + str(
                np.nanmax(
                    np.divide(
                        np.absolute(fcs_current - fcs_previous),
                        np.absolute(fcs_previous)))) + "\n")
            nruter = np.allclose(fcs_current, fcs_previous, atol=tolerance)
    else:
        nruter = None
    nruter = comm.bcast(nruter, root=0)
    return nruter
