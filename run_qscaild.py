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

"""  This script is a wrapper automatically performing the  QSCAILD calculations and allows to automatically train and use  machine learned potential
  In order for it to work, it is necessary to define the executables in calculator.config.py script 
  
  For the use of machine learning potentals, MLIP_mode must be available in the parameters file, as well as an absolute path to the potential and the training set

  MLIP_mode allows four different options:
  (i) 'off': Calculate only using vasp
  (ii) 'mlip_only': Calculate only using the MLIP
  (iii) 'active_learning': Use active learning to automatically train and use an MLIP
  (iv) 'train': Create a training file from all the configurations present in the folder, continue the qscaild run using active learning
"""

import os
import sys
import os.path
import shutil
import subprocess
import glob
import pprint
import datetime
import logging
import calculator
import mlip2vasp
import time
import numpy as np




def str2bool(v):
    return v.lower().strip() in ("yes", "true", "t", "1")

# Use machine learning interatomic potentials (requires MLIP installation)
# MLIP mode ("train": train on all current configurations, "active_learning": enable on the fly learning, "mlip_only": use only potential, no active learning)
MLIP_mode = "off"
#Location of training set
MLIP_train_set = "train.cfg"
#Location of potential
MLIP_potential = "pot.mtp"



# Read input file
with open("parameters", 'r') as f:
    for line in f.readlines():
        if 'MLIP_mode' in line:
            MLIP_mode = str(line.split("=")[1]).strip()
        if 'MLIP_train_set' in line:
            MLIP_train_set = str(line.split("=")[1]).strip()
        if 'MLIP_potential' in line:
            MLIP_potential = str(line.split("=")[1]).strip()

print("MLIP mode = " + str(MLIP_mode))
print("Training set =" + MLIP_train_set)
print("MLIP potential =" + MLIP_potential)

#Gather data from previously calculated DFT configs and train potential, then continue with active learning
if MLIP_mode == "train":
    print("Gathering training data from all configs, then train new potential and continue with active learning")
    for i in [ d for d in os.listdir() if "config" in d ]:
        calculator.add_to_train(i,MLIP_train_set)
    calculator.train(MLIP_train_set, MLIP_potential)
    MLIP_mode="active_learning"

while not os.path.isfile('finished'):
    
    print("Running QSCAILD iteration")
    calculator.qscaild()
    calc_dirs=[]
    with open('to_calc') as f:
        line=f.readline()
        while line:
            calc_dirs.append(line.strip())
            line=f.readline()
    print(str(MLIP_mode), "off")
    print(MLIP_mode == "off")

    #DFT only mode
    if str(MLIP_mode) == "off":
        for i in calc_dirs:
            print("vasp calculation for"+i)
            calculator.vasprun(i)
    
    #Using MLIP
    if MLIP_mode != "off":
        #create cfg files
        for dirs in calc_dirs:
            mlip2vasp.poscar2cfg(mlip2vasp.read_POSCAR(os.path.join(dirs,"POSCAR")),os.path.join(dirs,"config.cfg"))

        #Active learning loop
        if MLIP_mode=="active_learning":
            print("Active learning: Calculate grades")
            grades=[]
            for i in calc_dirs:
                grade=calculator.grade(i,MLIP_train_set, MLIP_potential)
                if grade != 0:
                    calculator.vasprun(i)
                    calculator.add_to_train(i,MLIP_train_set)
                    grades.append(grade)
            if sum(grades) > 0:
                calculator.train(MLIP_train_set, MLIP_potential)
        
        #Use only MLIP
        if MLIP_mode == "mlip_only":
            for i in calc_dirs:
                print(i)
                calculator.mlip(i,MLIP_potential)

