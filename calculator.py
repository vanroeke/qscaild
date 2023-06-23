#This file defines the functions to run vasp and QSCAILD calculations on a cluster
#Executables need to be defined in calculator_config.py


import subprocess
import mlip2vasp as mlv
import os
from calculator_config import mlip_exe, vasp_exe, mlip_env, vasp_env, mpirun, qscaild_path
import sys
import shutil

# If the environment variables are not defined in the config, use currently environment
if not mlip_env:
    mlip_env=os.environ
if not vasp_env:
    vasp_env=os.environ

threshold=3.0


def qscaild():
    subprocess.run([mpirun,"python", os.path.join(qscaild_path,"submit_qscaild.py")],stdout=sys.stdout, stderr=sys.stderr)
    return
 
def vasprun(directory):
    subprocess.run([mpirun,vasp_exe], cwd=directory ,shell=False, env=vasp_env, stdout=sys.stdout, stderr=sys.stderr)
    return

def grade(directory,MLIP_train_set, MLIP_potential):
     subprocess.run([mpirun, "-n", "1", mlip_exe,"calc-grade", MLIP_potential, MLIP_train_set,"config.cfg", "out_grade.cfg"],cwd=directory, env=mlip_env)
     with open(os.path.join(directory,"out_grade.cfg"),'r') as f:
          for line in f.readlines():
               if line.find("MV_grade") != -1:
                    if(float(line.split()[2]) < threshold):
                        subprocess.run([mpirun,"-n", "1",mlip_exe, "calc-efs",MLIP_potential,"config.cfg", "out_efs.cfg"],cwd=directory, env=mlip_env)
                        mlv.cfg2vasprun(mlv.read_cfg(os.path.join(directory,"out_efs.cfg")), os.path.join(directory,"vasprun.xml"))
                        return 0
                    else: 
                        return 1
 
def train(MLIP_train_set, MLIP_potential):
    subprocess.run([mpirun, mlip_exe, "train", MLIP_potential, MLIP_train_set, "--force-weight=1.0", "--energy-weight=0.1", "--stress-weight=1.0", "--update-mindist", "--max-iter=2000"], env=mlip_env)
    shutil.copy('Trained.mtp_', MLIP_potential)
    return

def mlip(directory, MLIP_potential):
    subprocess.run([mpirun, "-n", "1", mlip_exe, "calc-efs" , MLIP_potential ,"config.cfg", "out_efs.cfg"],cwd=directory, shell=False, env=mlip_env)
    mlv.cfg2vasprun(mlv.read_cfg(os.path.join(directory,"out_efs.cfg")), os.path.join(directory,"vasprun.xml"))
    return

def add_to_train(directory, MLIP_train_set):
    subprocess.run([mpirun, "-n", "1", mlip_exe, "convert-cfg", os.path.join(directory,"OUTCAR"), MLIP_train_set, "--input-format=vasp-outcar", "--append"], env=mlip_env)    
    return 
