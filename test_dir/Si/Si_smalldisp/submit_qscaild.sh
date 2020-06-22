#!/bin/bash
#PBS -N test_Si
#PBS -S /bin/bash
#PBS -q short
#PBS -l walltime=24:00:00,nodes=2:ppn=16

ulimit -s unlimited
module purge
module load icc/18 impi/18 mkl/18
module load python/anaconda3

cd $PBS_O_WORKDIR

while :
do
    mpirun -np $PBS_NP python ../../../submit_qscaild.py
    if [ -f finished ]
    then
        break
    fi
    while IFS='' read -r line || [[ -n "$line" ]]
    do
        folder="$line"
        echo "Run VASP for directory - $folder"
        cd $folder
        mpirun -np $PBS_NP /home/av245900/Codes/VASP_5.4.4_csp/vasp.5.4.4/bin/vasp_gam </dev/null
        cd ..
    done < "to_calc"
done

