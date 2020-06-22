#!/bin/bash
#PBS -N test_Si
#PBS -S /bin/bash
#PBS -q short
#PBS -l walltime=24:00:00,nodes=4:ppn=16

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
        mpirun -np $PBS_NP /W/av245900/HT-SHARE/vasp_sources/vasp_gam-intel15 </dev/null
        cd ..
    done < "to_calc"
done

