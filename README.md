# README

## General Info

This code computes 2nd- and 3rd-order force constants from small displacements and at finite temperature. It was developed by Ambroise van Roekeghem, Quintin N. Meier, Jesús Carrete, and Natalio Mingo.

The software is primarily written in Python 3 and depends on several packages, including:
- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [mpi4py](https://github.com/mpi4py/mpi4py)
- [Phonopy](https://atztogo.github.io/phonopy/)
- [Cython](https://cython.org/)
- The C version of [spglib](https://atztogo.github.io/spglib/) by Atsushi Togo

Currently, the DFT calculations rely on [VASP](https://www.vasp.at/), but the program could be adapted for other DFT codes.

The third-order force constants module, derived from [thirdorder.py](https://bitbucket.org/sousaw/thirdorder/), requires compilation using `./compile.sh` after modifying `setup.py` to match the system.

The code supports active learning

## Step-by-Step Installation

1. **Install Dependencies via Conda**
   - Create and activate a new Conda environment:
     ```sh
     conda create -n qscaild_env python=3.10
     conda activate qscaild_env
     ```
   - Install dependencies:
     ```sh
     conda install numpy scipy scikit-learn mpi4py cython phonopy -c conda-forge
     ```
   - Ensure VASP and the C version of [spglib](https://github.com/spglib/spglib) are installed on your system.
   
   - QSCAILD supports the use of moment tensor potentials, which need to be compiled separately
   - Download and compilation instructions for the mlip package: [mlip](https://gitlab.com/ashapeev/mlip-2)
   
2. **Compile the Third-Order Code**
   - Navigate to the directory containing `setup.py`.
   - Modify `setup.py` if necessary for your system.
   - Run:
     ```sh
     ./compile.sh
     ```  

3. **Adjust `calculator_config.py`**
   - Open `calculator_config.py` in a text editor:
     ```sh
     nano $PATH_TO_THE_CODE/calculator_config.py
     ```
   - Modify the following lines to specify the correct paths:
     ```python
     qscaild_path = "/path/to/qscaild" # installation directory of the qscaild code  
     vasp_exe="/path/to/vasp_std" #installation directory of vasp
     mlip_exe="/path/to/mlp" #path to mlip exectuable (if machine learning potentials are used)
     vasp_env=None #adjust using os.env if special modules need to be used for vasp
     mlip_env=None #adjust using os.env if special modules need to be used for mlip
     mpirun="mpirun" #MPI command on your system
     ```
## Running the code
1. **Prepare Input Files**
   - `parameters` contains all the input parameters for qscaild
   - `POSCAR` contains the atomic positions of the unit cell
   - `SPOSCAR` contains the atomic positions of the supercell
   - `POTCAR`,`INCAR` and `KPOINTS` are parameters for the DFT calculations
   - `FORCE_CONSTANTS` (optional) contains the initial force constants.
   
   When creating the input files, the supercell (`SPOSCAR`) can easily be created using phonopy:
   ```sh
   phonopy -d --dim="n0 n1 n2"
   ```
   where `n0` `n1` and `n2` describe the supercell dimension along each axis.
   If the `FORCE_CONSTANTS` are not provided, an initial set of force constants must be calculated using `use_smalldisp=True`.
   
2. If machine learning potentials are used, prepare a training set `train.cfg` and an (empty) potential `pot.mtp`

3. **Run the Code**
   - Execute the program (internally calls MPI)
     ```sh
     python /path/to/qscaild/run_qscaild.py

     ```



## Input Parameters

### Core Parameters
- **Temperature (Kelvin):** `T_K = 500`
- **Number of displaced configurations:** `nconf = 10`
- **Number of cycles:** `nfits = 5`
- **Supercell size:** `n0 = 3`, `n1 = 3`, `n2 = 3`
- **3rd-order force constants cutoff:** `cutoff = -5`
- **Enable 3rd order calculation:** `third = True`

### Volume and Pressure Control
- **Iterative equilibrium volume calculation:** `use_pressure = cubic`  (Other options are "tetragonal" or "orthorhombic")
- **Target stress tensor diagonal (in kB):** `pressure_diag = 0.,0.,0.`

### Execution Options
- **Use small displacements:** `use_smalldisp = False`
- **Compute symmetry matrices:** `calc_symm = True`
- **Apply acoustic sum rule:** `symm_acoustic = True`
- **Replace imaginary frequencies:** `imaginary_freq = 1.0`
- **Mixing between cycles:** `mixing = 0.6`
- **Memory between cycles:** `memory = 0.4`
- **Thermal displacement matrix grid size:** `grid = 20`
- **Convergence tolerance:** `tolerance = 0.01`
- **Pressure convergence tolerance:** `pdiff = 2.0`
- **Experimental acoustic sum rule enforcement:** `enforce_acoustic = False`

### MLIP and active learning
- **MLIP_mode:** `MLIP_mode="off"` ('off'=DFT, 'mlip'=use machine learning potential, 'active_learning'= create mlip potential using active learning)
- **MLIP_potential:** `MLIP_potential="/path/to/MLIP_potential"`
- **MLIP_train_set:** `MLIP_train_set="/path/to/MLIP_train_set"`


## Short Tutorial

### Example 1: Small Displacements
1. Navigate to `test_dir/Si/Si_smalldisp`.
2. Run:
   ```sh
   python /path/to/qscaild/run_qscaild.py
   ```
3. The program computes symmetries and generates `config-*` directories.
4. The program runs VASP in each `config-*` directory.
5. Disable symmetry computation (`calc_symm = False`) in `parameters`.
6. Relaunch the program to produce `FORCE_CONSTANTS`.

### Example 2: Finite Temperature Calculation
1. Navigate to `test_dir/Si/Si_500K`.
2. Run the program to generate `config-*` directories.
3. Compute forces in each directory and relaunch the program.
4. Check `out_convergence` for cycle convergence.

### Example 3: Thermal Expansion
1. Follow the same workflow as `Si_500K`, but in `test_dir/Si/Si_500K_volume`.
2. Other examples, such as `test_dir/SrTiO3`, are available for more complex cases.

This README provides an overview of the workflow and execution of the program. Further system-specific adaptations may be required.

