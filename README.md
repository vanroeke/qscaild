## General description

This code allows computing the 2nd- and 3rd-order force constants from small displacements and at finite temperature. It was contributed by Ambroise van Roekeghem, Jesús Carrete and Natalio Mingo.

The software is mainly written in Python 3 and uses some common packages: [numpy](http://www.numpy.org/), [scipy](http://www.scipy.org/), [scikit-learn](https://scikit-learn.org/), and [mpi4py](https://github.com/mpi4py/mpi4py). It also requires [Phonopy](https://atztogo.github.io/phonopy/) and the C version of [spglib](https://atztogo.github.io/spglib/) by Atsushi Togo. The current DFT part is based on [VASP](https://www.vasp.at/), but the program could be easily adapted to be used with other DFT codes.

The third-order part of this code, adapted from the original [thirdorder.py](https://bitbucket.org/sousaw/thirdorder/), should be recompiled for your system by running `./compile.sh` after modifying the `setup.py` file.

At present, the code can only handle isotropic thermal expansion or orthogonal lattices. There is an option to update the equilibrium position of atoms with a free parameter as a function of temperature, that requires additional `POSCAR_PARAM` and `SPOSCAR_PARAM` input files and is not documented yet.

## Workflow and input files

The main input file is called `parameters`, examples of which are given in the `test_dir` directory. Other input files include `POSCAR`, `SPOSCAR` (created by Phonopy), `POTCAR`, `INCAR` and `KPOINTS` files for the DFT calculations, and a starting `FORCE_CONSTANTS` file if small displacements are not used.

The symmetry-related parts of the program are not parallelized and have to be run once for a given system and supercell size. The symmetry files are written in the root directory and can be reused for different temperatures and volumes (however, be careful that the cutoff introduced in the 3rd-order part has to stay coherent with your new system).

Running the program performs one part of the self-consistent loop required by quantum self-consistent ab initio lattice dynamics: it reads the outputs of the previous DFT runs to compute the current interatomic force constants (or reads the starting force constants for the first loop), and produces input for the next DFT runs. The user then has to perform the DFT calculations by themselves (the relevant directories are mentioned in the `to_calc` file), before launching the program again for the next loop. Examples of how the full self-consistent loop can be performed within one single job are shown in the `test_dir` directory, but in general this will be system-dependent and for this reason additional support cannot be provided. To run one iteration of the program, just launch `mpirun -np $NUMBER_OF_CORES python $PATH_TO_THE_CODE/submit_qscaild.py` (or `python $PATH_TO_THE_CODE/submit_qscaild.py` for the serial version) from the folder containing your input files.

The program internally performs reweighting of the different configurations according to the current force constants, and convergence tests.

Below is an explanation of the value of the parameters:

* temperature of the calculation in kelvin

```
T_K = 500
```

* number of displaced configurations in each cycle. Usually a total number of forces 10 times larger than the number of irreducible elements is a safe choice

```
nconf = 10
```

* number of cycles. Usually 20 is more than enough to obtain convergence if the structure is kept fixed, otherwise it can be much longer

```
nfits = 5
```

* supercell size

```
n0 = 3
n1 = 3
n2 = 3
```

* cutoff for the third order force constants. This can either be a length in nm or a negative integer specifying a number of nearest neighbors.

```
cutoff = -5
```

* whether to calculate 3rd order force constants or not

```
third = True
```

* calculate the pressure to obtain the equilibrium volume iteratively. Values can be 'cubic', 'tetragonal', 'orthorhombic' or 'False', which means that no thermal expansion is considered

```
use_pressure = cubic
```

* target for the diagonal of the stress tensor, in multiples kB

```
pressure_diag = 0.,0.,0.
```

* if True, use small displacements (in that case the value of the temperature is not used, the initial `FORCE_CONSTANTS` file is not necessary and one cycle is enough)

```
use_smalldisp = False
```

* calculate the symmetry matrices that will be saved in the root directory (has to be done once for a given supercell and cutoff)

```
calc_symm = True
```

* apply acoustic sum rule in the calculation of the symmetry matrices (recommended if computationally feasible)

```
symm_acoustic = True
```

* all imaginary frequencies are replaced with this value. If -1 is used, "negative" frequencies are switched to positive like in the original SCAILD method

```
imaginary_freq = 1.0
```

* mixing between two cycles to stabilize convergence (here, 60% of the force constants of the previous cycle are kept)

```
mixing = 0.6
```

* memory between different cycles, if equal to 1 all computed configurations are taken into account with their proper reweighting. Mind that the reweighing scheme is not correct if the volume varies, so memory should be decreased strongly in that case.

```
memory = 0.4
```

* size of the (uniform in each direction) grid to compute the thermal displacement matrix and Grüneisen parameters

```
grid = 20
```

* tolerance for the convergence of the force constants

```
tolerance = 0.01
```

* tolerance for the convergence of the target pressure

```
pdiff = 2.0
```

* non-recommended, experimental way to enforce the acoustic sum rule in 2nd-order force constants when it cannot be included in the symmetry matrices

```
enforce_acoustic = False
```

## Short tutorial

Download the code and install all the dependencies on your system, then change directory to `$PATH_TO_YOUR_PROGRAM/test_dir/Si/Si_smalldisp`. This contains all the necessary input to compute the `FORCE_CONSTANTS` of silicon using small displacements. As you can see in the parameters file, it will also compute the symmetries of the system. You can just launch the program with `python ../../../submit_qscaild.py`. The part of the program that computes the symmetries is not parallel and can take some time to complete, depending of your system, so be patient. After completion, you should see information about the number of irreducible elements in the `out_sym` file, a bunch of directories `config-*`, and their list in the `to_calc` file. You need to run VASP in each of those folders, then switch off the computation of symmetries in the `parameters` file (`calc_symm = False`) and relaunch the program. This should produce a `FORCE_CONSTANTS` file, which will be close to what you would typically obtain using phonopy, for instance.

You can now switch to the `test_dir/Si/Si_500K` directory and launch the program (it will reuse the same symmetry files that you just computed in the previous example and that are in the `test_dir/Si` directory). Again, a bunch of `config-*` directories are produced, you need to compute the forces for each of those inputs, and after relaunching the program you will obtain a `FORCE_CONSTANTS_1` file and a `FORCE_CONSTANTS_CURRENT` file. The `FORCE_CONSTANTS_1` is the result of the fit for the first iteration, and `FORCE_CONSTANTS_CURRENT` is after mixing with the previous force constants. You also have a new list of `config-*` input folders to compute in the `to_calc` file, so you can again run VASP for those folders and relaunch the program. After the second cycle, information about the convergence of the cycle is present in the `out_convergence` file (by comparing the `FORCE_CONSTANTS_PREVIOUS` and `FORCE_CONSTANTS_CURRENT` files), so you can monitor what is happening. You should mainly be careful about possible divergences, which can happen mostly because of too low values for the imaginary_freq parameter when the spectrum is unstable, or because of too small mixing.

In the `test_dir/Si/Si_500K_volume` folder you have a similar example including thermal expansion. There is also input for similar calculations for the more interesting case of SrTiO3 in the `test_dir/SrTiO3` folder.
