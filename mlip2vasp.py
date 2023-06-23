import numpy as np
import os


def read_POSCAR(filename):
    """
    Return all the relevant information contained in a POSCAR file.
    WARNING: in this function lattvec is in Angstroms
    """
    nruter = dict()
    nruter["lattvec"] = np.empty((3, 3))
    f = open(filename, "r")
    firstline = next(f)
    factor = 1. * float(next(f).strip())
    for i in range(3):
        nruter["lattvec"][:, i] = [float(j) for j in next(f).split()]
    nruter["lattvec"] *= factor
    line = next(f)
    fields = next(f).split()
    old = False
    try:
        int(fields[0])
    except ValueError:
        old = True
    if old:
        nruter["elements"] = firstline.split()
        nruter["numbers"] = np.array([int(i) for i in line.split()])
        typeline = "".join(fields)
    else:
        nruter["elements"] = line.split()
        nruter["numbers"] = np.array([int(i) for i in fields], dtype=np.intc)
        typeline = next(f)
    natoms = nruter["numbers"].sum()
    nruter["positions"] = np.empty((3, natoms))
    for i in range(natoms):
        nruter["positions"][:, i] = [float(j) for j in next(f).split()]
    f.close()
    nruter["types"] = []
    for i in range(len(nruter["numbers"])):
        nruter["types"] += [i] * nruter["numbers"][i]
    if typeline[0] == "C" or typeline[0] == "c":
        nruter["positions"] = np.linalg.solve(nruter["lattvec"],
                                              nruter["positions"] * factor)
    return nruter


def read_cfg(cfgfile):
    """
    Return all the relevant information contained in a cfg file (for now, only works for a single configuration).
    """
    cfg = dict()
    cfg["lattvec"] = np.empty((3, 3))
    f = open(cfgfile,"r")
    nextline = next(f)
    nextline = next(f)
    cfg["natoms"] = int(next(f).strip())
    cfg["forces"] = np.empty((cfg["natoms"],3))
    nextline = next(f)
    for i in range(3):
        cfg["lattvec"][:, i] = [float(j) for j in next(f).split()]
    nextline = next(f)
    for i in range(cfg["natoms"]):
        line = next(f).split()
        cfg["forces"][i,0] = float(line[5].strip())
        cfg["forces"][i,1] = float(line[6].strip())
        cfg["forces"][i,2] = float(line[7].strip())
    nextline = next(f)
    cfg["energy"] = float(next(f).strip())
    nextline = next(f)
    cfg["stress"] = np.array([float(s.strip()) for s in next(f).split()])
    return cfg


def poscar2cfg(poscar,cfgfile):
    """
    Convert one poscar into mlip format and append to cfg file.
    """
    with open(cfgfile,'w') as f:
        f.write("BEGIN_CFG\n")
        f.write(" Size\n")
        f.write("    "+str(poscar["numbers"].sum())+"\n")
        f.write(" Supercell\n")
        for i in range(3):
            f.write("    {0[0]:>13.6f} {0[1]:>13.6f} {0[2]:>13.6f}\n".format(
                (poscar["lattvec"][:, i]).tolist()))
        f.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z\n")
        for i in range(poscar["positions"].shape[1]):
            nelement = -1
            for element in range(len(poscar["numbers"])):
                if i < poscar["numbers"][:element+1].sum():
                    nelement=element
                    break
            f.write("    "+"{0[0]:10d} {0[1]:4d}".format([i+1,nelement]))
            f.write("  {0[0]:>13.6f} {0[1]:>13.6f} {0[2]:>13.6f}\n".format(np.dot(poscar["lattvec"], poscar["positions"])[:, i].tolist()))
        f.write("END_CFG\n")
        f.write("\n")


def cfg2vasprun(cfg,vasprunfile):
    """
    Convert one cfg into dummy vasprun.xml containing forces, stress tensor and energy.
    """
    #calculate factor to convert mlip stress in eV to VASP stress in kbar
    volume = abs(np.linalg.det(cfg["lattvec"]))
    stressfactor = 1.602177*1e3/volume
    stress = cfg["stress"]*stressfactor

    with open(vasprunfile,'w') as file:
        file.write("""<?xml version="1.0" encoding="ISO-8859-1"?>\n""")
        file.write("<modeling>\n")
        file.write(" <calculation>\n")
        file.write("""  <varray name="forces" >\n""")
        for natom in range(cfg["forces"].shape[0]):
            file.write("   <v> "+str(cfg["forces"][natom,0])+"  "+str(cfg["forces"][natom,1])+"  "+str(cfg["forces"][natom,2])+" </v>\n")
        file.write("  </varray>\n")
        file.write("""  <varray name="stress" >\n""")
        file.write("   <v> "+str(stress[0])+"  "+str(stress[5])+"  "+str(stress[4])+" </v>\n")
        file.write("   <v> "+str(stress[5])+"  "+str(stress[1])+"  "+str(stress[3])+" </v>\n")
        file.write("   <v> "+str(stress[4])+"  "+str(stress[3])+"  "+str(stress[2])+" </v>\n")
        file.write("  </varray>\n")
        file.write("  <energy>\n")
        file.write("""   <i name="e_fr_energy"> """+str(cfg["energy"])+" </i>\n")
        file.write("  </energy>\n")
        file.write(" </calculation>\n")
        file.write("</modeling>\n")

