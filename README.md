This code is used to count the atoms of ions inside layers of LDH from the molecular dynamics simulation with GROMACS and the frame saved in a extension of *.gro. The SimplyCounterAtoms is a code that reads the gro file and search the metals atoms of LDH cluster with scikit-learn modules. The Scipy package proportion the Hull of the LDH layers and find the atoms inside the hull. The code return the Volume of hull and the number of atoms inside it (Chlorine, water and Nitrates/MBT).
The CounterAtoms.py do the same but the estructure of gro file is expanded in the 3 dimension (3x3x3 unit cells) and locate the cluster in the center to cut a sample of 10 nm from the center. This reduce the possibility of errors by the atoms that crosses the pbc box during the molecular dynamic simulation.
A new GRO file is generated with GraphicalCounterAtoms.py 
In order to be able a graphical tool to observe the atoms that have been counted inside the LDH layers with the CounterAtoms.py program, a new GRO file has been generated with GraphicalCounterAtoms.py. This extracted all the atoms around the LDH and the Cl, NO3, MBT, and water molecules inside the LDH layers are marked as CLi, N1i, S1i, and OWi, respectively. The metallic atoms (Al and Zn) of LDH layers have been marked with a resname CAP and with AU in the name of atom, for an easy selection in the visualization with the VMD software.

This code was developed in the frame of project DataCor (POCI-01-0145-FEDER-030256 and PTDC/QUI-QFI/30256/2017, https://datacoproject.wixsite.com/datacor) and published in Nanomaterials 2022, 12(22), 4039; (10.3390/nano12224039) with the title "Molecular Dynamics Model to Explore the Initial Stages of Anion Exchange involving Layered Double Hydroxide Particles" (https://www.mdpi.com/2079-4991/12/22/4039).
