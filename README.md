# multiphoton-selective-excitation

This is the code repository for Pulsed Selective Excitation in MRI.
To generate figures 2-6 in the paper, please install the following dependencies and run "python main.py". To generate a supplementary figure, simply run "python supplementary_figure#.py", where # should be replaced by the supplementary figure number.\
The *.npy files contain the experimental image data for their respective figures. Note that because the simulation code runs full 3D Boch equation simulations using an ODE solver, it takes about half an hour to run on a M1 Macbook Pro.

# Dependencies: 
    sigpy (https://sigpy.readthedocs.io/en/latest/mri_rf.html)
    numpy
    scipy
    matplotlib
