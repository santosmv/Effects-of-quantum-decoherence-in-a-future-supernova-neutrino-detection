# Effects of quantum decoherence in a future supernova neutrino detection

This repository is destinated to calculate propagation and detection of supernova (SN) neutrinos until Earth considering the impact of quantum decoherence (open quantum systems formalism).

It was created a system that generates statistical chi-square profiles and contours just running the file *run_and_minimize_chi_square.py*. You have 2 choices after running this file:
1. Selecting the desired analysis (**not implemented**)
2. Select an automated run, where all results for a specific distance D (in kpc) will be obtained

This system and its files are described as follows (in order of importation):

- **run_and_minimize_chi_square.py**: initial file (to be run). Here you find the selected options of analysis and also the `iminuit` (`Minuit`) minimization functions;

- **menu.py**: is an auxiliar file that is responsible to generate a menu and save the options in `config_list` variable;

- **chi_square.py**: it has the chi square functions for DUNE and HK  and also the combined for quantum decoherence effects. The same is valid for **chi_square_Pee.py** (used to calculate statistical limits on a free constant Pee) and **chi_square_Pi.py** (used to calculate P1e(x), P2e(x) and P3e(x) for the ternary plot);

- **rate.py**: events rate for quantum decoherence parameters (same is valid for **rate_Pee.py** and **rate_Pi.py**, but as said above, for a different kind of analysis).

- **flavor_conversion.py**: this file has most of standard and non-standard mixing probabilities;

- **interaction.py**: meets the cross-section functions;

- **fluxes.py**: has the functions that collect Garching fluxes from simulations;


There are some other files those are not directly part of this chain (system) but are important:

- **create_tables.py**: generate tables of the database used to save results;

- **Pij_sn.py** (and **Pij_sn_loss.py**): save the probability of a neutrino i goes to a j in SN matter profile in a compiled file into `compiled_numba` folder;

The results are save into `database` files in `data` folder and also in `results` folder in `.npy` format.

The IPython notebooks **regeneration.ipynb** and **SN_slab_approach.ipynb** describe in details the solution of standard (and non-standard) neutrino mixing at Earth and SN. 

The files with subscript `_loss` regard a test over a different model of neutrino loss along propagation. Files without this subscript will concern mass state coupling scenario. **Note**: in loos probability code, we included  the 2 factor multiplying Gamma, however in the plots_paper.ipynb, we absorb this factor dividing the results by 2.

Different choices of zenith were calculated in order to quantify regeneration effects. Angles between 270 and 90 degrees will not suffer Earth matter effects. The angles 120, 140, 160 and 180 were chosen for DUNE, then the zenith for HK and JUNO were calculated in the DUNE longitude. In this way there are situations where DUNE has Earth matter effects but the other detectors not and the opposite is also true.

Sensible data, usually from made available from other research groups was omitted in the `data` folder, which can make the task of reproducing the finding of the repository impracticable. Make contact with the repository manager (santosmv) to have more information on the access to such data.