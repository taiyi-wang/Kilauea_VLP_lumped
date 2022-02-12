# Kilauea_VLP_lumped_param
Lumped parameter caldera collapse model, wrapper for near-field ground motion prediction, and inversion interface with EMCEE hammer (Foreman-Mackey, 2013)

This repository contains the code used for 

Wang, T., Coppess, K., Segall, P., Dunham, E M., Ellsworth, W., Physics based model reconciles caldera collapse induced static and dynamic ground motion: application to Kilauea volcano in 2018: Geophysical Research Letters (2022).

Katherine Coppess wrote the code for convolving pre-computed Green's functions (using FK; Zhu & Rivera, 2002) with source time functions and implemented analytical Green's functions. Taiyi Wang wrote the code for the dynamic model, interface with Bayesian inversion, and other relevant code.

## Dynamic model
1. calcol.py: analytical solution to caldera collpase with static-dynamic friction. The model accounts for effect of magma inertia on caldera block movement
2. sphr_MT.py: compute time dependent moment tensor associated with uniform pressurization of a spheroidal cavity embedded in elastic full space (Esehlby, 1957)
3. cmp_Er: calculate the radiated energy from time dependent tri-axial expansion source single force in homogeneous, elastic, full space

## Wrapper for ground motion prediction 
1. Butterworth.py
2. bodywaves_functions.py: analytical full space body waves (Aki & Richards, 2002)
3. surfacewaves_functions.py: analytical Rayleigh waves (Aki & Richards, 2002)
4. helpers.py
5. source_setup.py
6. synthetics.py
7. load_gfs.py
8. testing.py: check that the static limit of dynamic ground motion is correct

## Inversion interface with EMCEE hammer
1. inversion.py: function to run when setup Bayesian MCMC inversion
2. load_all_gfs.py: load pre-computed Green's function and make them globally available
3. load_data.py: load static displacements, dynamic velocity waveforms, and uncertainties
4. log_prob.py: compute log likelihood, prior
5. objective.py: objective function to optimize
6. pred.py: use output from the dynamic model to compute ground motions

## Others
1. mk_plots.py: make plots for publication
2. plot_MCMC_analytics.py: analyze inversion results
3. scale_waveforms.ipynb: check waveform sensitivity to model parameters

## References

Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J.(2013). emcee: The mcmc hammer. Publications of the Astronomical Society of the Pacific, 360125(925), 306.

Eshelby, J. D. (1957). The determination of the elastic field of an ellipsoidal inclusion, and related problems. Proceedings of the royal society of London. Series A. Mathematical and physical sciences, 241(1226), 376–396.

Aki, K., & Richards, P. G. (2002). Quantitative seismology. In (p. 76-77). Sausalito, California: University Science Books.

Zhu, L., & Rivera, L. A.(2002). A note on the dynamic and static displacements from a point source in multilayered media. Geophysical Journal International,405148(3), 619–627.




