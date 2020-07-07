Slated for public release. XXX = arxiv link.

A code to calculate power spectra in real and redshift space in the Zeldovich approximation, pre- and post-reconstruction. Based on the real-space Zeldovich code

 https://github.com/martinjameswhite/CLEFT_GSM/tree/master/HaloZeldovich. 

These codes require SciPy and NumPy, as well as the mcfit library:

https://github.com/eelregit/mcfit

to perform Hankel transforms via FFTLog, which you will need to install. To instead compute the equivalent configuration-space numbers see

https://github.com/martinjameswhite/ZeldovichRecon

which we have checked agrees with our results to sub-percent levels except close to zeros of the correlation function.


The most general class is "zeldovich_rsd_recon.py" which computes the redshift space pre- and post-reconstruction power spectra for two reconstruction prescriptions, termed "Rec-Sym" and "Rec-Iso" in XXX. Two special cases--the pre-reconstruction redshift space power spectrum and the real-space post-reconstruction spectrum, can be calculated using "zeldovich_rsd.py" and "zeldovich_recon.py", respectively.

In addition, the smoothing legngth \Sigma only to damp the wiggle component of the power spectrum, both pre- and post-reconstruction, as first derived in

https://arxiv.org/abs/1509.02120

can be obtained by querying the quantities self.Xlins[species] and self.Ylins[species] for species = 'mm', 'dd', 'ds', 'ss' etc. See XXX for further details.

Finally, these codes make no assumption about the particular smoothing kernel used for reconstruction. As such it takes as an input the linear (auto and cross) power spectra between the fields "m" (for matter, pre-reconstruction), "d" (for the displaced galaxies" and "s" (for the shifted random particles).

Some example calls are given in the IPython notebook attached.
