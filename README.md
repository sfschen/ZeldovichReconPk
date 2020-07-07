A code to calculate power spectra in real and redshift space in the Zeldovich approximation, pre- and post-reconstruction. Based on the real-space Zeldovich code

https://github.com/sfschen/velocileptors

These codes require SciPy and NumPy, as well pyFFTW to perform Hankel transforms via FFTLog, which you will need to install. To instead compute the equivalen configuration-space numbers see

https://github.com/martinjameswhite/ZeldovichRecon

which we have checked agrees with our results to sub-percent levels except close to zeros of the correlation function. Based on formulas in arxiv:1907.00043.


The most general class is "zeldovich_rsd_recon_fftw.py" which computes the redshift space pre- and post-reconstruction power spectra for two reconstruction prescriptions, termed "Rec-Sym" and "Rec-Iso" in 1907.00043. A special case--the pre-reconstruction redshift space power spectrum and the real-space post-reconstruction spectrum, can be calculated using "zeldovich_recon.py".

In addition, the smoothing legngth \Sigma only to damp the wiggle component of the power spectrum, both pre- and post-reconstruction, as first derived in

https://arxiv.org/abs/1509.02120

can be obtained by querying the quantities self.Xlins[species] and self.Ylins[species] for species = 'mm', 'dd', 'ds', 'ss' etc. See the paper for further details.

Some example calls are given in the IPython notebook attached.

An older version of this code, which used the FFTLog library mcfit, is in "deprecated" directory.