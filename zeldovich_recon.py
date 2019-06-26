#
from __future__ import print_function,division

import numpy as np
from mcfit import SphericalBessel as sph
#mcfit multiplies by sqrt(2/pi)*x**2 to the function. 
#Divide the funciton by this to get the correct form 

from scipy.integrate import quad, simps
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.misc import derivative
import sys

class Zeldovich:
    '''
    Class to evaluate (real space) power spectra in Zeldovich reconstuction..
    
    ``Inspired'' by Chirag's code.
    
    '''
    def __init__(self, k, p, toler=1e-6):
        '''k,p are the linear theory power spectra in compatible units,
        e.g. h/Mpc and (Mpc/h)^3.
        Note that p contains should contain all the linear theory power spectra for matter, diplaced and shifted fields,
        s.t. p[0,:] = p_mm, p[1,:] = p_dm, p[2,:] = p_sm; p[3,:] = p_dd; p[4,:] = p_ds; p[5,:] = p_ss,
        where T stands for \theta_{bc}, normalized at redshifts z = 0
        '''
        
        self.kp    = k
        self.ps    = p
        
        # note: the spectra involving one power of the shift field, are actually for negative this field
        # this is because the shift field is the MINUS smoothed Zeldovich displacement
        self.pmm = p[:,0] ; pmax = np.max(self.pmm)
        self.pdm = p[:,1]
        self.psm = p[:,2]
        self.pdd = p[:,3]
        self.pds = p[:,4]
        self.pss = p[:,5]
        
        
        self.ilpk_mm    = self.loginterp(k,self.pmm)
        self.ilpk_dm    = self.loginterp(k[self.pdm>(toler*pmax)],self.pdm[self.pdm>(toler*pmax)])
        self.ilpk_sm    = self.loginterp(k[self.psm>(toler*pmax)],self.psm[self.psm>(toler*pmax)])
        self.ilpk_dd    = self.loginterp(k[self.pdd>(toler*pmax)],self.pdd[self.pdd>(toler*pmax)])
        self.ilpk_ds    = self.loginterp(k[self.pds>(toler*pmax)],self.pds[self.pds>(toler*pmax)])
        self.ilpk_ss    = self.loginterp(k[self.pss>(toler*pmax)],self.pss[self.pss>(toler*pmax)])
        
        self.ilpk = {'mm': self.ilpk_mm, 'dm':self.ilpk_dm, 'sm':self.ilpk_sm, 'dd':self.ilpk_dd, 'ds':self.ilpk_ds, 'ss':self.ilpk_ss}
        
        print("here!")
        self.renorm = np.sqrt(np.pi/2.) #mcfit normaliztion
        self.tpi2  = 2*np.pi**2.
        self.kint = np.logspace(-5, 5, 2e4)
        self.jn    = 10 #number of bessels to sum over
        
        self.pktable    = None
        self.pktable_dd = None
        self.pktable_ds = None
        self.pktable_ss = None
        self.num_power_components = 6 + 4 # four shear terms
        
        self.setup()
    
        #
    def setup(self):
        '''
        Create X_L, Y_L, xi_L, U1_L \& 0lag sigma.
        These will all be dictionaries labelled by 'species', except for the X, Y terms,
        which we'll have to deal with separately...
        
        '''
        self.qv, xi0v = self.xi0lin()
        
        pairs = ['mm','dm','sm','dd','ds','ss']
        self.corlins = {}
        self.Ulins   = {}
        self.Xlins   = {}
        self.Ylins   = {}
        self.XYlins = {}
        self.sigmas = {}
        self.yqs = {}
        
        for pair in pairs:
            
            # the sign of the spectra of the shift field are hereby fixed....
            if pair == 'sm' or pair == 'ds':
                s = -1
            else:
                s = +1
            
            q_p = -0.5
            
            print('Calculating 2-pt functions for species pair ' + pair,'tilt = ' + str(q_p))
            xi0lag = self.xi0lin0(species=pair)
            xi0v   = s*self.xi0lin(species=pair,tilt=-q_p)[1] # used to be 1 + q_p which was bad
            xi2v   = s*self.xi2lin(species=pair,tilt=-q_p)[1]

            self.Xlins[pair] = 2/3.*(xi0lag - xi0v - xi2v)

            
            # Check Ylin for zeros
            ylinv = 2*xi2v
            mask = (ylinv == 0)
            ylinv[mask] = interpolate(self.qv[~mask], ylinv[~mask])(self.qv[mask])
            self.Ylins[pair] = ylinv
            self.yqs[pair] = (1*self.Ylins[pair]/self.qv)
            
            self.XYlins[pair] = self.Xlins[pair] + self.Ylins[pair]
            self.sigmas[pair] = self.XYlins[pair][-1]
        
            self.corlins[pair] = s*self.corr(species=pair,tilt=1.5)[1] # used to be 3 + q_p
            self.Ulins[pair]   = s*self.u10lin(species=pair,tilt=1.5)[1] # used to be 2 + qp
        

        self.XYlin_mm = self.Xlins['mm'] + self.Ylins['mm']; self.sigma_mm = self.XYlin_mm[-1]
        self.XYlin_dd = self.Xlins['dd'] + self.Ylins['dd']; self.sigma_dd = self.XYlin_dd[-1]
        self.XYlin_ds = self.Xlins['ds'] + self.Ylins['ds']; self.sigma_ds = self.XYlin_ds[-1]
        self.XYlin_ss = self.Xlins['ss'] + self.Ylins['ss']; self.sigma_ss = self.XYlin_ss[-1]

        # calculate shear correlators
        self.zetas = {}
        self.chis  = {}
        self.Xs2s  = {}
        self.Ys2s  = {}
        self.Vs    = {}
            
        xi2lin_mm = self.xi2lin(species='mm',k_power=2)[1]
            
        for pair in ['mm','dm','sm']:
            if pair != 'mm':
                tilt = 0
            else:
                tilt = 1.5
            
            if pair == 'sm' or pair == 'ds':
                s = -1
            else:
                s = +1
                
            xi0lin = s*self.xi0lin(species=pair,k_power=2,tilt=1.5)[1]
            xi2lin = s*self.xi2lin(species=pair,k_power=2,tilt=0.5)[1]
            xi4lin = s*self.xi4lin(species=pair,k_power=2,tilt=0.5)[1]
                
            xi1lin = s*self.xi1lin(species=pair,k_power=1,tilt=0.5)[1]
            xi3lin = s*self.xi3lin(species=pair,k_power=1,tilt=0.5)[1]
                
            J2 = 2.*xi1lin/15 - 0.2*xi3lin
            J3 = -0.2*xi1lin - 0.2*xi3lin
            J4 = xi3lin

            self.zetas[pair] = 2*(4*xi0lin**2/45. + 8*xi2lin**2/63. + 8*xi4lin**2/35)
            self.chis[pair]  = 4*xi2lin**2/3.
            self.Vs[pair] = 4 * J2 * xi2lin_mm
            self.Xs2s[pair] = 4 * J3**2
            self.Ys2s[pair] = 6*J2**2 + 8*J2*J3 + 4*J2*J4 + 4*J3**2 + 8*J3*J4 + 2*J4**2


    ### Interpolate functions in log-sapce beyond the limits
    def loginterp(self, x, y, yint = None, side = "both",\
                  lorder = 15, rorder = 15, lp = 1, rp = -1, \
                  ldx = 1e-6, rdx = 1e-6):
        '''
        Extrapolate function by evaluating a log-index of left & right side
        '''
        if yint is None:
            yint = interpolate(x, y, k = 5)
        if side == "both":
            side = "lr"
            l =lp
            r =rp
        lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
        rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]
        print('Log index on left & right edges are = ', lneff, rneff)
        #
        xl = np.logspace(-18, np.log10(x[l]), 10**6.)
        xr = np.logspace(np.log10(x[r]), 10., 10**6.)
        yl = y[l]*(xl/x[l])**lneff
        yr = y[r]*(xr/x[r])**rneff
        #
        xint = x[l+1:r].copy()
        yint = y[l+1:r].copy()
        if side.find("l") > -1:
            xint = np.concatenate((xl, xint))
            yint = np.concatenate((yl, yint))
        if side.find("r") > -1:
            xint = np.concatenate((xint, xr))
            yint = np.concatenate((yint, yr))
        yint2 = interpolate(xint, yint, k = 5)
        #
        return yint2

    def dosph(self, n, x, f, tilt = 1.5, extrap = True):
        #Function to do bessel integral using FFTLog for kernels
        f = f*self.renorm
        return sph(x, nu = n, q = tilt)(f, extrap = extrap)
    
    #PT kernels below
    
    #0 lag
    def xi0lin0(self, species='mm', kmin = 1e-6, kmax = 1e3, k_power=0):
        # Note: the 0 lag piece here is the ONLY place the arithmetic mean instead
        #       of the geometric mean is used to evaluate a cross term.
        #       the k_power option allows you to calculate derivatives for higher derivatives of Psi
        
        X = species[0]; Y = species[1]
        integrand = lambda k: 0.5*(self.ilpk[X+X](k)+self.ilpk[Y+Y](k))
        
        val = simps(integrand(self.kp) * (self.kp>kmin) * (self.kp<kmax), self.kp ) / self.tpi2

        return val
    #j0
    def xi0lin(self, species='mm', kint = None, tilt = 0,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt)
    def xi1lin(self, species='mm', kint = None, tilt = 0,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt)
    #j2
    def xi2lin(self, species='mm', kint = None, tilt = 0,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(2, kint, integrand, tilt = tilt)
    def xi3lin(self, species='mm', kint = None, tilt = 0,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(3, kint, integrand, tilt = tilt)
    def xi4lin(self, species='mm', kint = None, tilt = 0,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(4, kint, integrand, tilt = tilt)
    
    #u1
    def u10lin(self, species='mm', kint = None,  tilt = 0,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = -1*kint*self.ilpk[species](kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt)

    #correlation function
    def corr(self, species='mm', kint = None, tilt = 0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint)
        integrand /= (1.*self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt)

    #################
    #Bessel Integrals for \mu
    def template(self, k, l, func, expon, suppress, power=1, za = False, expon_za = 1.,tilt=None, species='mm'):
        '''
        Generic template that is followed by mu integrals
        j0 is different since its exponent has sigma subtracted that is
        later used to suppress integral
        '''
        
        Fq = np.zeros_like(self.qv)
        
        if za == True and l == 0:
            Fq = expon_za * func * (self.yq)**l
        else:
            Fq = expon * func * (self.yq)**l
        
        if tilt is not None:
            q = max(0,tilt-l)
        else:
            q = max(0,1.5-l)
        
        # note that the angular integral for even powers of mu gives J_(l+1)
        ktemp, ftemp = sph(self.qv, nu= l+(power%2), q=q)(Fq*self.renorm,extrap = False)
        ftemp *= suppress

        return 1* k**l * np.interp(k, ktemp, ftemp)

    def setup_yq(self, D= 1, species='mm'):
        '''
            Use the right species in the Bessel expansion for Y(q)
            '''
        self.yq = self.Ylins[species] * D**2 / self.qv
    
    
    
        
    def p_integrals(self, k, D = 1,jn=None):
        '''
            Power spectra contributions for a single k.
            '''
        
        pair = 'mm'
        
        if jn == None:
            jn = self.jn
        
        ksq = k**2
        
        D2 = D**2; D4 = D2**2
        
        expon = np.exp(-0.5*ksq * D2 * (self.XYlin_mm - self.sigma_mm))
        exponm1 = np.expm1(-0.5*ksq * D2 *  (self.XYlin_mm - self.sigma_mm))
        suppress = np.exp(-0.5*ksq * D2 * self.sigma_mm)
        
        Ylin = self.Ylins['mm'] * D2
        
        Ulin = self.Ulins['mm'] * D2
        corlin = self.corlins['mm'] * D2
        V = self.Vs['mm'] * D4
        chi = self.chis['mm'] * D4
        zeta = self.zetas['mm'] * D4
        Xs2 = self.Xs2s['mm'] * D4
        Ys2 = self.Ys2s['mm'] * D4
        
        za, b1, b1sq, b2, b2sq, b1b2, bs, b1bs, b2bs, bssq = (0,)*self.num_power_components
        
        #l indep functions
        fza = 1.
        fb1 = -2 * k * Ulin
        fb1b2 = -2 * k * Ulin * corlin
        fb2sq = 0.5 * corlin**2
        fb1bs = - 2 * k * V
        fb2bs = chi
        fbssq  = zeta
        
        for l in range(jn):
            mu2fac = 1. - 2.*l/ksq/Ylin
            fb1sq = corlin - ksq * mu2fac * Ulin**2
            fb2 = - ksq * mu2fac * Ulin**2
            fbs = - ksq * (Xs2 + mu2fac * Ys2)
            
            #do integrals
            za += self.template(k,l,fza,expon,suppress,power=0,za=True,expon_za=exponm1,species=pair )
            b1 += self.template(k,l,fb1,expon,suppress,power=1,species=pair )
            b1sq += self.template(k,l,fb1sq,expon,suppress,power=0,species=pair )
            b2 += self.template(k,l,fb2,expon,suppress,power=0,species=pair )
            b2sq += self.template(k,l,fb2sq,expon,suppress,power=0,species=pair )
            b1b2 += self.template(k,l,fb1b2,expon,suppress,power=1,species=pair )
        
            bs += self.template(k,l,fbs,expon,suppress,power=0,species=pair )
            b1bs += self.template(k,l,fb1bs,expon,suppress,power=1,species=pair)
            b2bs += self.template(k,l,fb2bs,expon,suppress,power=0,species=pair )
            bssq += self.template(k,l,fbssq,expon,suppress,power=0,species=pair )
        
        return 4*np.pi*np.array([za,b1,b1sq,b2,b2sq,b1b2,bs,b1bs,b2bs,bssq])


    def make_ptable(self, D = 1, kmin = 1e-3, kmax = 2, nk = 200, jlfunc=None):
        '''Make a table of different terms of P(k) between a given
            'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
            This is the most time consuming part of the code.
            '''
        if jlfunc is None:
            jlfunc = lambda k: self.jn
        
        self.setup_yq(D=D,species='mm')
        
        self.pktable = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable[:, 0] = kv[:]
        print("Hankel transforms now off to the presses!")
        for foo in range(nk):
            print(foo)
            self.pktable[foo, 1:] = self.p_integrals(kv[foo],jn=jlfunc(kv[foo]),D=D)

        return self.pktable

        
    
    def pdd_integrals(self, k, D = 1, jn=None):
        '''Same as p_integrals but for the dd autospectrum.
            '''
        
        if jn == None:
            jn = self.jn
        
        pair = 'dd'
        
        D2 = D**2; D4 = D2**2
        
        ksq = k**2
        expon = np.exp(-0.5*ksq * D2 * (self.XYlin_dd - self.sigma_dd))
        exponm1 = np.expm1(-0.5*ksq * D2 * (self.XYlin_dd - self.sigma_dd))
        suppress = np.exp(-0.5*ksq*D2*self.sigma_dd)
        
        Ylin = self.Ylins['dd'] * D2
        
        Ulin = self.Ulins['dm'] * D2
        corlin = self.corlins['mm'] * D2
        V = self.Vs['dm'] * D4
        chi = self.chis['mm'] * D4
        zeta = self.zetas['mm'] * D4
        Xs2 = self.Xs2s['dm'] * D4
        Ys2 = self.Ys2s['dm'] * D4
        

        za, b1, b1sq, b2, b2sq, b1b2, bs, b1bs, b2bs, bssq = (0,)*self.num_power_components
        
        #l indep functions
        fza = 1.
        fb1 = -2 * k * Ulin
        fb1b2 = -2 * k * Ulin * corlin
        fb2sq = 0.5 * corlin**2
        fb1bs = - 2 * k * V
        fb2bs = chi
        fbssq  = zeta
        
        for l in range(jn):
            mu2fac = 1. - 2.*l/ksq/Ylin
            fb1sq = corlin - ksq * mu2fac * Ulin**2
            fb2 = - ksq * mu2fac * Ulin**2
            fbs = - ksq * (Xs2 + mu2fac * Ys2)

            #do integrals
            za += self.template(k,l,fza,expon,suppress,power=0,za=True,expon_za=exponm1,species=pair)
            b1 += self.template(k,l,fb1,expon,suppress,power=1,species=pair)
            b1sq += self.template(k,l,fb1sq,expon,suppress,power=0,species=pair)
            b2 += self.template(k,l,fb2,expon,suppress,power=0,species=pair)
            b2sq += self.template(k,l,fb2sq,expon,suppress,power=0,species=pair)
            b1b2 += self.template(k,l,fb1b2,expon,suppress,power=1,species=pair)
        
            bs += self.template(k,l,fbs,expon,suppress,power=0,species=pair)
            b1bs += self.template(k,l,fb1bs,expon,suppress,power=1,species=pair)
            b2bs += self.template(k,l,fb2bs,expon,suppress,power=0,species=pair)
            bssq += self.template(k,l,fbssq,expon,suppress,power=0,species=pair)
        
        return 4*np.pi*np.array([za,b1,b1sq,b2,b2sq,b1b2,bs,b1bs,b2bs,bssq])
    

    def make_pddtable(self, D = 1, kmin = 1e-3, kmax = 2, nk = 200, jlfunc=None):
        '''Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        
        if jlfunc is None:
            jlfunc = lambda k: self.jn
        
        self.setup_yq(species='dd',D=D)
        
        self.pktable_dd = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_dd[:, 0] = kv[:]
        print("Hankel transforms now off to the presses!")
        for foo in range(nk):
            print(foo)
            self.pktable_dd[foo, 1:] = self.pdd_integrals(kv[foo],jn=jlfunc(kv[foo]),D=D)

        return self.pktable_dd

    def pds_integrals(self, k, D = 1, jn = None):
        '''Same but for the ds cross spectrum. Roughly half as many terms.
            
            '''
        if jn == None:
            jn = self.jn
        
        D2 = D**2; D4 = D2**2
        
        pair = 'ds'
        ksq = k**2
        expon = np.exp(-0.5*ksq * D2 * (self.XYlin_ds - self.sigma_ds))
        exponm1 = np.expm1(-0.5*ksq * D2* (self.XYlin_ds - self.sigma_ds))
        suppress = np.exp(-0.5*ksq * D2 * self.sigma_ds)
        Ylin = self.Ylins[pair] * D2
        
        za, b1, b1sq, b2, b2sq, b1b2, bs, b1bs, b2bs, bssq = (0,)*self.num_power_components
        
        #l indep functions
        fza = 1.
        fb1 = -k * self.Ulins['sm'] * D2
        
        for l in range(jn):
            mu2fac = 1. - 2.*l/ksq/self.Ylins[pair]/D2
            fb2 = - 0.5 * ksq * mu2fac * self.Ulins['sm']**2 * D4
            fbs = - 0.5 * ksq * (self.Xs2s['sm'] + mu2fac * self.Ys2s['sm']) * D4
            
            #do integrals
            za += self.template(k,l,fza, expon,suppress,power=0,za=True,expon_za=exponm1,species=pair )
            b1 += self.template(k,l,fb1,expon,suppress,power=1,species=pair )
            b2 += self.template(k,l,fb2,expon,suppress,power=0,species=pair )
            bs += self.template(k,l,fbs,expon,suppress,power=0,species=pair )
        
        return 4*np.pi*np.array([za,b1,b1sq,b2,b2sq,b1b2,bs,b1bs,b2bs,bssq])
    
    
    def make_pdstable(self, D = 1, kmin = 1e-3, kmax = 2, nk = 200, jlfunc = None):
        '''Make a table of different terms of P(k) between a given
            'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
            This is the most time consuming part of the code.
            '''
        
        if jlfunc == None:
            jlfunc = lambda k: self.jn
        
        self.setup_yq(species='ds',D=D)
        
        self.pktable_ds = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_ds[:, 0] = kv[:]
        print("Hankel transforms now off to the presses!")
        for foo in range(nk):
            print(foo)
            self.pktable_ds[foo, 1:] = self.pds_integrals(kv[foo],jn=jlfunc(kv[foo]),D=D)

        return self.pktable_ds



    def pss_integrals(self, k, D = 1, jn = None):
        '''
            Only one term in the ss autospectrum!
                        '''
        pair = 'ss'
        
        if jn == None:
            jn = self.jn
        
        ksq = k**2
        D2 = D**2
        
        expon = np.exp(-0.5*ksq * D2 * (self.XYlin_ss - self.sigma_ss))
        exponm1 = np.expm1(-0.5*ksq * D2 * (self.XYlin_ss - self.sigma_ss))
        suppress = np.exp(-0.5*ksq * D2 * self.sigma_ss)
        
        Ylin = self.Ylins[pair] * D2
    
        za, b1, b1sq, b2, b2sq, b1b2, bs, b1bs, b2bs, bssq = (0,)*self.num_power_components
        
        #l indep functions
        fza = 1.

        for l in range(jn):
            #do integrals
            za += self.template(k,l,fza, expon,suppress,power=0,za=True,expon_za=exponm1,species=pair )

        return 4*np.pi*np.array([za,b1,b1sq,b2,b2sq,b1b2,bs, b1bs, b2bs, bssq])
    
    
    def make_psstable(self, D=1,kmin = 1e-3, kmax = 2, nk = 200, jlfunc=None):
        '''Make a table of different terms of P(k) between a given
            'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
            This is the most time consuming part of the code.
            '''
        if jlfunc is None:
            jlfunc = lambda k: self.jn
        
        self.setup_yq(species='ss',D=D)
        
        self.pktable_ss = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_ss[:, 0] = kv[:]
        print("Hankel transforms now off to the presses!")
        for foo in range(nk):
            print(foo)
            self.pktable_ss[foo, 1:] = self.pss_integrals(kv[foo],jn=jlfunc(kv[foo]),D=D)

        return self.pktable_ss


if __name__ == '__main__':
    print(0)
