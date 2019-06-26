import numpy as np
from mcfit import SphericalBessel as sph
#mcfit multiplies by sqrt(2/pi)*x**2 to the function.
#Divide the funciton by this to get the correct form

from matplotlib import pyplot as plt

from scipy.integrate import quad, simps
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.misc import derivative
from scipy.special import hyp2f1, gamma, hyperu
import sys

class Zeldovich_Recon_RSD:
    '''
        Class to evaluate the reconstructed Zeldovich power spectrum in redshift space.
        
        '''
    def __init__(self, k, p, toler=1e-6):
        '''
            k,p are the linear theory power spectra in compatible units,
            e.g. h/Mpc and (Mpc/h)^3.
            '''
        
        self.kp    = k
        self.ps    = p
        
        self.pmm = p[:,0]; pmax = np.max(self.pmm)
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
        self.sqrtpi = np.sqrt(np.pi)
        self.kint = np.logspace(-5, 5, 2e4)
        self.jn    = 6 #number of bessels to sum over
        
        self.pktable    = None
        self.pktable_dd = None
        self.pktable_ds = None
        self.pktable_ss = None
        self.num_power_components = 10
        
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

            q_p = -0.5
            
            # the sign of the spectra of the shift field are hereby fixed....
            if pair == 'sm' or pair == 'ds':
                s = -1
            else:
                s = +1
            
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

    
    
    def setup_method_ii(self):
        '''
            Correlators without zero lag for Rec-Iso.
            '''
        self.XYlin_ds0lag = self.XYlins['ds'] - self.XYlins['ds'][-1]
        self.sigma_ds0lag = self.XYlin_ds0lag[-1]
    
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
    
    def dosph(self, n, x, f, tilt = 1.5):
        #Function to do bessel integral using FFTLog for kernels
        f = f*self.renorm
        return sph(x, nu = n, q = tilt)(f, extrap = True)
    
    #PT kernels below
    
    
    #0 lag
    def xi0lin0(self, species='mm', kmin = 1e-6, kmax = 1e3, k_power=0):
        # Note: the 0 lag piece here is the ONLY place the arithmetic mean instead
        #       of the geometric mean is used to evaluate a cross term.
        # the k_power option allows you to calculate derivatives for higher derivatives of Psi
        X = species[0]; Y = species[1]
        integrand = lambda k: 0.5*(self.ilpk[X+X](k)+self.ilpk[Y+Y](k))
        val = simps(integrand(self.kp) * (self.kp>kmin) * (self.kp<kmax), self.kp ) / self.tpi2
        
        return val
    #j0
    def xi0lin(self, species='mm', kint = None, tilt = 1.5,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt)
    def xi1lin(self, species='mm', kint = None, tilt = 1.5,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt)
    #j2
    def xi2lin(self, species='mm', kint = None, tilt = 1.5,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(2, kint, integrand, tilt = tilt)
    def xi3lin(self, species='mm', kint = None, tilt = 1.5,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(3, kint, integrand, tilt = tilt)
    def xi4lin(self, species='mm', kint = None, tilt = 1.5,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(4, kint, integrand, tilt = tilt)
    
    
    #u1
    def u10lin(self, species='mm', kint = None,  tilt = 1.5,damp_fac=False,k_power=0):
        if kint is None:
            kint = self.kint
        integrand = -1*kint*self.ilpk[species](kint) * kint**k_power
        if damp_fac:
            integrand *= (np.exp(-kint*0.5)+1e-10)
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt)
    
    
    #correlation function
    def corr(self, species='mm', kint = None, tilt = 1.5, damp_fac=False):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk[species](kint)
        if damp_fac:
            integrand *= (np.exp(-kint*0.5) + 1e-10) # not sure why I put this here
        integrand /= (1.*self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt)
    
    
    #### Define RSD Kernels #######
    
    def setup_rsd_facs(self,f,nu,pair='mm',D=1):
        self.f = f
        self.nu = nu
        self.D = D
        self.Kfac = np.sqrt(1+f*(2+f)*nu**2); self.Kfac2 = self.Kfac**2
        self.s = f*nu*np.sqrt(1-nu**2)/self.Kfac
        self.c = np.sqrt(1-self.s**2); self.c2 = self.c**2; self.ic2 = 1/self.c2; self.c3 = self.c**3
        self.Bfac = -0.5 * self.Kfac2 * self.Ylins[pair] * self.D**2 # this times k is "B"
        self.kaiser = 1 + f*nu**2
        self.Hlpower = -0.5 * f * nu * np.sqrt(1-f*nu**2) * self.Ylins['ds'] * self.D**2
        self.Hlfac = - f*nu*np.sqrt(1-nu**2)/self.kaiser
        self.Bfac_ds = -0.5 * self.kaiser * self.Ylins['ds'] * self.D**2
    
    
    def _G0_l_n(self,n,m,k):
        return gamma(m+n+0.5)/gamma(m+1)/gamma(n+0.5)/gamma(1-m+n) * hyp2f1(0.5-n,-n,0.5-m-n,self.ic2)
    
    def _dG0dA_l_n(self,n,m,k):
        x = self.ic2
        fnm = gamma(m+n+0.5)/gamma(m+1)/gamma(n+0.5)/gamma(1-m+n)
        ret = fnm*(2*n/self.c-2*n*self.c)*hyp2f1(0.5-n,-n,0.5-m-n,x)/(k*self.qv)
        if n > 0:
            ret += fnm*n*(n-0.5)/(0.5-m-n)*(2/self.c-2/self.c3)*hyp2f1(1.5-n,1-n,1.5-m-n,x)/(k*self.qv)
        return ret
    
    def _d2G0dA2_l_n(self,n,m,k):
        x = self.ic2
        fnm = gamma(m+n+0.5)/gamma(m+1)/gamma(n+0.5)/gamma(1-m+n)
        ret = fnm /(k*self.qv)**2 * (1-1./x) * ( (2*m-1-4*n*(m+1))*hyp2f1(0.5-n,-n,0.5-m-n,x) \
                                                +(1-4*n**2+m*(4*n-2))*hyp2f1(1.5-n,-n,0.5-m-n,x) )
        return ret
    
    
    def _G0_l(self,l,k,nmax=10):
        powerfac = self.Bfac * k**2/self.ic2
        g0l = 0
        
        for ii in range(nmax):
            n = l+ii
            g0l += powerfac**n * self._G0_l_n(n,l,k)
        
        return g0l
    
    def _dG0dA_l(self,l,k,nmax=10):
        powerfac = self.Bfac * k**2/self.ic2
        dg0l = 0
        
        for ii in range(nmax):
            n = l+ii
            dg0l += powerfac**n * self._dG0dA_l_n(n,l,k)
        
        return dg0l
    
    
    def _d2G0dA2_l(self,l,k,nmax=10):
        powerfac = self.Bfac * k**2/self.ic2
        dg0l = 0
        
        for ii in range(nmax):
            n = l+ii
            dg0l += powerfac**n * self._d2G0dA2_l_n(n,l,k)
        
        return dg0l

    
    def _H0_l_np(self,N,M,A):
        return (-1)**(N-M) * A**(2*M-N) * gamma(M+0.5)/gamma(2*M+1)/gamma(2*M-N+1)/gamma(N-M+1)/self.sqrtpi
    
    def _H0_l(self,N,k):
        # This can probably be sped up using an interpolation routine on the parameter A
        # ... but I'll leave this for later
        powerfac = k**2 * self.Hlpower
        ret = 0
        for ii in np.arange(np.ceil(0.5*N),N+1):
            ret += self._H0_l_np(N,ii,powerfac)
        return ret
    
    def _K_ds_n(self,n,k,lmax=10,power=0):
        ksq = k**2
        ret = 0
        
        if power == 0 or power == 1:
            for ll in range(lmax):
                ret += self.Hlfac**ll * self._H0_l(ll,k) * hyperu(-ll,n-ll+1,-ksq*self.Bfac_ds)
        elif power == 2:
            B = ksq * self.Bfac_ds
            for ll in range(lmax):
                ret += self.Hlfac**ll * self._H0_l(ll,k) * (hyperu(-ll,n-ll+1,-B) \
                                                       + n/B*hyperu(-ll,n-ll,-B))
    
        return ret

    
    
    #################
    #Bessel Integrals for \mu
    def template_MII(self, k, l, func, expon, suppress, power=1, za = False, expon_za = 1.,tilt=None):
        ''' Generic template that is followed by mu integrals using method MII.
            j0 is different since its exponent has sigma subtracted that is
            later used to suppress integral
            '''
        
        Fq = np.zeros_like(self.qv)
        
        if za == True and l == 0:
            #Fq = expon_za * func * (-2./k/self.qv)**l
            Fq = expon * func * (-2./k/self.qv)**l - 1
        else:
            Fq = expon * func * (-2./k/self.qv)**l
        
        if tilt is not None:
            q = tilt
        else:
            q = 1.5-l
        
        # note that the angular integral for even powers of mu gives J_(l+1)
        ktemp, ftemp = sph(self.qv, nu= l+(power%2), q=q)(Fq*self.renorm,extrap = False)
        ftemp *= suppress
        
        return np.interp(k, ktemp, ftemp)

    def template_MI(self, k, l, func, expon, suppress, power=0, za = False, expon_za = 1.,tilt=None, pair='mm'):
        ''' Generic template that is followed by mu integrals using method MI.
            j0 is different since its exponent has sigma subtracted that is
            later used to suppress integral
            '''
        
        D2 = self.D**2
        Fq = np.zeros_like(self.qv)
        
        if za == True and l == 0:
            #Fq = expon_za * func * (-2./k/self.qv)**l
            Fq = expon * func - 1
        else:
            Fq = expon * func * (D2*self.yqs[pair])**l
        
        if tilt is not None:
            q = tilt
        else:
            q = max(0,1.5-l)
        
        # note that the angular integral for even powers of mu gives J_(l+1)
        ktemp, ftemp = sph(self.qv, nu= l+power, q=q)(Fq*self.renorm,extrap = False)
        
        ftemp *= suppress
        
        
        return 1* k**l * np.interp(k, ktemp, ftemp)
    
    
    def p_integrals(self, k, nmax=6, jn=None):
        '''Do integrals for a single k (and nu, set by setup_rsd_facs) for unreconstructed power spectrum.
            '''
        
        if jn is None:
            jn = self.jn
        
        K = k*self.Kfac; Ksq = K**2
        D2 = self.D**2; D4 = D2**2
        corlin = self.corlins['mm'] * D2
        Ulin = self.Ulins['mm'] * D2
        Xs2 = self.Xs2s['mm'] * D4; Ys2 = self.Ys2s['mm'] * D4
        V = self.Vs['mm'] * D4
        chi = self.chis['mm'] * D4
        zeta = self.zetas['mm'] * D4
        
        
        expon = np.exp(-0.5*Ksq * D2* (self.XYlin_mm - self.sigma_mm))
        exponm1 = np.expm1(-0.5*Ksq * D2* (self.XYlin_mm - self.sigma_mm))
        suppress = np.exp(-0.5*Ksq * D2* self.sigma_mm)
        
        A = k*self.qv*self.c
        d2Gs = [self._d2G0dA2_l(ii,k,nmax=nmax) for ii in range(jn)]
        dGs = [self._dG0dA_l(ii,k,nmax=nmax) for ii in range(jn)] + [0]
        G0s = [self._G0_l(ii,k,nmax=nmax)    for ii in range(jn)] + [0] + [0]
        
        G1s = [-(dGs[ii] + 0.5*A*G0s[ii-1])   for ii in range(jn)]
        G2s = [-(d2Gs[ii] + A * dGs[ii-1] + 0.5*G0s[ii-1] + 0.25 * A**2 *G0s[ii-2]) for ii in range(jn)]
        
        za, b1, b1sq, b2, b2sq, b1b2, bs, b1bs, b2bs, bssq = (0,)*self.num_power_components
        
        for l in range(jn):
            #l-dep functions
            G0 = G0s[l]; G1 = G1s[l]; G2 = G2s[l]
            fza = 1 * G0
            fb1 = -2 * K * Ulin * G1
            fb1sq = corlin * G0 - Ksq*G2*Ulin**2
            fb2 = - Ksq * G2 * Ulin**2
            fb1b2 = - 2 * K * G1 * Ulin * corlin
            fb2sq = 0.5 * G0 * corlin**2
            fbs = - Ksq * (Xs2 * G0 + Ys2 * G2)
            fb1bs = - 2 * K * V * G1
            fb2bs = chi * G0
            fbssq = zeta * G0
            
            #do integrals
            za += self.template_MII(k,l,fza,expon,suppress,power=0,za=True,expon_za=exponm1)
            b1 += self.template_MII(k,l,fb1,expon,suppress,power=0)
            b1sq += self.template_MII(k,l,fb1sq,expon,suppress,power=0)
            b2 += self.template_MII(k,l,fb2,expon,suppress,power=0)
            b2sq += self.template_MII(k,l,fb2sq,expon,suppress,power=0)
            b1b2 += self.template_MII(k,l,fb1b2,expon,suppress,power=0)
            bs += self.template_MII(k,l,fbs,expon,suppress,power=0)
            b1bs += self.template_MII(k,l,fb1bs,expon,suppress,power=0)
            b2bs += self.template_MII(k,l,fb2bs,expon,suppress,power=0)
            bssq += self.template_MII(k,l,fbssq,expon,suppress,power=0)
        
        return 4*np.pi*np.array([za,b1,b1sq,b2,b2sq,b1b2,bs,b1bs,b2bs,bssq])
    
    
    def make_ptable(self, f, nu, kmin = 1e-2, kmax = 0.5, nmax=8, nk = 30, D=1, jlfunc = None):
        '''Make a table of different terms of P(k) at a given angle nu between a given
            'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
            This is the most time consuming part of the code.
            '''
        
        if jlfunc is None:
            jlfunc = lambda x: self.jn
        
        self.setup_rsd_facs(f,nu,D=D)
        self.pktable = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable[:, 0] = kv[:]
        
        for foo in range(nk):
            print(foo)
            self.pktable[foo, 1:] = self.p_integrals(kv[foo],nmax=nmax,jn=jlfunc(kv[foo]))
        
        return self.pktable

    def p_integrals_dd(self, k, nmax = 6, jn = None):
        '''All the terms in the dd autospectrum, otheriwse like p_integrals.
            
            '''
        
        if jn is None:
            jn = self.jn
        
        K = k*self.Kfac; Ksq = K**2; D2 = self.D**2; D4 = D2**2
        Ulin = self.Ulins['dm'] * D2
        corlin = self.corlins['mm'] * D2
        Xs2 = self.Xs2s['dm'] * D4; Ys2 = self.Ys2s['dm'] * D4
        V = self.Vs['dm'] * D4
        chi = self.chis['mm'] * D4
        zeta = self.zetas['mm'] * D4
        
        expon = np.exp(-0.5*Ksq * D2* (self.XYlin_dd - self.sigma_dd))
        exponm1 = np.expm1(-0.5*Ksq * D2* (self.XYlin_dd - self.sigma_dd))
        suppress = np.exp(-0.5*Ksq * D2* self.sigma_dd)
        
        
        A = k*self.qv*self.c
        d2Gs = [self._d2G0dA2_l(ii,k,nmax=nmax) for ii in range(jn)]
        dGs = [self._dG0dA_l(ii,k,nmax=nmax) for ii in range(jn)] + [0]
        G0s = [self._G0_l(ii,k,nmax=nmax)    for ii in range(jn)] + [0] + [0]
        
        G1s = [-(dGs[ii] + 0.5*A*G0s[ii-1])   for ii in range(jn)]
        G2s = [-(d2Gs[ii] + A * dGs[ii-1] + 0.5*G0s[ii-1] + 0.25 * A**2 *G0s[ii-2]) for ii in range(jn)]
        
        za, b1, b1sq, b2, b2sq, b1b2, bs, b1bs, b2bs, bssq = (0,)*self.num_power_components
        
        for l in range(jn):
            #l-dep functions
            G0 = G0s[l]; G1 = G1s[l]; G2 = G2s[l]
            fza = 1 * G0
            fb1 = - 2 * K * Ulin * G1
            fb1sq = corlin * G0 - Ksq*G2*Ulin**2
            fb2 = -Ksq * G2 * Ulin**2
            fb1b2 = -2 * K *  G1 * Ulin * corlin
            fb2sq = 0.5 * G0 * corlin**2
            fbs = - Ksq * (Xs2 * G0 + Ys2 * G2)
            fb1bs = - 2 * K * V * G1
            fb2bs = chi * G0
            fbssq = zeta * G0
            
            #do integrals
            za += self.template_MII(k,l,fza,expon,suppress,power=0,za=True,expon_za=exponm1)
            b1 += self.template_MII(k,l,fb1,expon,suppress,power=0)
            b1sq += self.template_MII(k,l,fb1sq,expon,suppress,power=0)
            b2 += self.template_MII(k,l,fb2,expon,suppress,power=0)
            b2sq += self.template_MII(k,l,fb2sq,expon,suppress,power=0)
            b1b2 += self.template_MII(k,l,fb1b2,expon,suppress,power=0)
            bs += self.template_MII(k,l,fbs,expon,suppress,power=0)
            b1bs += self.template_MII(k,l,fb1bs,expon,suppress,power=0)
            b2bs += self.template_MII(k,l,fb2bs,expon,suppress,power=0)
            bssq += self.template_MII(k,l,fbssq,expon,suppress,power=0)
        
        return 4*np.pi*np.array([za,b1,b1sq,b2,b2sq,b1b2,bs,b1bs,b2bs,bssq,za,0,0])
    
    
    def make_pddtable(self, f, nu, kmin = 1e-2, kmax = 0.5, nk = 30, D=1, nmax=8, jlfunc=None):
        ''' Like make_ptable.
            '''
        
        if jlfunc is None:
            jlfunc = lambda x: self.jn
        
        self.setup_rsd_facs(f,nu,pair='dd',D=D)
        self.pktable_dd = np.zeros([nk, self.num_power_components+1+3]) # one column for ks + three for counterterms
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_dd[:, 0] = kv[:]
        
        for foo in range(nk):
            print(foo)
            self.pktable_dd[foo, 1:] = self.p_integrals_dd(kv[foo],nmax=nmax,jn=jlfunc(kv[foo]))
        
        return self.pktable_dd

    def p_integrals_ds_RecSym(self, k, nmax=6, jn = None):
        '''All the terms in the ds cross spectrum, otheriwse like p_integrals, for Rec-Sym
            
            '''
        
        if jn is None:
            jn = self.jn
        
        K = k*self.Kfac; Ksq = K**2; D2 = self.D**2; D4 = D2**2
        Ulin = self.Ulins['sm'] * D2
        Xs2 = self.Xs2s['sm'] * D4; Ys2 = self.Ys2s['sm'] * D4
        
        expon = np.exp(-0.5*Ksq * D2* (self.XYlin_ds - self.sigma_ds))
        exponm1 = np.expm1(-0.5*Ksq * D2* (self.XYlin_ds - self.sigma_ds))
        suppress = np.exp(-0.5*Ksq * D2* self.sigma_ds)
        
        A = k*self.qv*self.c
        d2Gs = [self._d2G0dA2_l(ii,k,nmax=nmax) for ii in range(jn)]
        dGs = [self._dG0dA_l(ii,k,nmax=nmax) for ii in range(jn)] + [0]
        G0s = [self._G0_l(ii,k,nmax=nmax)    for ii in range(jn)] + [0] + [0]
        
        G1s = [-(dGs[ii] + 0.5*A*G0s[ii-1])   for ii in range(jn)]
        G2s = [-(d2Gs[ii] + A * dGs[ii-1] + 0.5*G0s[ii-1] + 0.25 * A**2 *G0s[ii-2]) for ii in range(jn)]

        
        za, b1, b1sq, b2, b2sq, b1b2, bs, b1bs, b2bs, bssq = (0,)*self.num_power_components
        
        for l in range(jn):
            #l-dep functions
            fza = 1 * G0s[l]
            fb1 = - K * Ulin * G1s[l]
            fb2 = - 0.5 * Ksq * Ulin**2 * G2s[l]
            fbs = - 0.5 * Ksq * (Xs2 * G0s[l] + Ys2 * G2s[l])
            
            #do integrals
            za += self.template_MII(k,l,fza,expon,suppress,power=0,za=True,expon_za=exponm1)
            b1 += self.template_MII(k,l,fb1,expon,suppress,power=0)
            b2 += self.template_MII(k,l,fb2,expon,suppress,power=0)
            bs += self.template_MII(k,l,fbs,expon,suppress,power=0)
        
        return 4*np.pi*np.array([za,b1,b1sq,b2,b2sq,b1b2,bs,b1bs,b2bs,bssq,0,za,0])
    
    def p_integrals_ds_RecIso(self, k, nmax=6, jn = None):
        '''Since the templates function is modified this currently contains only a subset of terms.
            
            '''
        
        if jn is None:
            jn = self.jn
        
        ksq = k**2; D2 = self.D**2; D4 = D2**2
        Ulin = self.Ulins['sm'] * D2
        Xs2 = self.Xs2s['sm'] * D4; Ys2 = self.Ys2s['sm'] * D4
        
        expon = np.exp(-0.5*ksq * self.kaiser * D2 * (self.XYlin_ds0lag - self.sigma_ds0lag))
        exponm1 = np.expm1(-0.5*ksq * self.kaiser * D2 * (self.XYlin_ds0lag - self.sigma_ds0lag))
        suppress = np.exp(-0.5*ksq * self.kaiser * D2 * self.sigma_ds0lag)
        damp_fac = np.exp(-0.25 * ksq * D2* (self.Kfac2 * self.sigma_dd + self.sigma_ss))
        
        K0s = [ self._K_ds_n(l,k,lmax=nmax) for l in range(jn)  ]
        K2s = [ self._K_ds_n(l,k,lmax=nmax,power=2) for l in range(jn)  ]
        
        za, b1, b1sq, b2, b2sq, b1b2, bs, b1bs, b2bs, bssq = (0,)*self.num_power_components
        
        for l in range(jn):
            kaiser_l = self.kaiser**l
            #l-dep functions
            fza = 1 * K0s[l]
            fb1 = -k * Ulin * K0s[l]
            fb2 = -0.5 * ksq * Ulin**2 * K2s[l]
            fbs = - 0.5 * ksq * (Xs2 * K0s[l] + Ys2 * K2s[l])
            
            
            #do integrals
            za += kaiser_l * self.template_MI(k,l,fza,expon,suppress,power=0,za=True,expon_za=exponm1,pair='ds',)
            b1 += kaiser_l * self.template_MI(k,l,fb1,expon,suppress,power=1,pair='ds')
            b2 += kaiser_l * self.template_MI(k,l,fb2,expon,suppress,power=0,pair='ds')
            bs += kaiser_l * self.template_MI(k,l,fbs,expon,suppress,power=0,pair='ds')
    
        return 4*np.pi*damp_fac*np.array([za,b1,b1sq,b2,b2sq,b1b2,bs,b1bs,b2bs,bssq,0,za,0])
    
    def make_pdstable(self, f, nu, kmin = 1e-2, kmax = 0.5, nk = 30,D=1,nmax=8,jlfunc=None, method = 'RecSym'):
        '''Make a table of different terms of P(k) between a given
            'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
            This is the most time consuming part of the code.
            '''
        
        if jlfunc is None:
            jlfunc = lambda x: self.jn
        
        self.setup_rsd_facs(f,nu,pair='ds',D=D)
        self.pktable_ds = np.zeros([nk, self.num_power_components+1+3]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_ds[:, 0] = kv[:]
        
        if method == 'RecSym':
            for foo in range(nk):
                print(foo)
                self.pktable_ds[foo, 1:] = self.p_integrals_ds_RecSym(kv[foo],nmax=nmax,jn=jlfunc(kv[foo]))
        elif method == 'RecIso':
            self.setup_method_ii()
            
            for foo in range(nk):
                print(foo)
                self.pktable_ds[foo, 1:] = self.p_integrals_ds_RecIso(kv[foo],nmax=nmax,jn=jlfunc(kv[foo]))

        return self.pktable_ds

    def p_integrals_ss(self, k, nmax=6,jn = None):
        '''Since the templates function is modified this currently contains only a subset of terms.
            
            '''
        if jn is None:
            jn = self.jn
        
        K = k*self.Kfac; Ksq = K**2; D2 = self.D**2
        
        expon = np.exp(-0.5*Ksq * D2* (self.XYlin_ss - self.sigma_ss))
        exponm1 = np.expm1(-0.5*Ksq * D2* (self.XYlin_ss - self.sigma_ss))
        suppress = np.exp(-0.5*Ksq * D2* self.sigma_ss)
        
        
        G0s = [ self._G0_l(l,k,nmax=nmax) for l in range(jn)  ]
        
        za, b1, b1sq, b2, b2sq, b1b2, bs, b1bs, b2bs, bssq = (0,)*self.num_power_components
        
        for l in range(jn):
            #l-dep functions
            fza = 1 * G0s[l]
            
            #do integrals
            za += self.template_MII(k,l,fza,expon,suppress,power=0,za=True,expon_za=exponm1)
        
        return 4*np.pi*np.array([za,b1,b1sq,b2,b2sq,b1b2,bs,b1bs,b2bs,bssq,0,0,za])
    
    
    def make_psstable(self, f, nu, kmin = 1e-2, kmax = 0.5, nk = 30,D=1, nmax=8, jlfunc = None):
        '''Make a table of different terms of P(k) between a given
            'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
            This is the most time consuming part of the code.
            '''
        
        if jlfunc is None:
            jlfunc = lambda x: self.jn
        
        self.setup_rsd_facs(f,nu,pair='ss',D=D)
        self.pktable_ss = np.zeros([nk, self.num_power_components+1+3]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_ss[:, 0] = kv[:]
        
        for foo in range(nk):
            print(foo)
            self.pktable_ss[foo, 1:] = self.p_integrals_ss(kv[foo],nmax=nmax,jn=jlfunc(kv[foo]))
        
        return self.pktable_ss

    def make_pltable(self,f, D=1,ngauss = 2, kmin = 1e-3, kmax = 0.5, nk = 30, nmax=8, recon = True, method = 'RecIso', jlfunc=None, nmaxfunc = None, a_perp = 1, a_par = 1):
        ''' Make a table of the monopole and quadrupole in k space.
            Using gauss legendre integration.
            With a_perp and a_par, this gives the observed (and not ``true'') multipoles.'''

        if jlfunc is None:
            jlfunc = lambda k: self.jn
        
        if nmaxfunc is None:
            nmaxfunc = lambda k: nmax
        
        if method == 'RecIso':
            self.setup_method_ii()
        
        # since we are always symmetric in nu, can ignore negative values
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        
        pknutable = np.zeros((len(nus),nk,self.num_power_components+3))
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        
        for ii, nu in enumerate(nus_calc):
            fac = np.sqrt(1 + nu**2 * ((a_perp/a_par)**2-1))
            k_apfac = fac / a_perp
            nu_obs = nu * a_perp/a_par/fac
            
            for jj, k in enumerate(kv):
                print(ii,jj)
                k_obs = k_apfac * k
                nsum = nmaxfunc(k_obs)
                if recon and method == 'RecSym':
                    self.setup_rsd_facs(f,nu_obs,pair='dd',D=D)
                    pknutable[ii,jj,:] = self.p_integrals_dd(k_obs,nmax=nsum,jn=jlfunc(k))
                    self.setup_rsd_facs(f,nu_obs,pair='ss',D=D)
                    pknutable[ii,jj,:] += self.p_integrals_ss(k_obs,nmax=nsum,jn=jlfunc(k))
                    self.setup_rsd_facs(f,nu_obs,pair='ds',D=D)
                    pknutable[ii,jj,:] -= 2*self.p_integrals_ds_RecSym(k_obs,nmax=nsum,jn=jlfunc(k))
            
                elif recon and method == 'RecIso':
                    self.setup_rsd_facs(f,nu_obs,pair='dd',D=D)
                    pknutable[ii,jj,:] = self.p_integrals_dd(k_obs,nmax=nsum,jn=jlfunc(k))
                    self.setup_rsd_facs(f,nu_obs,pair='ds',D=D)
                    pknutable[ii,jj,:] -= 2*self.p_integrals_ds_RecIso(k_obs,nmax=nsum,jn=jlfunc(k))
                    self.setup_rsd_facs(0,nu_obs,pair='ss',D=D)
                    pknutable[ii,jj,:] += self.p_integrals_ss(k_obs,nmax=nsum,jn=jlfunc(k))
                    
                else:
                    self.setup_rsd_facs(f,nu_obs,pair='mm',D=D)
                    pknutable[ii,jj,:-3] = self.p_integrals(k_obs,nmax=nsum,jn=jlfunc(k))
    
        pknutable[ngauss:,:,:] = np.flip(pknutable[0:ngauss],axis=0)

        self.kv = kv
        
        if recon:
            self.p0ktable_recon = 0.5 * np.sum((ws*L0)[:,None,None]*pknutable,axis=0)
            self.p2ktable_recon = 2.5 * np.sum((ws*L2)[:,None,None]*pknutable,axis=0)
        
            return kv, self.p0ktable_recon, self.p2ktable_recon
        
        else:
            self.p0ktable = 0.5 * np.sum((ws*L0)[:,None,None]*pknutable,axis=0)
            self.p2ktable = 2.5 * np.sum((ws*L2)[:,None,None]*pknutable,axis=0)

            return kv, self.p0ktable, self.p2ktable
        




if __name__ == '__main__':
    
    print(0)

