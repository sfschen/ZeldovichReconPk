
import numpy as np
#import kernels
from mcfit import SphericalBessel as sph
#mcfit multiplies by sqrt(2/pi)*x**2 to the function.
#Divide the funciton by this to get the correct form

from matplotlib import pyplot as plt

from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.misc import derivative
from scipy.special import hyp2f1, gamma
import sys

class Zeldovich_RSD:
    '''
        Class to evaluate the Zeldovich power spectrum in redshift space.
        
        Uses "Method II" in the paper.
        
        '''
    def __init__(self, k, p):
        '''k,p are the linear theory power spectra in compatible units,
            e.g. h/Mpc and (Mpc/h)^3.
            f is the growth-factor derivative'''
        self.kp    = k
        self.p     = p
        self.ilpk  = self.loginterp(k, p)
        self.renorm=np.sqrt(np.pi/2.) #mcfit normaliztion
        self.tpi2  = 2*np.pi**2.
        self.kint  = np.logspace(-5, 5, 2e4)
        self.jn    = 10 #number of bessels to sum over
        
        # set up velocity table
        self.pktable_MII = None
        self.num_power_components = 10
        
        
        
        self.setup()
    #
    def setup(self):
        '''
            Create X_L, Y_L, xi_L, U1_L \& 0lag sigma.
            '''
        self.xi0lag = self.xi0lin0()
        self.qv, xi0v = self.xi0lin()
        xi2v = self.xi2lin(tilt=-0.5)[1] # this tilt gives better low q behavior
        self.corlin = self.corr()[1]
        self.Ulin = self.u10lin()[1]
        #
        self.Xlin = 2/3.*(self.xi0lag - xi0v - xi2v)
        ylinv = 2*xi2v
        #Since we divide by ylin, check for zeros
        mask = (ylinv == 0)
        ylinv[mask] = interpolate(self.qv[~mask], ylinv[~mask])(self.qv[mask])
        self.Ylin = ylinv
        self.XYlin = (self.Xlin + self.Ylin)
        self.sigma = self.XYlin[-1]
        self.yq = (1*self.Ylin/self.qv)
    
        # calculate shear terms
        xi0lin = self.xi0lin(k_power=2,tilt=1.5)[1]
        xi2lin = self.xi2lin(k_power=2,tilt=0.5)[1]
        xi4lin = self.xi4lin(k_power=2,tilt=0.5)[1]
    
        xi1lin = self.xi1lin(k_power=1,tilt=0.5)[1]
        xi3lin = self.xi3lin(k_power=1,tilt=0.5)[1]
    
        J2 = 2.*xi1lin/15 - 0.2*xi3lin
        J3 = -0.2*xi1lin - 0.2*xi3lin
        J4 = xi3lin

        self.zeta = 2*(4*xi0lin**2/45. + 8*xi2lin**2/63. + 8*xi4lin**2/35)
        self.chi  = 4*xi2lin**2/3.
        self.V = 4 * J2 * xi2lin
        self.Xs2 = 4 * J3**2
        self.Ys2 = 6*J2**2 + 8*J2*J3 + 4*J2*J4 + 4*J3**2 + 8*J3*J4 + 2*J4**2
    
    
    
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
    def xi0lin0(self, kmin = 1e-6, kmax = 1e3):
        val = quad(self.ilpk, kmin, kmax, limit = 200)[0]/self.tpi2
        return val
    #j0
    def xi0lin(self, kint = None, tilt = 1.5, k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt)
    def xi1lin(self, kint = None, tilt = 1.5, k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt)
    #j2
    def xi2lin(self, kint = None, tilt = 1.5, k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(2, kint, integrand, tilt = tilt)
    def xi3lin(self, kint = None, tilt = 1.5, k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(3, kint, integrand, tilt = tilt)
    def xi4lin(self, kint = None, tilt = 1.5, k_power=0):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint) * kint**k_power
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(4, kint, integrand, tilt = tilt)

    #u1
    def u10lin(self, kint = None,  tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = -1*kint*self.ilpk(kint)
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt)
    #correlatin function
    def corr(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (1.*self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt)
    
    
    
    #### Define RSD Kernels #######
    
    def setup_rsd_facs(self,f,nu,D=1):
        self.f = f
        self.nu = nu
        self.D = D
        self.Kfac = np.sqrt(1+f*(2+f)*nu**2); self.Kfac2 = self.Kfac**2
        self.s = f*nu*np.sqrt(1-nu**2)/self.Kfac
        self.c = np.sqrt(1-self.s**2); self.c2 = self.c**2; self.ic2 = 1/self.c2; self.c3 = self.c**3
        self.Bfac = -0.5 * self.Kfac2 * self.Ylin * self.D**2 # this times k is "B"
    
    
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
    
    
    #################
    #Bessel Integrals for \mu
    def template_MII(self, k, l, func, expon, suppress, power=1, za = False, expon_za = 1.,tilt=None):
        ''' Simplified vs the original code. Beta.
            Generic template that is followed by mu integrals
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
            q = max(0,1.5-l)
        
        # note that the angular integral for even powers of mu gives J_(l+1)
        ktemp, ftemp = sph(self.qv, nu= l+(power%2), q=q)(Fq*self.renorm,extrap = False)
        
        ftemp *= suppress
        
        
        return np.interp(k, ktemp, ftemp)


    def p_integrals_MII(self, k, nmax=8, jn=None):
        '''Since the templates function is modified this currently contains only a subset of terms.
        
        '''
        
        if jn == None:
            jn = self.jn
        
        K = k*self.Kfac; Ksq = K**2; D2 = self.D**2; D4 = D2**2
    
        expon = np.exp(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*Ksq * D2* self.sigma)
        
        
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
            fb1 = -2 * K * self.Ulin * G1s[l] * D2
            fb1sq = self.corlin * G0s[l]*D2 - Ksq*G2s[l]*self.Ulin**2*D4
            fb2 = - Ksq * G2s[l] * self.Ulin**2 * D4
            fb1b2 = - 2 * K * G1s[l] * self.Ulin * self.corlin * D4
            fb2sq = 0.5 * G0s[l] * self.corlin**2 * D4
            fbs = - Ksq * (self.Xs2 * G0s[l] + self.Ys2 * G2s[l]) * D4
            fb1bs = - 2 * K * self.V * G1s[l] * D4
            fb2bs = self.chi * G0s[l] * D4
            fbssq = self.zeta * G0s[l] * D4

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




    def make_ptable_MII(self, f, nu, kmin = 1e-2, kmax = 0.5, nk = 30, nmax=8, jlfunc=None):
        '''Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        
        if jlfunc is None:
            jlfunc = lambda k: self.jn
        
        self.setup_rsd_facs(f,nu)
        self.pktable_MII = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_MII[:, 0] = kv[:]

        for foo in range(nk):
            print(foo)
            self.pktable_MII[foo, 1:] = self.p_integrals_MII(kv[foo],nmax=nmax,jn=jlfunc(kv[foo]))

        return self.pktable_MII


    def make_pltable_MII(self,f, ngauss = 2, kmin = 1e-3, kmax = 0.5, nk = 30, nmax=8, jlfunc=None):
        ''' Make a table of the monopole and quadrupole in k space.
            Using gauss legendre integration.'''
        
        if jlfunc is None:
            jlfunc = lambda k: self.jn
        
        # since we are always symmetric in nu, can ignore negative values
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        
        self.pknutable = np.zeros((len(nus),nk,self.num_power_components))
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        
        for ii, nu in enumerate(nus_calc):
            self.setup_rsd_facs(f,nu)
            
            for jj, k in enumerate(kv):
                print(ii,jj)
                self.pknutable[ii,jj,:] = self.p_integrals_MII(k,nmax=nmax,jn=jlfunc(k))
        
        self.pknutable[ngauss:,:,:] = np.flip(self.pknutable[0:ngauss],axis=0)
        
        self.kv = kv
        self.p0ktable = 0.5 * np.sum((ws*L0)[:,None,None]*self.pknutable,axis=0)
        self.p2ktable = 2.5 * np.sum((ws*L2)[:,None,None]*self.pknutable,axis=0)
        
        return kv, self.p0ktable, self.p2ktable



if __name__ == '__main__':

    print(0)
