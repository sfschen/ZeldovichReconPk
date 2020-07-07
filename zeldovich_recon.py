import numpy as np
from loginterp import loginterp
import time

from scipy.interpolate import interp1d

from spherical_bessel_transform_fftw import SphericalBesselTransform
from qfuncfft_recon import QFuncFFT

class Zeldovich_Recon:
    '''
    Class to evaluate Zeldovich power spectra post-reconstruction.
    
    Based on the soon-to-be-available velocilptors code.
    
    '''

    def __init__(self, k, p, R = 15., cutoff=20, jn=15, N = 4000, threads=1, extrap_min = -6, extrap_max = 3, shear = True, import_wisdom=False, wisdom_file='./zelda_wisdom.npy'):
    
        self.shear = shear
        
        self.N = N
        self.extrap_max = extrap_max
        self.extrap_min = extrap_min
        
        
        # set up integration/FFTlog grid
        self.cutoff = cutoff
        self.kint = np.logspace(extrap_min,extrap_max,self.N)
        self.pint = loginterp(k,p)(self.kint) * np.exp(-(self.kint/self.cutoff)**2)
        self.qint = np.logspace(-extrap_max,-extrap_min,self.N)
        
        # self up linear correlation between fields:
        Sk = np.exp(-0.5 * (R*self.kint)**2)
        Skm1 = - np.expm1(-0.5 * (R*self.kint)**2)
        
        self.plins = {}
        self.plins['mm'] = self.pint
        self.plins['dd'] = self.pint * Skm1**2
        self.plins['ds'] = -self.pint * Skm1 * Sk
        self.plins['ss'] = self.pint * Sk**2
        self.plins['dm'] = self.pint * Skm1
        self.plins['sm'] = -self.pint * Sk
        
        # ... and calculate Lagrangian correlators
        self.setup_2pts()
        
        # setup hankel transforms for power spectra
        if self.shear:
            self.num_power_components = 10
            self.num_power_components_ds = 4
        else:
            self.num_power_componens = 6
            self.num_power_components_ds = 3
            
            
        self.jn = jn
        self.threads = threads
        self.import_wisdom = import_wisdom
        self.wisdom_file = wisdom_file
        self.sph = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.num_power_components, threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        self.sphx = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.num_power_components_ds, threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        self.sphs = SphericalBesselTransform(self.qint, L=self.jn, ncol=1, threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        
        # indices for the cross
        self.ds_inds = [1,2,4,7]
        
        
    def setup_2pts(self):
        
        # define the various power spectra and compute xi_l_n
        
        species = ['s','d','m']
        self.qfs = {}
        
        # matter
        self.qfs['mm'] = QFuncFFT(self.kint, self.plins['mm'], pair_type='matter',\
                                                    qv=self.qint,  shear=self.shear)
        # dd, ds, ss
        self.qfs['dd'] = QFuncFFT(self.kint, self.plins['dd'], pair_type='disp x disp',\
                                                    qv=self.qint,  shear=self.shear)
        self.qfs['ds'] = QFuncFFT(self.kint, self.plins['ds'], pair_type='disp x disp',\
                                                    qv=self.qint,  shear=self.shear)
        self.qfs['ss'] = QFuncFFT(self.kint, self.plins['ss'], pair_type='disp x disp',\
                                                    qv=self.qint,  shear=self.shear)
            
        # dm, sm
        self.qfs['dm'] = QFuncFFT(self.kint, self.plins['dm'], pair_type='disp x bias',\
                                                    qv=self.qint,  shear=self.shear)
        self.qfs['sm'] = QFuncFFT(self.kint, self.plins['sm'], pair_type='disp x bias',\
                                                    qv=self.qint,  shear=self.shear)

        # Now piece together the various X, Y, U, ...
        # First for the pure displacements
        pairs = ['mm','dd','ds','ss']
        
        self.Xlins = {}
        self.Ylins = {}
        self.XYlins = {}
        self.yqs = {}
        self.sigmas = {}
        
        for pair in pairs:
            a, b = pair[0], pair[1]
            self.Xlins[pair] = 2./3 * ( 0.5*self.qfs[a+a].xi0m2[0] + 0.5*self.qfs[b+b].xi0m2[0] \
                                            - self.qfs[pair].xi0m2 - self.qfs[pair].xi2m2 )
            self.Ylins[pair] = 2 * self.qfs[pair].xi2m2
            
            self.XYlins[pair] = self.Xlins[pair] + self.Ylins[pair]
            self.sigmas[pair] = self.Xlins[pair][-1]
            self.yqs[pair] = (1*self.Ylins[pair]/self.qint)
            
            
        # Now for the bias x displacmment terms
        pairs = ['mm', 'dm', 'sm']
    
        self.Ulins = {}
        self.Vs = {} # dm, sm
        self.Xs2s = {} # dm, sm
        self.Ys2s = {} # dm, sm
        
        for pair in pairs:
            a, b = pair[0], pair[1]
            self.Ulins[pair] = - self.qfs[pair].xi1m1
            
            if self.shear:
                J2 = 2.*self.qfs[pair].xi1m1/15 - 0.2*self.qfs[pair].xi3m1
                J3 = -0.2*self.qfs[pair].xi1m1 - 0.2*self.qfs[pair].xi3m1
                J4 = self.qfs[pair].xi3m1
                
                self.Vs[pair] = 4 * J2 * self.qfs['mm'].xi20
                self.Xs2s[pair] = 4 * J3**2
                self.Ys2s[pair] = 6*J2**2 + 8*J2*J3 + 4*J2*J4 + 4*J3**2 + 8*J3*J4 + 2*J4**2
                
        # ... and finally the pure bias terms, i.e. matter
        self.corlins = {'mm':self.qfs['mm'].xi00}
        if self.shear:
            self.zetas = {'mm': 2*(4*self.qfs['mm'].xi00**2/45. + 8*self.qfs['mm'].xi20**2/63. + 8*self.qfs['mm'].xi40**2/35)}
            self.chis = {'mm':4*self.qfs['mm'].xi20**2/3.}
                
                
                
                
    def p_integrals(self, k):
        '''
        Only a small subset of terms included for now for testing.
        '''
        ksq = k**2; kcu = k**3
        
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*ksq *self.sigma)
        
        ret = np.zeros(self.num_power_components)
        
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
        
        if self.shear:
            zero_lags = np.array([1,0,0,0,0,0,-ksq*self.sigmas2,0,0,0])
        else:
            zero_lags = np.array([1,0,0,0,0,0])
        
        for l in range(self.jn):
            # l-dep functions
            shiftfac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            mu3fac = 1. - 2.*(l-1)/ksq/self.Ylin # mu3 terms start at j1 so l -> l-1
            
            bias_integrands[0,:] = 1. # za
            bias_integrands[1,:] = -2 * k * self.Ulin * shiftfac  # b1
            bias_integrands[2,:] = self.corlin - ksq*mu2fac*self.Ulin**2# b1sq
            bias_integrands[3,:] = - ksq * mu2fac * self.Ulin**2 # b2
            bias_integrands[4,:] = (-2 * k * self.Ulin * self.corlin) * shiftfac # b1b2
            bias_integrands[5,:] = 0.5 * self.corlin**2 # b2sq
            
            if self.shear:
                bias_integrands[6,:] = -ksq * (self.Xs2 + mu2fac*self.Ys2)# bs
                bias_integrands[7,:] = -2*k*self.V*shiftfac # b1bs
                bias_integrands[8,:] = self.chi # b2bs
                bias_integrands[9,:] = self.zeta # bssq


            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= zero_lags[:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)
    
    
        #ret += ret[0] * zero_lags
        
        return 4*suppress*np.pi*ret

    def make_ptable(self, kmin = 1e-3, kmax = 3, nk = 100):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        pair = 'mm'
        self.Xlin = self.Xlins[pair]
        self.Ylin = self.Ylins[pair]
        self.sigma = self.sigmas[pair]
        self.yq = self.yqs[pair]
        self.XYlin = self.XYlins[pair]
        
        self.Ulin = self.Ulins[pair]
        self.corlin = self.corlins[pair]
        
        if self.shear:
            self.V = self.Vs[pair]
            self.Xs2 = self.Xs2s[pair]; self.sigmas2 = self.Xs2[-1]
            self.Ys2 = self.Ys2s[pair]
            self.chi = self.chis[pair]
            self.zeta = self.zetas[pair]
            
        
        self.pktable = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable[:, 0] = kv[:]
        for foo in range(nk):
            self.pktable[foo, 1:] = self.p_integrals(kv[foo])
            
            
    def make_pddtable(self, kmin = 1e-3, kmax = 3, nk = 100):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        pair = 'dd'
        self.Xlin = self.Xlins[pair]
        self.Ylin = self.Ylins[pair]
        self.sigma = self.sigmas[pair]
        self.yq = self.yqs[pair]
        self.XYlin = self.XYlins[pair]
        
        self.Ulin = self.Ulins['dm']
        self.corlin = self.corlins['mm']
        
        if self.shear:
            self.V = self.Vs['dm']
            self.Xs2 = self.Xs2s['dm']; self.sigmas2 = self.Xs2[-1]
            self.Ys2 = self.Ys2s['dm']
            self.chi = self.chis['mm']
            self.zeta = self.zetas['mm']
            
        
        self.pktable_dd = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_dd[:, 0] = kv[:]
        for foo in range(nk):
            self.pktable_dd[foo, 1:] = self.p_integrals(kv[foo])
            
        
        
    def pds_integrals(self, k):
        '''
        Only a small subset of terms included for now for testing.
        '''
        ksq = k**2; kcu = k**3
        
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*ksq *self.sigma)
        
        ret = np.zeros(self.num_power_components_ds)
        
        bias_integrands = np.zeros( (self.num_power_components_ds,self.N)  )
        
        if self.shear:
            zero_lags = np.array([1,0,0,-ksq*self.sigmas2])
        else:
            zero_lags = np.array([1,0,0])
        
        for l in range(self.jn):
            # l-dep functions
            shiftfac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            mu3fac = 1. - 2.*(l-1)/ksq/self.Ylin # mu3 terms start at j1 so l -> l-1
            
            bias_integrands[0,:] = 1. # za
            bias_integrands[1,:] = - k * self.Ulin * shiftfac  # b1
            bias_integrands[2,:] = - 0.5 * ksq * mu2fac * self.Ulin**2 # b2

            
            if self.shear:
                bias_integrands[3,:] = - 0.5 * ksq * (self.Xs2 + mu2fac*self.Ys2)# bs

            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= zero_lags[:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sphx.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)
    
        
        return 4*suppress*np.pi*ret

    def make_pdstable(self, kmin = 1e-3, kmax = 3, nk = 100):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        pair = 'ds'
        self.Xlin = self.Xlins[pair]
        self.Ylin = self.Ylins[pair]
        self.sigma = self.sigmas[pair]
        self.yq = self.yqs[pair]
        self.XYlin = self.XYlins[pair]
        
        self.Ulin = self.Ulins['sm']
        
        if self.shear:
            self.Xs2 = self.Xs2s['sm']; self.sigmas2 = self.Xs2[-1]
            self.Ys2 = self.Ys2s['sm']

        self.pktable_ds = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_ds[:, 0] = kv[:]
        for foo in range(nk):
            self.pktable_ds[foo, self.ds_inds] = self.pds_integrals(kv[foo])
            
            
            
    def pss_integrals(self, k):
        '''
        Only a small subset of terms included for now for testing.
        '''
        ksq = k**2; kcu = k**3
        
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*ksq *self.sigma)
        
        ret = 0
        
        bias_integrands = np.zeros( (1,self.N)  )
        zero_lags = 1

        
        for l in range(self.jn):
            bias_integrands[0,:] = 1. # za

            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= zero_lags # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sphs.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)
            
        return 4*suppress*np.pi*ret

    def make_psstable(self, kmin = 1e-3, kmax = 3, nk = 100):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        pair = 'ss'
        self.Xlin = self.Xlins[pair]
        self.Ylin = self.Ylins[pair]
        self.sigma = self.sigmas[pair]
        self.yq = self.yqs[pair]
        self.XYlin = self.XYlins[pair]
        

        self.pktable_ss = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_ss[:, 0] = kv[:]
        for foo in range(nk):
            self.pktable_ss[foo, 1] = self.pss_integrals(kv[foo])
