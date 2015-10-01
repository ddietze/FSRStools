""" 
.. module: FSRStools.rraman
   :platform: Windows
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu> 

Resonance Raman excitation profile calculation based on the time-domain picture of resonance Raman. See Myers and Mathies in *Biological Applications of Raman Spectroscopy*, Vol. 2, pp. 1-58 (John Wiley and Sons, New York, 1987) for details (referred to as Myers in the following). The code is mainly based on Myers' Fortran 77 code (see Appendix of PhD Thesis of K. M. Spillane, 2011, UC Berkeley for source code).

Here is a short example calculating Myers' *Gedankenmolecule* from Myers and Mathies::

    import numpy as np

    # parameters:
    # -----------
    # displacements
    D = np.array([1.27, 0.3, 0.7, 0.53])
    
    # ground state frequencies
    RMg = np.array([1550.0, 1300.0, 1150.0, 1000.0])
    
    # excited state frequencies
    RMe = np.array([1550.0, 1300.0, 1150.0, 1000.0])
    
    # electronic zero-zero energy
    E0 = 20700.0
    
    # homogeneous linewidth and shape parameter
    Gamma = 200.0
    halpha = 0
    
    # inhomogeneous linewidth and shape parameter
    sig = 400.0
    ialpha = 1
    
    # electronic transition dipole length
    M = 0.8
    
    # index of refraction of surrounding medium
    IOR = 1.0
    
    # time axis parameters for integrations
    tmax = 5000
    dt = 0.2
    
    # just calculate fundamentals
    nquanta = np.identity(len(RMg))
    sshift = np.dot(nquanta, RMg)
    
    # calculation part
    # ----------------
    # create axes
    t, wn = getAxes(tmax, dt)

    # zero-zero energy and damping
    # add here all time domain stuff
    TDpart = getZeroZeroEnergy(t, E0) * rr.getHomogeneousDamping(t, Gamma, halpha)

    # time dependent overlap integrals
    OVLPS = getOverlaps(t, D, RMg, RMe, nquanta)

    # calculate cross-sections
    sigmaA, sigmaR = getCrossSections(wn, dt, TDpart, OVLPS, sshift, M, IOR, sig, ialpha)

..
   This file is part of the FSRStools python module.

   The FSRStools python module is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The FSRStools python module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with the FSRStools python module. If not, see <http://www.gnu.org/licenses/>.

   Copyright 2014 Daniel Dietze <daniel.dietze@berkeley.edu>.   
""" 
import numpy as np
import pylab as pl
from scipy.interpolate import interp1d
import sys

# some constants
hbar = 5308.880986  #: Planck's constant over 2 pi, hbar, in `cm-1 fs`
c0 = 2.99792458e-5  #: speed of light in `cm / fs`
kB = 0.695          #: Boltzman's constant in `cm-1 / K`

# -------------------------------------------------------------------------------------------------------------------   
# some useful functions

def radperfs2wn(w):
    """Angular frequency (rad / fs) to wavenumber (cm-1).
    """
    return hbar * w

# 
def wn2radperfs(e):
    """Wavenumber (cm-1) to angular frequency (rad / fs).
    """
    return e / hbar

def getAxes(tmax, dt):
    """Create time and frequency axes for the resonance Raman calculations.
    
    :param float tmax: Endpoint for time domain calculation (fs). This value should be high enough to capture the full dephasing.
    :param float dt: Increment of time axis (fs). This value should be small enough to capture the highest vibronic feature in the excited state.
    :returns: Time axis (fs) and frequency axis (cm-1).
    """
    t = np.arange(0, tmax+dt, dt)
    numPoints = len(t)
    wn = np.arange(numPoints) / (c0 * dt * numPoints)
    return t, wn

def molarExtinction2AbsCS(eSpctr, IOR):
    """Convert molar extinction (cm-1 / M) to molecular absorption cross section (A**2 / molec).

    See McHale, Resonance Raman Spectroscopy, Wiley, (2002), p. 545 or Myers & Mathies for details. The absorption cross section in solution has to be scaled by index of refraction unless the molar extinction has not been corrected.
    
    :param array eSpctr: Extinction spectrum in (cm-1 / M).
    :param float IOR: Index of refraction of surrounding solvent / medium.
    :returns: Absorption spectrum in units of (A**2 / molec.), same shape as eSpcrt.
    """
    return 1e3 * np.log( 10.0 ) * eSpctr / 6.0221e23 * 1e8 * 1e8 / IOR

def diff2absRamanCS(diffRaCS, rho):
    """Convert the differential Raman cross section (A**2/molec sr) to absolute Raman cross section in (A**2 / molec) for a given depolarization ratio rho.

    :param float diffRaCS: Differential Raman cross section (A**2/molec sr).
    :param float rho: Associated depolarization ratio of this Raman mode.
    :returns: Absolute Raman cross section in (A**2 / molec).
    """
    return 8.0 * np.pi / 3.0 * (1.0 + 2.0 * rho) / (1.0 + rho) * diffRaCS
    
# ----------------------------------------------------------------------------------------------------------------------------------- 
# time dependent overlap integrals with equal ground and excited state vibrational frequencies
# the t00 overlap does not contain the factors exp(-1j wVIB t) nor exp(-1j E0/hbar t) as these are taken care of when assembling the cross section
# Myers eqs. (37) - (39)
# Delta = displacement in dimensionless coordinates
# eVIB = vibrational frequency (cm-1)
# t = time axis in fs
def t00A(t, Delta, eVIB):
    """Time dependent overlap integral between vibrational ground states of electronic ground and excited state with equal ground and excited state vibrational frequencies.
    
    :param array t: Time axis in (fs).
    :param float Delta: Displacement of excited state potential energy surface along this vibrational coordinate in dimensionless coordinates.
    :param float eVIB: Vibrational frequency (cm-1).
    :returns: 0-0 overlap integral as function of time (same shape as t).
    
    .. seealso:: Myers, Eqs. (37) - (39).
    """
    # The 0-0 overlap does not contain the factors :math:`e^{-j w_{VIB} t}` nor :math:`e^{-j E_0 / \\hbar t}` as these are taken care of when assembling the cross section.
    return np.exp(-Delta**2 / 2.0 * (1.0 - np.exp(-1j * eVIB / hbar * t)))

def t10A(t, Delta, eVIB):
    """Time dependent overlap integral between vibrational ground and first excited state of electronic ground and excited state with equal ground and excited state vibrational frequencies.
    
    :param array t: Time axis in (fs).
    :param float Delta: Displacement of excited state potential energy surface along this vibrational coordinate in dimensionless coordinates.
    :param float eVIB: Vibrational frequency (cm-1).
    :returns: 1-0 overlap integral as function of time (same shape as t).
    
    .. seealso:: Myers, Eqs. (37) - (39).
    """
    return Delta / np.sqrt(2) * (np.exp(-1j * eVIB / hbar * t) - 1.0) * t00A(t, Delta, eVIB)
    
def t20A(t, Delta, eVIB):
    """Time dependent overlap integral between vibrational ground and second excited state of electronic ground and excited state with equal ground and excited state vibrational frequencies.
    
    :param array t: Time axis in (fs).
    :param float Delta: Displacement of excited state potential energy surface along this vibrational coordinate in dimensionless coordinates.
    :param float eVIB: Vibrational frequency (cm-1).
    :returns: 2-0 overlap integral as function of time (same shape as t).
    
    .. seealso:: Myers, Eqs. (37) - (39).
    """
    return Delta**2 / (2 * np.sqrt(2)) * (np.exp(-1j * eVIB / hbar * t) - 1.0)**2 * t00A(t, Delta, eVIB)

# ------------------------------------------------------------------------------------------------------------------------------------------------- 
# same with different frequency in ground and excited state
# Myers eqs. (42) - (44)
# Delta = displacement in dimensionless coordinates
# eg = ground state vibrational frequency (cm-1)
# ee = excited state vibrational frequency (cm-1)
# t = time axis in fs
def t00B(t, Delta, eg, ee):
    """Time dependent overlap integral between vibrational ground states of electronic ground and excited state with different ground and excited state vibrational frequencies.
    
    :param array t: Time axis in (fs).
    :param float Delta: Displacement of excited state potential energy surface along this vibrational coordinate in dimensionless coordinates.
    :param float eg: Vibrational frequency in the ground state (cm-1).
    :param float ee: Vibrational frequency in the excited state (cm-1).
    :returns: 0-0 overlap integral as function of time (same shape as t).
    
    .. seealso:: Myers, Eqs. (42) - (44).
    """    
    wg = eg / hbar
    we = ee / hbar
    
    swe = np.sin(we * t)
    cwe = np.cos(we * t)
    
    pt = we / wg * Delta * swe
    qt = Delta * (1 - cwe)
    
    # the log reduces to 0.5 * eg / hbar * t when eg = ee
    # this is the factor that is taken out in the t00A case, as it cancels with the exp in the integral later on
    # however, np.log returns values such that -pi < arg(log(..)) < pi
    gt = 1j / 2.0 * np.log( 1j * wg / we * swe + cwe) + pt * (qt - Delta) / 2.0 # skip -E0 t / hbar
    # add the following term to recover t00A for eg = ee
    gt = gt - 1j / 2.0 * np.log( 1j * np.sin(wg * t) + np.cos(wg * t))
    
    at = -0.5 * 1j * (1j * cwe - (we / wg) * swe)/(1j * (wg / we) * swe + cwe)

    a = at + 0.5
    pp = pt - 2.0 * 1j * at * qt
    gp = 1j * at * qt**2 - pt * qt + gt
    
    return a**(-0.5) * np.exp(-pp**2/(4.0 * a)) * np.exp(1j * gp)
    
def t10B(t, Delta, eg, ee):
    """Time dependent overlap integral between vibrational ground and first excited state of electronic ground and excited state with different ground and excited state vibrational frequencies.
    
    :param array t: Time axis in (fs).
    :param float Delta: Displacement of excited state potential energy surface along this vibrational coordinate in dimensionless coordinates.
    :param float eg: Vibrational frequency in the ground state (cm-1).
    :param float ee: Vibrational frequency in the excited state (cm-1).
    :returns: 1-0 overlap integral as function of time (same shape as t).
    
    .. seealso:: Myers, Eqs. (42) - (44).
    """    
    wg = eg / hbar
    we = ee / hbar
    
    swe = np.sin(we * t)
    cwe = np.cos(we * t)
    
    pt = we / wg * Delta * swe
    qt = Delta * (1 - cwe)
    # the log reduces to 0.5 * eg / hbar * t when eg = ee
    # this is the factor that is taken out in the t00A case, as it cancels with the exp in the integral later on
    # however, np.log returns values such that -pi < arg(log(..)) < pi
    gt = 1j / 2.0 * np.log( 1j * wg / we * swe + cwe) + pt * (qt - Delta) / 2.0 # skip -E0 t / hbar
    # add the following term to recover t00A for eg = ee
    gt = gt - 1j / 2.0 * np.log( 1j * np.sin(wg * t) + np.cos(wg * t))
    
    
    at = -0.5 * 1j * (1j * cwe - (we / wg) * swe)/(1j * (wg / we) * swe + cwe)

    a = at + 0.5
    pp = pt - 2.0 * 1j * at * qt
    gp = 1j * at * qt**2 - pt * qt + gt
    return 2**(-0.5) * pp / (1j * a) * t00B(t, Delta, eg, ee)
    
def t20B(t, Delta, eg, ee):
    """Time dependent overlap integral between vibrational ground and second excited state of electronic ground and excited state with different ground and excited state vibrational frequencies.
    
    :param array t: Time axis in (fs).
    :param float Delta: Displacement of excited state potential energy surface along this vibrational coordinate in dimensionless coordinates.
    :param float eg: Vibrational frequency in the ground state (cm-1).
    :param float ee: Vibrational frequency in the excited state (cm-1).    
    :returns: 2-0 overlap integral as function of time (same shape as t).
    
    .. seealso:: Myers, Eqs. (42) - (44).
    """  
    wg = eg / hbar
    we = ee / hbar
    
    swe = np.sin(we * t)
    cwe = np.cos(we * t)
    
    pt = we / wg * Delta * swe
    qt = Delta * (1 - cwe)
    # the log reduces to 0.5 * eg / hbar * t when eg = ee
    # this is the factor that is taken out in the t00A case, as it cancels with the exp in the integral later on
    # however, np.log returns values such that -pi < arg(log(..)) < pi
    gt = 1j / 2.0 * np.log( 1j * wg / we * swe + cwe) + pt * (qt - Delta) / 2.0 # skip -E0 t / hbar
    # add the following term to recover t00A for eg = ee
    gt = gt - 1j / 2.0 * np.log( 1j * np.sin(wg * t) + np.cos(wg * t))
    
    at = -0.5 * 1j * (1j * cwe - (we / wg) * swe)/(1j * (wg / we) * swe + cwe)

    a = at + 0.5
    pp = pt - 2.0 * 1j * at * qt
    gp = 1j * at * qt**2 - pt * qt + gt
    return -8**(-0.5) * (pp**2 / a**2 + 2. * (1. - 1./a)) * t00B(t, Delta, eg, ee)  
    
# ----------------------------------------------------------------------------------------------------------------------------------
# same for linear dissociative excited state surfaces
# Myers eqs. (52) - (54)
# beta = slope of potential energy surface (dV / dq) in cm-1 (q is dimensionless coordinate)
# eVIB = vibrational frequency (cm-1)
def t00D(t, beta, eVIB):
    """Time dependent overlap integral between vibrational ground states of electronic ground and excited state with a linear dissociative excited state surface along this vibrational coordinate.
    
    :param array t: Time axis in (fs).
    :param float beta: Slope of excited state potential energy surface (dV / dq) in (cm-1) (q is dimensionless coordinate).
    :param float eVIB: Vibrational frequency (cm-1).
    :returns: 0-0 overlap integral as function of time (same shape as t).
    
    .. seealso:: Myers, Eqs. (52) - (54).
    """  
    return (1.0 + 1j * eVIB / hbar * t / 2.0)**(-0.5) * np.exp(-beta**2 * ( 6 * t**2 + 1j * eVIB / hbar * t**3 ) / (24 * hbar**2))
    
def t10D(t, beta, eVIB):
    """Time dependent overlap integral between vibrational ground and first excited state of electronic ground and excited state with a linear dissociative excited state surface along this vibrational coordinate.
        
    :param array t: Time axis in (fs).
    :param float beta: Slope of excited state potential energy surface (dV / dq) in (cm-1) (q is dimensionless coordinate).
    :param float eVIB: Vibrational frequency (cm-1).
    :returns: 1-0 overlap integral as function of time (same shape as t).
    
    .. seealso:: Myers, Eqs. (52) - (54).
    """  
    return -1j * 2**(-0.5) * (beta * t / hbar) * t00D(t, beta, eVIB)
    
def t20D(t, beta, eVIB):
    """Time dependent overlap integral between vibrational ground and second excited state of electronic ground and excited state with a linear dissociative excited state surface along this vibrational coordinate.
    
    :param array t: Time axis in (fs). 
    :param float beta: Slope of excited state potential energy surface (dV / dq) in (cm-1) (q is dimensionless coordinate).
    :param float eVIB: Vibrational frequency (cm-1).
    :returns: 2-0 overlap integral as function of time (same shape as t).
    
    .. seealso:: Myers, Eqs. (52) - (54).
    """  
    return -2**(-0.5) * (beta**2 * t**2 / (2.0 * hbar**2) - 1j * eVIB / hbar * t / (2.0 + 1j * eVIB / hbar * t)) * t00D(t, beta, eVIB)

# ---------------------------------------------------------------------------------------------------------------------------------
def getOverlaps(t, D, RMg, RMe, nquanta):
    """Calculate the time dependent overlap integrals / Franck-Condon factors :math:`<i|i(t)>_k` and :math:`<f|i(t)>_k`.
    
    :param array t: Time axis in (fs).
    :param array D: Array of N normalized displacements of excited state surfaces (deltas), or slope of linear dissociative excited state surface.
    :param array RMg: N Raman ground state frequencies (cm-1).
    :param array RMe: N Raman excited state frequencies (cm-1) or -1 if excited state surface is dissociative.
    :param array nquanta: M x N array containing the quanta of the N possible Raman modes for the M Raman lines to calculate. Use :py:func:`numpy.identity` to just calculate the fundamentals. Possible values are 0, 1, 2.    
    :returns: M + 1 - dimensional array containing the Rayleigh and M Raman overlaps.
    """
    ovlps = []
    
    N = len(D)
    M = nquanta.shape[0]
    
    # Frank-Condon factors <i|i(t)>_k and <f|i(t)>_k
    FC0 = []
    FC1 = []
    FC2 = []
    for i in range(N):
        if(RMg[i] == RMe[i]):
            FC0.append(t00A(t, D[i], RMg[i]))
            FC1.append(t10A(t, D[i], RMg[i]))
            FC2.append(t20A(t, D[i], RMg[i]))
        elif(RMe[i] == -1):
            FC0.append(t00D(t, D[i], RMg[i]))
            FC1.append(t10D(t, D[i], RMg[i]))           
            FC2.append(t20D(t, D[i], RMg[i]))
        else:
            FC0.append(t00B(t, D[i], RMg[i], RMe[i]))
            FC1.append(t10B(t, D[i], RMg[i], RMe[i]))
            FC2.append(t20B(t, D[i], RMg[i], RMe[i]))
    
    # Rayleigh / absorption overlap
    o = 1.0
    for i in range(N):
        o = o * FC0[i]
    ovlps.append(o) 

    # actual Raman overlaps
    for j in range(M):
        o = 1.0
        for i in range(N):
            if(nquanta[j][i] == 1):
                o = o * FC1[i]
            elif(nquanta[j][i] == 2):
                o = o * FC2[i]
        ovlps.append(o) 
    
    return ovlps
    
# ---------------------------------------------------------------------------------------------------------------------------------
def getZeroZeroEnergy(t, E0):
    """Calculate the oscillation term in the time domain due to the electronic zero-zero energy E0.
    
    :param array t: Time axis (fs).
    :param float E0: Difference between excited and ground state vibrational ground state energies, *zero-zero energy* (cm-1).    
    """
    return np.exp(-1j * E0 / hbar * t)  
    
# -----------------------------------------------------------------------------------------------------------------------------   
# Calculate the damping terms as function of time t.
def getHomogeneousDamping(t, Gamma, alpha = 0):
    """Calculates the damping term arising from the homogeneous linewidth of the electronic transition.
    
    :param array t: Time axis (fs).
    :param float Gamma: Decay rate according to :math:`1 / \\tau` in (cm-1), where :math:`tau` is exponential dephasing time.
    :param float alpha: Line shape parameter:
                        
                        - 1 = Gaussian
                        - 0 = Lorentzian
    :returns: Damping term in the time domain, :math:`e^{-g(t)}`.
    """
    g = alpha * (Gamma**2/hbar**2 * t**2) + (1 - alpha) * (Gamma/hbar * t)
    return np.exp(-g)
    
def getKuboDamping(t, Delta, Lambda):
    """Calculates the damping term using Kubo's *stochastic model*.
    
    :param array t: Time axis (fs).
    :param float Delta: Magnitude of solvent energy gap fluctuations (cm-1). This parameter also controls the effective line shape:
                            
                        - Delta >> Lambda = Lorentzian
                        - Delta << Lambda = Gaussian
    :param float Lambda: Effective frequency of solvent fluctuations (cm-1).
    :returns: Damping term in the time domain, :math:`e^{-g(t)}`.
    
    .. seealso:: Myers, *J. Raman. Spectrosc.* **28**, 389 (1997)
    """
    return np.exp( -(Delta/Lambda)**2 * ( np.exp(-Lambda/hbar*t) + Lambda/hbar*t - 1.0 ))

def getBrownianDamping(t, kappa, T, egamma, cutoff = 1e-6):
    """Calculate the damping term using Mukamel's Brownian oscillator model. This function also incorporates the homogeneous linewidth Gamma.
       
    :param array  t: Time axis (fs).
    :param float kappa: Lineshape parameter:

                        - kappa >> 1 = Lorentzian,
                        - kappa << 1 = Gaussian.
    :param float T: Temperature in K.
    :param float egamma: Electronic homogeneous linewidth (**FWHM**, cm-1).
    :param float cutoff: Cutoff for sum over Brownian oscillators. Typically between 1e-6 (default) and 1e-8. Check for convergence by re-running with different values.
    :returns: Damping term in the time domain, :math:`e^{-g(t)}`.

    .. seealso:: Myers, *J. Raman. Spectrosc.* **28**, 389 (1997)
    """
    temp = np.absolute(T)
    
    # ----------------------------------------------------------
    # 1: derive Mukamel's parameters from kappa, temp and egamma
    # I do not have a reference for this part - it's taken from Myers fortran code
    # Boltzmann beta
    beta = 1.0 / (kB * temp)                    # 1/cm-1
    
    # some 'a' parameter (this comes from Myers Fortran program)
    a = (2.355 + 1.76 * kappa) / (1.0 + 0.85 * kappa + 0.88 * kappa**2)
    
    # these are Mukamel's parameters in Myers, J. Raman. Spec. 28, 389 (1997), eqs. (35) to (38) 
    Lambda = kappa * egamma / a                 # cm-1 
    lmbda = beta * (Lambda / kappa)**2 / 2.0    # cm-1
    
    # ----------------------------------------------------------
    # 2: calculate the sum over n Brownian oscillators
    vs = np.zeros(len(t))               # this is the sum over the n oscillators as function of time in (cm-1)**-3
    n = 0
    while(True):
        n = n + 1
        vn = 2.0 * np.pi * n / beta     # cm-1
        vinc = (np.exp(-vn/hbar * t) + vn/hbar * t - 1) / (vn * (vn**2 - Lambda**2))
        vs = vs + vinc
        if(np.amax(np.absolute(vinc[1:] / vs[1:])) < cutoff):   # the first element of vs is always 0
            break

    # ----------------------------------------------------------
    # 3: calculate the damping function g(t)
    gexp = np.exp(-Lambda / hbar * t) + Lambda / hbar * t - 1.0     # dimensionless
    
    greal = (lmbda / Lambda) / np.tan(beta * Lambda / 2.0) * gexp   # dimensionless
    greal = greal + 4.0 * lmbda * Lambda / beta * vs                # dimensionless
    gimag = -(lmbda / Lambda) * gexp                                # dimensionless
    
    g = greal + 1j * gimag                                          # dimensionless
    
    return np.exp(-g)

# ---------------------------------------------------------------------------------------------------------------------------------
# 
def applyInhomogeneousBroadening(wn, y, sig, alpha = 1):
    """Convolute a spectrum with a Gaussian/Lorentzian to account for inhomogeneous broadening.
        
    :param array wn: Frequency axis in same units as sig (cm-1).
    :param array y: Input spectrum, same shape as wn.
    :param float sig: Width of convolution function in same units as x (standard deviation of Gaussian distribution). Must not be zero.
    :param float alpha: Lineshape parameter: 
    
                        - 1 = Gaussian, 
                        - 0 = Lorentzian.
    :returns: Convoluted spectrum (same shape as y).
    """    
    ck = alpha / (sig * np.sqrt(2 * np.pi)) * np.exp(-(wn - (wn[-1] + wn[0]) / 2.0)**2 / (2.0 * sig**2))
    ck += (1 - alpha) * sig / (np.pi * ((wn - (wn[-1] + wn[0])/2)**2 + sig**2))  
    # np.convolve uses a sum, whereas the function we want uses an integral; wn[1] - wn[0] is dwn    
    return (wn[1] - wn[0]) * np.convolve(y, ck, 'same')   
    
# --------------------------------------------------------------------------------------------------------------------------------
def prefA(eEL, M, IOR, dt):
    """Return the prefactor for the absorption cross section calculation in (A**2 / molec).
    
    :param array eEL: Laser excitation energy in (cm-1). May also be a single float value.
    :param float M: Electronic transition dipole length in (A).
    :param float IOR: Index of refraction of surrounding solvent / medium.
    :param float dt: Time increment used for integration (fs).
    :returns: Prefactor for absorption cross section calculation.  
    
    .. seealso:: Myers, Eq. (35).
    """
    # Version from Fortran code.
    #return 5.745e-3 * 1e-3 * M**2 * eEL / IOR
    return 5.745e-6 * M**2 * eEL * dt / IOR
    
    # The second version assumes that we use a full Fourier transform ( full FT = 2x half FT ) and corresponds to the numeric value of the prefactor in eq. (35)
    #return 2.8788e-6 * M**2 * eEL / IOR * dt

# -------------------------------------------------------------------------------------------------------------------------------
def prefR(eEL, M, eR, dt):
    """Return the prefactor for the Raman excitation profile calculation (A**2 / molec).
    
    :param array eEL: Laser excitation energies in (cm-1). Can also be a single floating point value.
    :param float M: Electronic transition dipole moment in (A).
    :param float eR: Stokes shift of the Raman line in (cm-1).
    :param float dt: Time increment for the integration (fs).
    :returns: The prefactor for the Raman excitation profile calculation.
    
    .. seealso:: Myers, Eq. (34) and following text.
    """
    # get energy of stokes shifted photons    
    eES = eEL - eR
    # the 1e-6 is for fs instead of ps in the integral and is consistent with Myers fortran code (it is different however from the 1e4 factor in Valley & Hoffman code!!)
    return 2.08288e-20 * 1e-6 * M**4 * eES**3 * eEL * dt**2

# ----------------------------------------------------------------------------------------------------------------------------
def getCrossSections(wn, dt, tdpart, ovlps, sshift, M, IOR, sig=0, ialpha=1):
    """Calculate the absorption and Raman cross-sections.
    
    :param array wn: Wavenumber axis in (cm-1).
    :param float dt: Time increment for integration / tdpart.
    :param array tdpart: Zero-zero energy and damping terms in the time domain, same shape as wn.
    :param array ovlps: M + 1 Absorption and Raman overlap integrals.
    :param float sshift: Vibrational freqencies of M Raman modes to calculate (cm-1).
    :param float IOR: Index of refraction of surrounding medium / solvent.
    :param float sig: Linewidth for inhomogeneous damping (standard deviation of Gaussian), set to zero if not used (default).
    :param float ialpha: Lineshape parameter for inhomogeneous damping:
    
                           - 1 = Gaussian (default),
                           - 0 = Lorentzian.
    :returns: Absorption and M Raman cross sections (arrays with same shape as wn): sigmaA, sigmaR[M].
    """    
    Npoints = len(wn)
    
    # absorption cross section - using the half sided FT (equivalent to rfft)
    tmp = np.real(Npoints * np.fft.irfft(ovlps[0] * tdpart, Npoints))
    if(sig > 0):
        tmp = applyInhomogeneousBroadening(wn, tmp, sig, ialpha)
    sigmaA = prefA(wn, M, IOR, dt) * tmp

    # Raman cross sections - using a standard FT
    sigmaR = []
    for i in range(len(ovlps)-1):   # iterate over all lines    
        tmp = np.absolute(Npoints * np.fft.ifft(ovlps[i+1] * tdpart, Npoints))**2 # use again the inverse transform to get "exp(1j w t)"
        if(sig > 0):
            tmp = applyInhomogeneousBroadening(wn, tmp, sig, ialpha)
        sigmaR.append(prefR(wn, M, sshift[i], dt) * tmp)        

    return sigmaA, sigmaR 
    
