"""
.. module: FSRStools.rraman
   :platform: Windows
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu>

Resonance Raman excitation profile calculation based on the time-domain picture of resonance Raman. See Myers and Mathies in *Biological Applications of Raman Spectroscopy*, Vol. 2, pp. 1-58 (John Wiley and Sons, New York, 1987) for details (referred to as Myers in the following). The code is mainly based on Myers' Fortran 77 code (see Appendix of PhD Thesis of K. M. Spillane, 2011, UC Berkeley for source code).

**Changelog:**

*10-7-2015:*

   - Added / modified functions for calculating fluorescence spectra.
   - Added a convenience function to calculate Raman spectra from a set of excitation profiles.
   - Added some more damping functions and phenomenological support for Stokes shift in simple homogeneous damping function.

*10-21-2015:*

    - Some bug fixes concerning the prefactors and the normalization of the fluorescence spectra.
    - Fixed a bug regarding the Raman overlaps.


**Example Code**

Here is a short example calculating Myers' *Gedankenmolecule* from Myers and Mathies::

    import numpy as np
    import FSRStools.rraman as rr

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
    t, wn = rr.getAxes(tmax, dt)

    # zero-zero energy and damping
    # add here all time domain stuff
    TDpart = rr.getHomogeneousDamping(t, Gamma, halpha)

    # time dependent overlap integrals
    OVLPS = rr.getOverlaps(t, D, RMg, RMe, nquanta)

    # calculate cross-sections
    sigmaA, sigmaR, kF = rr.getCrossSections(t, wn, E0, OVLPS, sshift, M, IOR, TDpart, sig, ialpha)


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

   Copyright 2014, 2015 Daniel Dietze <daniel.dietze@berkeley.edu>.
"""
import numpy as np

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


def wn2radperfs(e):
    """Wavenumber (cm-1) to angular frequency (rad / fs).
    """
    return e / hbar


def wn2lambda(w):
    """Convert wavenumber (cm-1) to wavelength (nm).
    """
    return 1e7 / w


def lambda2wn(w):
    """Convert wavelength (nm) to wavenumber (cm-1).
    """
    return 1e7 / w


def getWnIndex(wn, wn0):
    """Get the index into an array of wavenumbers wn with wavenumber closest to wn0. Use this function for :py:func:`getRamanSpectrum`.
    """
    if np.amin(wn) > wn0 or np.amax(wn) < wn0:
        print "Warning: wn0 lies outside of wn."
    return np.argmin(np.absolute(wn - wn0))


def getAxes(tmax, dt):
    """Create time and frequency axes for the resonance Raman calculations.

    :param float tmax: Endpoint for time domain calculation (fs). This value should be high enough to capture the full dephasing.
    :param float dt: Increment of time axis (fs). This value should be small enough to capture the highest vibronic feature in the excited state.
    :returns: Time axis (fs) and frequency axis (cm-1).
    """
    t = np.arange(0, tmax + dt, dt)
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
    return 1e3 * np.log(10.0) * eSpctr / 6.0221e23 * 1e8 * 1e8 / IOR


def diff2absRamanCS(diffRaCS, rho):
    """Convert the differential Raman cross section (A**2/molec sr) to absolute Raman cross section in (A**2 / molec) for a given depolarization ratio rho.

    :param float diffRaCS: Differential Raman cross section (A**2/molec sr).
    :param float rho: Associated depolarization ratio of this Raman mode.
    :returns: Absolute Raman cross section in (A**2 / molec).
    """
    return 8.0 * np.pi / 3.0 * (1.0 + 2.0 * rho) / (1.0 + rho) * diffRaCS


def getRamanSpectrum(wn, iEL, RMg, nquanta, sigmaR, dw=10.0, alpha=0):
    """
    Convenience function to calculate the Raman spectrum. The spectrum is scattered power per infinitesimal frequency normalized to incident power times molecular density (cm-3) times path length (cm). See Myers, *Chem. Phys.* **180**, 215 (1994), Eq. 7 for details.

    :param array wn: Wavenumber axis (Stokes shift, not electronic).
    :param int iEL: Index into sigmaR corresponding to the pump energy of the laser.
    :param array RMg: Ground state Raman frequencies
    :param array nquanta: M x N array containing the quanta of the N possible Raman modes for the M Raman lines to calculate. Use :py:func:`numpy.identity` to just calculate the fundamentals. Possible values are 0, 1, 2.
    :param array sigmaR: Array of M Raman cross sections that have been calculated by :py:func:`getCrossSections` (in A**2 / molec).
    :param float dw: Phenomenological FWHM linewidth of the Raman lines in cm-1 (default = 10 cm-1).
    :param float alpha: Line shape parameter to be used for the Raman spectrum:

        - 1 = Gaussian
        - 0 = Lorentzian (default)
    :returns: Calculated Raman spectrum (same shape as wn).
    """
    spectrum = np.zeros(len(wn))
    if iEL < 0 or iEL >= len(sigmaR[0]):
        print "Error: iEL is out of range!"
        return spectrum

    # iterate over all M modes
    for i, nM in enumerate(nquanta):
        # get frequency of this mode
        wR = np.sum(nM * RMg)

        # add Lorentzian part of lineshape
        spectrum = spectrum + (1.0 - alpha) * sigmaR[i][iEL] * 1e-16 * (dw / (2.0 * np.pi * ((wn - wR)**2 + dw**2 / 4.0)))

        # add Gaussian part of lineshape
        spectrum = spectrum + alpha * sigmaR[i][iEL] * 1e-16 * ((2.0 * np.sqrt(np.log(2) / np.pi)) / dw * np.exp(-4.0 * np.log(2.0) * (wn - wR)**2 / dw**2))

    return spectrum


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
    return Delta / np.sqrt(2) * (np.exp(-1j * eVIB / hbar * t) - 1.0)  # * t00A(t, Delta, eVIB)


def t20A(t, Delta, eVIB):
    """Time dependent overlap integral between vibrational ground and second excited state of electronic ground and excited state with equal ground and excited state vibrational frequencies.

    :param array t: Time axis in (fs).
    :param float Delta: Displacement of excited state potential energy surface along this vibrational coordinate in dimensionless coordinates.
    :param float eVIB: Vibrational frequency (cm-1).
    :returns: 2-0 overlap integral as function of time (same shape as t).

    .. seealso:: Myers, Eqs. (37) - (39).
    """
    return Delta**2 / (2 * np.sqrt(2)) * (np.exp(-1j * eVIB / hbar * t) - 1.0)**2  # * t00A(t, Delta, eVIB)


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
    gt = 1j / 2.0 * np.log(1j * wg / we * swe + cwe) + pt * (qt - Delta) / 2.0  # skip -E0 t / hbar
#    gt = gt + wg * t / 2.0 # add +w t / 2 using ground state frequency as this compensates the -w t / 2.0 term coming from the FFT

    # add the following term to recover t00A for eg = ee
    gt = gt - 1j / 2.0 * np.log(1j * np.sin(wg * t) + np.cos(wg * t))

    at = -0.5 * 1j * (1j * cwe - (we / wg) * swe) / (1j * (wg / we) * swe + cwe)

    a = at + 0.5
    pp = pt - 2.0 * 1j * at * qt
    gp = 1j * at * qt**2 - pt * qt + gt

    return a**(-0.5) * np.exp(-pp**2 / (4.0 * a)) * np.exp(1j * gp)


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
    at = -0.5 * 1j * (1j * cwe - (we / wg) * swe) / (1j * (wg / we) * swe + cwe)

    a = at + 0.5
    pp = pt - 2.0 * 1j * at * qt
    return 2**(-0.5) * pp / (1j * a)  # * t00B(t, Delta, eg, ee)


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
    at = -0.5 * 1j * (1j * cwe - (we / wg) * swe) / (1j * (wg / we) * swe + cwe)

    a = at + 0.5
    pp = pt - 2.0 * 1j * at * qt
    return -8**(-0.5) * (pp**2 / a**2 + 2. * (1. - 1. / a))  # * t00B(t, Delta, eg, ee)


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

    tmp = (1.0 + 1j * eVIB / hbar * t / 2.0)**(-0.5) * np.exp(-beta**2 * (6 * t**2 + 1j * eVIB / hbar * t**3) / (24 * hbar**2))
    tmp = tmp * np.exp(1j * eVIB / hbar * t / 2.0)      # add this term to compensate for the -1j w t / 2 term coming from the FFt
    return tmp


def t10D(t, beta, eVIB):
    """Time dependent overlap integral between vibrational ground and first excited state of electronic ground and excited state with a linear dissociative excited state surface along this vibrational coordinate.

    :param array t: Time axis in (fs).
    :param float beta: Slope of excited state potential energy surface (dV / dq) in (cm-1) (q is dimensionless coordinate).
    :param float eVIB: Vibrational frequency (cm-1).
    :returns: 1-0 overlap integral as function of time (same shape as t).

    .. seealso:: Myers, Eqs. (52) - (54).
    """
    return -1j * 2**(-0.5) * (beta * t / hbar)  # * t00D(t, beta, eVIB)


def t20D(t, beta, eVIB):
    """Time dependent overlap integral between vibrational ground and second excited state of electronic ground and excited state with a linear dissociative excited state surface along this vibrational coordinate.

    :param array t: Time axis in (fs).
    :param float beta: Slope of excited state potential energy surface (dV / dq) in (cm-1) (q is dimensionless coordinate).
    :param float eVIB: Vibrational frequency (cm-1).
    :returns: 2-0 overlap integral as function of time (same shape as t).

    .. seealso:: Myers, Eqs. (52) - (54).
    """
    return -2**(-0.5) * (beta**2 * t**2 / (2.0 * hbar**2) - 1j * eVIB / hbar * t / (2.0 + 1j * eVIB / hbar * t))  # * t00D(t, beta, eVIB)


# ---------------------------------------------------------------------------------------------------------------------------------
def getOverlaps(t, D, RMg, RMe, nquanta):
    """Calculate the time dependent overlap integrals / Franck-Condon factors :math:`<i|i(t)>_k` and :math:`<f|i(t)>_k`.

    .. versionchanged:: 10-07-2015
        Format of return value changed.

    :param array t: Time axis in (fs).
    :param array D: Array of N normalized displacements of excited state surfaces (deltas), or slope of linear dissociative excited state surface.
    :param array RMg: N Raman ground state frequencies (cm-1).
    :param array RMe: N Raman excited state frequencies (cm-1) or -1 if excited state surface is dissociative.
    :param array nquanta: M x N array containing the quanta of the N possible Raman modes for the M Raman lines to calculate. Use :py:func:`numpy.identity` to just calculate the fundamentals. Possible values are 0 (no excitation), 1 (fundamental), 2 (first overtone).
    :returns: M + 2 - dimensional array containing the Rayleigh, fluorescence and M Raman overlaps.
    """
    ovlps = []

    N = len(D)
    M = nquanta.shape[0]

    # Frank-Condon factors <i|i(t)>_k and <f|i(t)>_k
    FC0 = []
    FC0p = []
    FC1 = []
    FC2 = []
    for i in range(N):
        if(RMg[i] == RMe[i]):
            FC0.append(t00A(t, D[i], RMg[i]))
            FC0p.append(FC0[-1])  # fluorescence overlap is identical to absorption overlap when frequencies are equal
            FC1.append(t10A(t, D[i], RMg[i]))
            FC2.append(t20A(t, D[i], RMg[i]))
        elif(RMe[i] == -1):
            FC0.append(t00D(t, D[i], RMg[i]))
            FC0p.append(np.zeros(len(t)))  # fluorescence is negligible from dissociative surface
            FC1.append(t10D(t, D[i], RMg[i]))
            FC2.append(t20D(t, D[i], RMg[i]))
        else:
            FC0.append(t00B(t, D[i], RMg[i], RMe[i]))
            FC0p.append(t00B(t, D[i], RMe[i], RMg[i]))  # fluorescence overlap has excited state and ground state Raman frequencies switched
            FC1.append(t10B(t, D[i], RMg[i], RMe[i]))
            FC2.append(t20B(t, D[i], RMg[i], RMe[i]))

    # go to numpy array..
    FC0 = np.array(FC0)
    FC0p = np.array(FC0p)
    FC1 = np.array(FC1)
    FC2 = np.array(FC2)

    # Rayleigh / absorption overlap
    oabs = 1.0 + 0.0 * 1j       # reuse this term for the raman overlaps
    for i in range(N):
        oabs = oabs * FC0[i]
    ovlps.append(oabs)

    # fluorescence overlap
    o = 1.0 + 0.0 * 1j
    for i in range(N):
        o = o * FC0p[i]
    ovlps.append(o)

    # actual Raman overlaps
    for j in range(M):
        o = 1.0 * oabs                      # all raman modes are based on this product and additional terms given by the excited modes
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
def getHomogeneousDamping(t, Gamma, alpha=0, lmbda=0):
    """Calculates the damping term arising from the homogeneous linewidth of the electronic transition. Offers phenomenological support for Stokes shift.

    .. note:: Added phenomenological Stokes shift to input parameters on 10-12-2015. See for example *New J Phys* **11**, 015001 (2009), Eqs. (1) and (2).

    :param array t: Time axis (fs).
    :param float Gamma: Decay rate according to :math:`1 / \\tau` in (cm-1), where :math:`tau` is exponential dephasing time.
    :param float alpha: Line shape parameter:

                        - 1 = Gaussian
                        - 0 = Lorentzian
    :param float lmbda: Phenomenological Stokes shift (cm-1) which is added as imaginary part to g(t). Compared to the Brownian oscillator models, lmbda **is** the observed Stokes shift. (default = 0)
    :returns: Damping term in the time domain, :math:`e^{-g(t) - i \lambda t / 2 \hbar}`.
    """
    g = alpha * (Gamma**2 / hbar**2 * t**2) + (1 - alpha) * (Gamma / hbar * t) + 1j * lmbda / 2.0 * t / hbar
    return np.exp(-g)


def getKuboDamping(t, Delta, Lambda):
    """Calculates the damping term using Kubo's *stochastic model*. This model describes the broadening, but does not yield solvent induced Stokes shifts.

    :param array t: Time axis (fs).
    :param float Delta: Magnitude of solvent energy gap fluctuations (cm-1). This parameter also controls the effective line shape:

                        - Delta >> Lambda = Lorentzian
                        - Delta << Lambda = Gaussian
    :param float Lambda: Effective frequency of solvent fluctuations (cm-1).
    :returns: Damping term in the time domain, :math:`e^{-g(t)}`.

    .. seealso:: Myers, *J. Raman. Spectrosc.* **28**, 389 (1997)
    """
    return np.exp(-(Delta / Lambda)**2 * (np.exp(-Lambda / hbar * t) + Lambda / hbar * t - 1.0))


def getBrownianDamping(t, kappa, T, egamma, cutoff=1e-6):
    """Calculate the damping term using Mukamel's Brownian oscillator model based on Myers Fortran code. The real part of g(t) leads to a Gaussian broadening of the spectra, while the imaginary part leads to a solvent induced Stokes shift.

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
        vinc = (np.exp(-vn / hbar * t) + vn / hbar * t - 1) / (vn * (vn**2 - Lambda**2))
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


def getBrownianDamping2(t, lmbda, Lambda, T=298.0, cutoff=1e-6):
    """Calculate pure electronic dephasing due to interaction with solvent using frictionally overdamped Brownian oscillator model.
    The real part of g(t) leads to a Gaussian broadening of the spectra, while the imaginary part leads to a solvent induced Stokes shift.

    :param array t: Time axis in fs.
    :param float lmbda: Solvent contribution to reorganization energy (cm-1).
    :param float Lambda: Inverse of characteristic time scale for solvent fluctuations (fs-1).
    :param float T: Temperature (K, default = 298 K).
    :param float cutoff: Cutoff value for summation over brownian oscillators (default 1e-6).
    :returns: Damping term in the time domain, :math:`e^{-g(t)}`.

    .. seealso:: This implementation is taken from Kulinowksi, *J Phys Chem* **99**, 9017 (1995), Eqs. (10a) to (10d).
    """
    beta = 1.0 / (kB * np.absolute(T))
    lmb = lmbda / hbar  # convert to fs-1

    # calculate real part as sum over oscillators
    gR = 0.0
    i = 1.0
    while(1):
        nun = 2.0 * np.pi / (hbar * beta) * i       # frequency of ith oscillator
        dg = (np.exp(-nun * t) + nun * t - 1.0) / (nun * (nun**2 - Lambda**2))
        gR = gR + dg
        i = i + 1.0
        if np.sum(np.absolute(np.dg)) / np.sum(np.absolute(gR)) < cutoff:
            break

    gR = gR * 4.0 * lmb * Lambda / (hbar * beta)
    gR = gR + (lmb / Lambda) * np.cot(hbar * beta * Lambda / 2.0) * (np.exp(-Lambda * t) + Lambda * t - 1.0)

    # calculate imaginary part = Stokes shift
    gI = -(lmb / Lambda) * (np.exp(-Lambda * t) - 1.0)

    # assemble
    g = gR + 1j * gI                                          # dimensionless

    return np.exp(-g)


def getBrownianDampingSlowMod(t, lmbda, T=298.0):
    """Calculate pure electronic dephasing due to interaction with solvent using frictionally overdamped Brownian oscillator model in the high-temperature and slow-modulation limit.
    The real part of g(t) leads to a Gaussian broadening of the spectra, while the imaginary part leads to a solvent induced Stokes shift.

    :param array t: Time axis in fs.
    :param float lmbda: Solvent contribution to reorganization energy (cm-1).
    :param float T: Temperature (K, default = 298 K).
    :returns: Damping term in the time domain, :math:`e^{-g(t)}`.

    .. seealso:: This implementation is taken from Kulinowksi, *J Phys Chem* **99**, 9017 (1995), Eq. (11).
    """
    lmb = lmbda / hbar      # convert to fs-1
    return np.exp(-(lmb * kB * np.absolute(T) * t**2 / hbar + 1j * lmb * t))


# ---------------------------------------------------------------------------------------------------------------------------------
#
def applyInhomogeneousBroadening(wn, y, sig, alpha=1):
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
    ck += (1 - alpha) * sig / (np.pi * ((wn - (wn[-1] + wn[0]) / 2)**2 + sig**2))
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
    # to convert from esu to SI divide by 4 pi eps0
    # the factor / 2 arises from the normalization of numpy of the rfft to match the amplitude of fft
    # so rfft is not completely identical to half-sided FT integral
    return 5.7579e-6 * M**2 * eEL * dt / IOR / 2.0


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
    # to convert from esu to SI divide by (4 pi eps0)**2
    return 2.0831e-20 * 1e-6 * M**4 * eES**3 * eEL * dt**2


# --------------------------------------------------------------------------------------------------------------------------------
def prefF(eEF, M, IOR, dt):
    """Return the prefactor for the fluorescence efficiency calculation (unitless). See :py:func:`getCrossSections` for more details.

    :param array eEF: Fluorescence energy in (cm-1). May also be a single float value.
    :param float M: Electronic transition dipole length in (A).
    :param float IOR: Index of refraction of surrounding solvent / medium.
    :param float dt: Time increment used for integration (fs).
    :returns: Prefactor for fluorescence efficiency calculation.

    .. seealso:: Myers, *Chem. Phys.* **180**, 215 (1994), Eqs. (6) and (26).
    """
    # to convert from esu to SI divide by 4 pi eps0
    # the factor / 2 arises from the normalization of numpy of the rfft to match the amplitude of fft
    # so rfft is not completely identical to half-sided FT integral
    return 3.6656e-22 * IOR * M**2 * eEF**3 * dt / 2.0


# ----------------------------------------------------------------------------------------------------------------------------
def getCrossSections(t, wn, E0, ovlps, sshift, M, IOR, damp=1, sig=0, ialpha=1):
    """Calculate the absorption and Raman cross-sections and the fluorescence efficiency. The latter is a unitless quantity which may be used
    to calculate the fluorescence rate (=rate of spontaneous emission) by integrating over the frequency axis (see Myers, *Chem. Phys.* **180**, 215 (1994) Eq. 6 and discussion).

    .. note:: Changed shape of input parameters and shape of return values on 10-07-2015.

    :param array t: Time axis in (fs). This axis is used for the calculation of the zero-zero energy term in the time domain.
    :param array wn: Wavenumber axis in (cm-1). Same shape as t.
    :param array E0: Zero-zero energy. This function then calculates the time domain part using `getZeroZeroEnergy`.
    :param array ovlps: M + 2 Absorption, fluorescence and Raman overlap integrals.
    :param float sshift: Vibrational freqencies of M Raman modes to calculate (cm-1).
    :param float M: Electronic transition dipole length (A).
    :param float IOR: Index of refraction of surrounding medium / solvent.
    :param array damp: Damping function in the time domain. Same shape as t. Set to 1 if no damping is used (default).
    :param float sig: Linewidth for inhomogeneous damping (standard deviation of Gaussian), set to zero if not used (default).
    :param float ialpha: Lineshape parameter for inhomogeneous damping:

                           - 1 = Gaussian (default),
                           - 0 = Lorentzian.
    :returns: Absorption (sigmaA), M Raman cross sections (sigmaR[M]), both in A**2 / mol., and fluorescence efficiency spectrum, kF (arrays have same shape as wn); all as function of excitation wavenumber.
    """
    Npoints = len(wn)
    dt = t[1] - t[0]

    # caluclate zero-zero time domain part
    tdpart = getZeroZeroEnergy(t, E0)

    # absorption cross section - using the half sided FT (equivalent to rfft)
    tmp = np.real(Npoints * np.fft.irfft(ovlps[0] * tdpart * damp, Npoints))
    if(sig > 0):
        tmp = applyInhomogeneousBroadening(wn, tmp, sig, ialpha)
    sigmaA = prefA(wn, M, IOR, dt) * tmp

    # fluorescence rate / intensity - using half sided FT - similar to absorption
    # in order to account for the sign change, the zero-zero energy time domain part and the damping term had to be separated;
    # use the tdpart conjugated and change irfft by hfft to get the factor exp(-1j w t)
    # numpy does not normalize the forward FFT, so no factor Npoints
    tmp = np.real(np.fft.hfft(ovlps[1] * np.conjugate(tdpart) * damp, Npoints))
    if(sig > 0):
        tmp = applyInhomogeneousBroadening(wn, tmp, sig, ialpha)
    kF = prefF(wn, M, IOR, dt) * tmp

    # Raman cross sections - using a standard FT
    sigmaR = []
    for i, ovlp in enumerate(ovlps[2:]):   # iterate over all lines
        tmp = np.absolute(Npoints * np.fft.ifft(ovlp * tdpart * damp, Npoints))**2  # use again the inverse transform to get "exp(1j w t)"
        if(sig > 0):
            tmp = applyInhomogeneousBroadening(wn, tmp, sig, ialpha)
        sigmaR.append(prefR(wn, M, sshift[i], dt) * tmp)

    return sigmaA, sigmaR, kF
