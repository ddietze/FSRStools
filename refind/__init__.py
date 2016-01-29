"""
.. module: FSRStools.refind
   :platform: Windows
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu>

A collection of functions related to the optical refractive index.
Apart from providing tabulated indices for commonly used optical materials including dielectrics, common solvents and metals,
this module defines functions for more general use. These include Kramers-Kronig transformations, the Lorentz oscillator model,
general support for Sellmeier equations, inhomogeneous broadening and first, second and third order derivatives with respect to
wavelength. Furthermore, functions for conversion from refractive index to group index and permittivity and vice versa are provided.

.. note:: All functions for tabulated refractive indices expect wavelength to be in um.

**Change log:**

*01-29-2016*:

    - Changed syntax in `load_data` in response to numpy V 1.10 changes.

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
from scipy.interpolate import interp1d
import FSRStools.fitting as ft
import pkgutil
from StringIO import StringIO


# #################################################################################################################
# helper function to load local data files as numpy arrays independent of the absolute/relative paths of the module
# and the calling script
def load_data(filename, **argv):
    s = pkgutil.get_data(__package__, filename)
    ss = StringIO(s)

    # fix change in numpy version
    if np.__version__.startswith('1.10') and 'skiprows' in argv:
        argv['skip_header'] = argv.pop('skiprows')

    data = np.genfromtxt(ss, dtype='float', unpack=True, **argv)
    return data


# #################################################################################################################
# derivatives of refractive indices
# index N is function of wl and arguments *args
# wavelength in um
def dN(N, wl, *args):
    """First order derivative of the refractive index as function of wavelength using first-order finite difference.

    :param func N: (Complex) Refractive index as function of wavelength and possible arguments: N(wl, args).
    :param array wl: Wavelenth axis in um (can also be a single float).
    :param mixed args: Variable number of arguments to be provided to N.
    :returns: First order derivative of N with same shape as wl.
    """
    eps = 1e-4
    return (N(wl + eps, *args) - N(wl - eps, *args)) / (2.0 * eps)


def d2N(N, wl, *args):
    """Second order derivative of the refractive index as function of wavelength using first-order finite difference.

    :param func N: (Complex) Refractive index as function of wavelength and possible arguments: N(wl, args).
    :param array wl: Wavelenth axis in um (can also be a single float).
    :param mixed args: Variable number of arguments to be provided to N.
    :returns: First order derivative of N with same shape as wl.
    """
    eps = 1e-4
    return (dN(N, wl + eps, *args) - dN(N, wl - eps, *args)) / (2.0 * eps)


def d3N(N, wl, *args):
    """Third order derivative of the refractive index as function of wavelength using first-order finite difference.

    :param func N: (Complex) Refractive index as function of wavelength and possible arguments: N(wl, args).
    :param array wl: Wavelenth axis in um (can also be a single float).
    :param mixed args: Variable number of arguments to be provided to N.
    :returns: First order derivative of N with same shape as wl.
    """
    eps = 1e-4
    return (d2N(N, wl + eps, *args) - d2N(N, wl - eps, *args)) / (2.0 * eps)


# #################################################################################################################
# group velocity refractive index
# N is refractive index in the form N(l, *args)
def n_group(N, wl, *args):
    """Group velocity refractive index.

    :param func N: (Complex) Refractive index as function of wavelength and possible arguments: N(wl, args).
    :param array wl: Wavelenth axis in um (can also be a single float).
    :param mixed args: Variable number of arguments to be provided to N.
    :returns: Group velocity refractive index Ng = N - wl * dN with same shape as wl.
    """
    return N(wl, *args) - wl * dN(N, wl, *args)


# #################################################################################################################
# convert complex refractive index to dielectric function
def n_to_eps(n):
    """Convert (complex) refractive index to (complex) permittivity.

    :param array n: Refractive index to be converted to permittivity.
    :returns: Permittivity (same shape as n).
    """
    return np.real(n)**2 - np.imag(n)**2 + 1j * 2.0 * np.real(n) * np.imag(n)


# and vice versa
def eps_to_n(eps):
    """Convert (complex) permittivity to (complex) refractive index.

    :param array eps: Permittivity to be converted to refractive index.
    :returns: Refractive index (same shape as eps).
    """
    return np.lib.scimath.sqrt(eps)


#################################################################################################################
# inhomogeneous broadening function
def applyInhomogeneousBroadening(x, y, dx):
    """Convolute a refractive index / permittivity profile with a Gaussian to introduce inhomogeneous broadening.
    This broadened refractive index / permittivity still complies with Kramers Kronig relations.

    :param array x: Frequency axis in same units as dx.
    :param array y: Refractive index / permittivity profile to convolute (same shape as x).
    :param float dx: FWHM of Gaussian in same units as x.
    :returns: Convoluted / broadened refractive index / permittivity profile.
    """
    N = len(y)
    Npad = int(N / 2)
    delta = x[1] - x[0]
    ypad = np.pad(y, Npad, mode='reflect', reflect_type='odd')
    g = ft.gauss(delta * (np.arange(len(ypad)) - len(ypad) / 2), 1, 0, dx)

    # np convolve uses a sum, whereas the function we want uses an integral; x[1] - x[0] is dx
    # prefactor is normalization to unit area Gaussian
    return np.convolve(ypad, g, 'same')[Npad:-Npad] / np.sum(g)


# #################################################################################################################
# calculate the Kramers-Kronig transformation of absorption spectrum k as function of frequency w
# IMPORTANT: k and w have to be sampled on a uniform grid!!!
# this formulation is to be used for the refractive index!
def kramers_kronig(w, k):
    """Calculate the Kramers-Kronig transformation of absorption spectrum k as function of angular frequency w.
    This formulation is to be used for the refractive index, not for the permittivity!

    :param array w: Angular frequency axis.
    :param array k: Absorption spectrum (same shape as w).
    :returns: Real part of refractive index as obtained by Kramers-Kronig transformation.

    .. important:: k and w have to be sampled on a uniform grid!!!
    """
    h = np.absolute(w[1] - w[0])
    S = np.zeros(len(k))
    for i, _ in enumerate(k):
        if(i % 2 == 1):     # odd
            S[i] = np.sum(0.5 * (k[0::2] / (w[0::2] - w[i]) + k[0::2] / (w[0::2] + w[i])))
        else:               # even
            S[i] = np.sum(0.5 * (k[1::2] / (w[1::2] - w[i]) + k[1::2] / (w[1::2] + w[i])))

    return 2.0 / np.pi * 2.0 * h * S


# #################################################################################################################
# metals: type = gold, silver, copper, aluminum, chromium
# use interpolation of experimental data from Johnson and Christy for Ag, Au, Cu and Rakic for Al

# wavelength wl in um
def n_metal(wl, type='gold'):
    """Returns refractive index of some metals (Au, Ag, Cu, Al, Cr).
    Uses cubic interpolation of experimental values obtained from Johnson and Christy 1972 (Ag, Au, Cu) and Rakic 1998 (Al, Cr).

    :param array wl: Wavelength axis in um.
    :param str type: Type of metal ('Au'/'gold', 'Ag'/'silver', 'Cu'/'copper', 'Al'/'aluminum'/'aluminium', 'Cr'/'chromium').
    :returns: (Complex) Refractive index for given wavelength axis (same shape as wl).

    .. note:: Interpolation is done via SciPy's interp1d function, which throws an exception when the new x-coordinate is out of range.
    """
    if(type in ['gold', 'Au', 'copper', 'Cu', 'silver', 'Ag']):

        A = load_data("nobelmetals.dat")
        w0 = 2.9979e8 / (A[0] * 1.6022e-19 / 6.6261e-34) * 1e6  # eV to um

        if type == "silver" or type == "Ag":
            n0 = A[3] + 1j * A[4]
        elif type == "copper" or type == "Cu":
            n0 = A[1] + 1j * A[2]
        else:
            n0 = A[5] + 1j * A[6]

        B = interp1d(np.flipud(w0), np.flipud(n0), kind='cubic')

    elif type in ['aluminum', 'aluminium', 'Al']:

        A = load_data("METALS_Aluminium_Rakic.txt", skiprows=1)
        n0 = A[1] + 1j * A[2]
        B = interp1d(np.flipud(A[0]), np.flipud(n0), kind='cubic')

    elif type in ['chromium', 'Cr']:    # filmetrics.com

        A = load_data("Cr.txt", skiprows=2)
        n0 = A[1] + 1j * A[2]
        B = interp1d(A[0] / 1000.0, n0, kind='cubic')

    return B(wl)


# use Etchegoin model: Drude term (epsinf, lp, gp) plus Lorentz poles (A, phi, lambda, gamma)
# wl is wavelength in um
def n_metal_model(wl, epsinf, lp, gp, *poles):
    """Use the model presented in Etchegoin et al., *J Chem Phys* **125**, 164705 (2006) to describe the refractive index of nobel metals using a combination of Drude and Lorentz models.

    :param array wl: Wavelength axis in um.
    :param float epsinf: High frequency limit of permittivity.
    :param float lp: Plasma wavelength for Drude model (same units as wl).
    :param float gp: Damping term for Drude model (same units as wl).
    :param mixed poles: Parameters for Lorentz poles. For each pole provide (amplitude, phase offset, resonance wavelength and width).
    :returns: Complex refractive index (same shape as wl).
    """
    epsilon = complex(epsinf)
    epsilon -= 1.0 / (lp**2 * (1.0 / wl**2 + 1j / (gp * wl)))

    N = int(len(poles) / 4)
    for i in range(N):
        epsilon += poles[i * 4 + 0] / poles[i * 4 + 2] * (np.exp(1j * poles[i * 4 + 1]) / (1.0 / poles[i * 4 + 2] - 1.0 / wl - 1j / poles[i * 4 + 3]) + np.exp(-1j * poles[i * 4 + 1]) / (1.0 / poles[i * 4 + 2] + 1.0 / wl + 1j / poles[i * 4 + 3]))

    return np.lib.scimath.sqrt(epsilon)


# general sellmeier type of refractive index
# p = B0, B1, B2, .. Bx, C0, C1, C2, .. Cx
def n_sellmeier(wl, *p):
    """General Sellmeier-type of refractive index.

    :param array wl: Wavelength axis in um.
    :param mixed p: Sellmeier coefficients. For each term provide two parameters Bi and Ci in the form p = B0, B1, B2, .. Bx, C0, C1, C2, .. Cx.
    :returns: Refractive index n^2 - 1 = sum(Bi * wl^2 / (wl^2 - Ci)) with same shape as wl.
    """
    N = int(len(p) / 2)
    eps = 1.0
    for i in range(N):
        eps += p[i] * wl**2 / (wl**2 - p[N + i])
    return np.sqrt(eps)


# #################################################################################################################
# refractive indices of standard air

# wl is wavelength in um, data taken from refractiveindex.com
# dry, 15degC, 101325 Pa with 450ppm CO2
def n_air(wl):
    """Refractive index of standard air (dry, 15degC, 101325 Pa with 450ppm CO2). Data taken from refractiveindex.info.

    :param array wl: Wavelength axis in um.
    """
    return 1.0 + 0.05792105 / (238.0185 - 1.0 / wl**2) + 0.00167917 / (57.362 - 1.0 / wl**2)


# #################################################################################################################
# refractive indices of common glasses

# refractive index of common glasses
# type = SiO2, F2, NF2, SF11, BK7, SF10, Sapphire_E, Sapphire_O
# CoverGlass is siliconized 22x22mm thick cover glass, parameters from fit to absorption spectrum
# wl is in um
def n_glass(wl, type="SiO2"):
    """Refractive index of some common glasses using their respective Sellmeier coefficients.

    :param array wl: Wavelength axis in um.
    :param str type: Type of glass ('SiO2', 'F2', 'NF2', 'SF10', 'SF11', 'BK7', 'Sapphire_E', 'Sapphire_O', 'CoverGlass'). Cover glass refers to silanized microscope cover glasses with parameters fitted to experimental data.
    :returns: Refractive index (same shape as wl).
    """
    Bdict = {"SiO2": [0.696166300, 0.407942600, 0.897479400],
             "F2": [1.34533359, 0.209073118, 0.937357162],
             "NF2": [1.39757037, 0.159201403, 1.26865430],
             "SF11": [1.73759695, 0.313747346, 1.89878101],
             "BK7": [1.03961212, 0.231792344, 1.01046945],
             "SF10": [1.61625977, 0.259229334, 1.07762317],
             "Sapphire_E": [1.50397590, 0.550691410, 6.5927379],
             "Sapphire_O": [1.43134930, 0.650547130, 5.34140210],
             "CoverGlass": [4.21885399e-03, 3.80246387e-01, 1.27271765e+00]}

    Cdict = {"SiO2": [0.00467914826, 0.0135120631, 97.9340025],
             "F2": [0.00997743871, 0.0470450767, 111.886764],
             "NF2": [0.00995906143, 0.0546931752, 119.248346],
             "SF11": [0.0113188707, 0.0623068142, 155.236290],
             "BK7": [0.00600069867, 0.0200179144, 103.560653],
             "SF10": [0.0127534559, 0.0581983954, 116.607680],
             "Sapphire_E": [0.00548041129, 0.0147994281, 402.895140],
             "Sapphire_O": [0.00527992610, 0.0142382647, 325.017834],
             "CoverGlass": [1.14731882e-01, 9.92082291e-04, 6.53964380e+02]}

    try:
        args = Bdict[type] + Cdict[type]
    except:
        print("unknown glass type!")

    return n_sellmeier(wl, *args)


# #################################################################################################################
# refractive index of common materials used for optical coatings
# data taken from refractiveindex.info

def n_coatings(wl, type="MgF2_o"):
    """Refractive index of common materials used for optical coatings. Data taken from refractiveindex.info.

    .. versionadded:: 10-28-2015
        Added indices for rutile TiO2.

    :param array wl: Wavelength axis in um.
    :param str type: Type of coating ('MgF2_o', 'MgF2_e', 'ZnSe', 'TiO2_o', 'TiO2_e').
    :returns: Rerfractive index (same shape as wl).
    """
    if type == "TiO2_o":
        return np.sqrt(5.913 + 0.2441 / (wl**2 - 0.0803))
    elif type == "TiO2_e":
        return np.sqrt(7.197 + 0.3322 / (wl**2 - 0.0843))
    else:
        Bdict = {"MgF2_o": [0.27620, 0.60967, 0.0080, 2.14973],
                 "MgF2_e": [0.25385, 0.66405, 1.0899, 0.1816, 2.1227],
                 "ZnSe": [4.45813734, 0.467216334, 2.89566290]}
        Cdict = {"MgF2_o": [0.0, 0.08636**2, 18.0**2, 25.0**2],
                 "MgF2_e": [0.0, 0.08504**2, 22.2**2, 24.4**2, 40.6**2],
                 "ZnSe": [0.200859853**2, 0.391371166**2, 47.1362108**2]}

        try:
            args = Bdict[type] + Cdict[type]
        except:
            raise ValueError("unknown coating type!")

        return n_sellmeier(wl, *args)


# #################################################################################################################
# refractive index of common liquids
# data taken from refractiveindex.info
# ----------------------------------------------------------------------------------------------------------------------------

def n_liquid(wl, type='water'):
    """Refractive index of common solvents. Data taken from refractiveindex.info.

    :param array wl: Wavelength axis in um.
    :param str type: Type of liquid / solvent ('water', 'cyclohexane', 'ethanol', 'methanol').
    :returns: Rerfractive index (same shape as wl).
    """
    if(type == "cyclohexane"):
        return 1.41545 + 0.00369 / wl**2 + 0.00004 / wl**4

    elif(type == "ethanol"):
        return 1.35265 + 0.00306 / wl**2 + 0.00002 / wl**4

    elif(type == "methanol"):
        return 1.294611 + 12706.403e-6 / wl**2

    else:
        A = load_data("LIQUIDS_Water_Hale.txt", skiprows=1)
        n0 = A[1] + 1j * A[2]
        B = interp1d(A[0], n0, kind='cubic')
        return B(wl)


# #################################################################################################################
# refractive index of polymers
# data taken partly from refractiveindex.info

def n_polymer(wl, type="PVA"):
    """Refractive index of some polymers. Data for PVA taken from *J. Phys. D* **44**, 205105 (2011).

    :param array wl: Wavelength axis in um.
    :param str type: Type of polymer ('PVA').
    :returns: Rerfractive index (same shape as wl).
    """
    if type == "PVA":
        # taken from J. Phys. D 44, 205105 (2011)
        return np.lib.scimath.sqrt(2.34 - 3.06e-2 * wl**2)

    return []


# #################################################################################################################
# Lorentz Oscillator Model  - see http://de.wikipedia.org/wiki/Lorentzoszillator#
# wl is wavelength (in um)
# ebg is background dielectric constant
# p contains parameters for each oscillator (A, lambda0, dlambda)
def n_LorOsc(wl, ebg, *p):
    """Refractive index from Lorentz oscillator model. See http://de.wikipedia.org/wiki/Lorentzoszillator for more details.

    :param array wl: Wavelength axis in um.
    :param float ebg: Background / high frequency dielectric constant.
    :param mixed p: Parameters for each oscillator / pole. For each pole provide (amplitude, resonance wavelength, width).
    :returns: Complex refractive index (same shape as wl).
    """
    eps = complex(ebg)
    for i in range(int(len(p) / 3)):
        eps += p[3 * i + 0] / ((1 / p[3 * i + 1])**2 - (1 / wl)**2 - 1j * p[3 * i + 2] / (p[3 * i + 1]**2 + p[3 * i + 2] * p[3 * i + 1]) / wl)
    return np.lib.scimath.sqrt(eps)


# shortcut to absorption coefficient
# result in 1/um
def alpha_LorOsc(wl, ebg, *p):
    """Absorption coefficient from Lorentz oscillator model. See http://de.wikipedia.org/wiki/Lorentzoszillator for more details.

    :param array wl: Wavelength axis in um.
    :param float ebg: Background / high frequency dielectric constant.
    :param mixed p: Parameters for each oscillator / pole. For each pole provide (amplitude, resonance wavelength, width).
    :returns: Absorption coefficient (same shape as wl).
    """
    return 4.0 * np.pi / wl * np.imag(n_LorOsc(wl, ebg, *p))
