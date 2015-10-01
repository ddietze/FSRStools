""" 
.. module: FSRStools.fitting
   :platform: Windows
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu> 

A comprehensive bundle of mathematical functions and routines for fitting experimental data. 
Provides methods for
 
    - polynomials, Lorentzians, Gaussians, Voigts, exponentials, etc.
    - Lorentz oscillator model,
    - fitting beam waists using the knife edge method, 
    - LPSVD fitting of periodic noisy data,
    - short-time Fourier transform,
    - principal component analysis (using NIPALS),


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
import scipy.optimize as spopt
from scipy.special import erf
from scipy.linalg import hankel, block_diag

# ====================================================================================================================
# Wrapper around SciPy fitting functions
# ====================================================================================================================

def curve_fit(*args):
    """A convenience wrapper for :py:func:`scipy.optimize.curve_fit`, so you do not have to import scipy in addition.
    
    :param mixed args: Arguments that are directly passed to :py:func:`scipy.optimize.curve_fit`.
    :returns: Fitting results and covariance matrix (popt, pcov).
    """
    return spopt.curve_fit(*args)

# ====================================================================================================================
# Weighted average
# ====================================================================================================================
def weighted_mean(x, s):
    """Calculate the weighted average over a given data set and its uncertainties.
    
    :param array x: Data array.
    :param array s: Standard deviation / uncertainty for each value in x.
    :returns: Weighted average of <x> and <s>.
    """
    x = np.array(x)
    s = np.array(s)
    sm = np.sqrt( 1.0 / np.sum( 1.0 / s**2 ) )
    xm = sm**2 * np.sum( x / s**2 )
    return xm, sm   
    
# ====================================================================================================================
# Polynomials
# ====================================================================================================================

def const(x, a):
    """Returns a constant.
    
    :param array x: x-values.
    :param float a: Value of the constant.
    :returns: :math:`f(x) = a` (same shape as x).
    """
    return np.ones(x.shape) * a

def proportional(x, b):
    """Proportionality, i.e., a line through the origin.
    
    :param array x: x-values.
    :param float b: Slope.
    :returns: :math:`f(x) = b x`.
    """
    return x * b
    
def line(x, a, b):
    """A line.
    
    :param array x: x-values.
    :param float a: y-intercept.
    :param float b: Slope.
    :returns: :math:`f(x) = b x + a`. 
    """
    return a + x * b

def parabola(x, a, b, c):
    """A parabola.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :returns: :math:`f(x) = c x^2 + b x + a`.
    """
    return a + x * b + x**2 * c

def parabola2(x, y0, x0, a):
    """Another version of the parabola using the vertex (x0, y0).
    
    :param array x: x-values.
    :param float y0: y-coordinate of the vertex.
    :param float x0: x-coordinate of the vertex.
    :param float a: curvature.
    :returns: :math:`f(x) = a (x - x_0)^2 + y_0`.
    """
    return y0 + a * (x-x0)**2

def cubic(x, a, b, c, d):
    """Third order polynomial.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :returns: :math:`f(x) = d x^3 + c x^2 + b x + a`.
    """
    return a + x * b + x**2 * c + x**3 * d

def quartic(x, a, b, c, d, e):
    """Fourth order polynomial.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :param float e: Quartic term.    
    :returns: :math:`f(x) = e x^4 + d x^3 + c x^2 + b x + a`.
    """
    return a + x * b + x**2 * c + x**3 * d + x**4 * e

def poly(x, *coeffs):
    """A general polynomial.
    
    :param array x: x-values.
    :param variable coeffs: Sequence of polynomial terms *k* in increasing order.
    :returns: :math:`f(x) = \\sum_i k_i x^i`.

    Example::

        # a third order polynomial
        y = poly(x, 0, 1, 2, 3)
        
        # which is the same as 
        y = cubic(x, 0, 1, 2, 3)
    """
    k = 0.0
    res = 0.0
    for c in coeffs:
        res += c * np.power(x, k)
        k += 1.0
    return res

def lagrange(x, x0, y0):
    """Construct the polynomial over x with the least degree that passes through all points (x0, y0) using Lagrange polynomials.
    
    :param array x: x-values.
    :param array x0: List of x-coordinates of interpolation points.
    :param array y0: List of y-coordinates of interpolation points (same shape as x0).
    :returns: Lagrange polynomial :math:`f(x)` that passes through all data points.
    
    .. note:: The Lagrange form is very susceptible to Runge's phenomenon, i.e., interpolation divergence; especially when many data points are used.
    
    Example::
    
        # a parabola through three points
        x = np.linspace(-5, 5, 64)
        y = lagrange(x, [-3, 0, 3], [1, -3, 6])
    
    """
    l = np.ones((len(x0), len(x)))
    for i in range(len(x0)):
        for j in range(len(x0)):
            if(i != j):
                l[i] = l[i] * (x - x0[j]) / (x0[i] - x0[j])
    
    out = np.zeros(len(x))
    for i in range(len(x0)):
        out += y0[i] * l[i]
    return out  
 
# ====================================================================================================================
# Powers
# ====================================================================================================================

# single power law without offset
def power(x, A, a):
    """Power law without offset.
    
    :param array x: x-values.
    :param float A: Amplitude.
    :param float a: Exponent.
    :returns: :math:`f(x) = A x^a`.
    """
    return A * x**a

# single power law with offset
def power_const(x, y0, A, a):
    """Power law with constant offset.
    
    :param array x: x-values.
    :param float y0: Offset.
    :param float A: Amplitude.
    :param float a: Exponent.
    :returns: :math:`f(x) = A x^a + y_0`.
    """
    return A * x**a + y0
    
# ====================================================================================================================
# Exponentials and related decay functions
# ====================================================================================================================

# a simple exponential
def exp(x, A, dx):
    """A single *decaying* exponential without baseline.
    
    :param array x: x-values.
    :param float A: Amplitude (can be negative).
    :param float dx: Decay constant (can be negative, which results in a growing exponential).
    :returns: :math:`f(x) = A e^{-x \\Delta x}`.
    """
    return A * np.exp(-x / dx)

# as many exponentials as you want plus constant
# for every exponential provide A, dx in *p
def exponentials(x, y0, *p):
    """Sum over a variable number of *decaying* exponentials with constant offset.
    
    :param array x: x-values.
    :param float y0: Offset.
    :param variable p: Coefficients of exponentials. For every exponential provide 
    
                       - A amplitude (can be negative),
                       - dx decay constant (can be negative, which results in a growing exponential).
    :returns: :math:`f(x) = y_0 + \\sum_i A_i e^{-x / \\Delta x_i}`.
    """
    out = np.ones(len(x)) * y0
    for i in range(int(len(p)/2)):
        out += p[2*i+0] * np.exp(-x / p[2*i+1])
    return out
    
# exponentials convoluted with a Gaussian
# y0 is offset
# s is sigma of Gaussian # FWHM of Gaussian
# x0 is time offset
# for every exponential provide A, dx in *p
def exp_gauss(x, y0, s, x0, *p):
    """Sum over variable number of exponentials convoluted with a single Gaussian. Use this version for fitting kinetic data in order to take into account the instrument response function.
    
    :param array x: x-values.
    :param float y0: Offset.
    :param float s: Sigma / width of Gaussian.
    :param float x0: (Time) Offset along x-axis.
    :param variable p: Coefficients for exponentials. For every exponential provide 
    
                       - A amplitude (can be negative),
                       - dx decay constant (can be negative, which results in a growing exponential).
    :returns: Convoluted exponentials :math:`f(x)`.
    """
    out = y0
    for i in range(0,int(len(p)),2):
        out += p[i] * np.exp(2.0 * x0 / p[i+1] + 0.25 * s**2 / p[i+1]**2 - x / p[i+1]) * (1.0 - erf(0.5 * ( 4.0 * p[i+1] * x0 + s**2 - 2.0 * p[i+1] * x )/(p[i+1]*s)))
    return out
    
# ====================================================================================================================
# Lorentzians
# ====================================================================================================================

# single lorentzian w/o baseline
def lor(x, A, x0, dx):
    """Single Lorentzian without baseline / offset.
    
    :param array x: x-values.
    :param float A: Amplitude.
    :param float x0: Center.
    :param float dx: FWHM.
    :returns: :math:`f(x) = \\frac{A}{1 + 4 (x - x_0)^2 / \\Delta x^2}`.
    """
    return A / (1.0 + ((x - x0)*(2.0 / dx))**2)

# sum of lorentzians w/o baseline
def lorentzians(x, *p):
    """Sum over variable number of Lorentzians without baseline.
    
    :param array x: x-values.
    :param variable p: Coefficients for Lorentzians. For each Lorentzian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Lorentzians :math:`f(x) = \\sum_i L_i(x)`.
    
    .. seealso: :py:func:`lor` for used mathematical notation.
    """
    y = np.zeros(len(x))
    N = int(len(p)/3)
    for i in range(N):
        y += lor(x, *p[3*i:3*i+3])
    return y

# same with const. baseline
def lorentzians_const(x, a, *p):
    """Sum over variable number of Lorentzians on a constant background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param variable p: Coefficients for Lorentzians. For each Lorentzian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Lorentzians :math:`f(x) = \\sum_i L_i(x) + a`.
    
    .. seealso: :py:func:`lorentzians` for details.
    """
    return const(x, a) + lorentzians(x, *p)

# same with linear baseline
def lorentzians_line(x, a, b, *p):
    """Sum over variable number of Lorentzians on a linear background.
    
    :param array x: x-values.
    :param float a: y-intercept.
    :param float b: Slope.
    :param variable p: Coefficients for Lorentzians. For each Lorentzian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Lorentzians :math:`f(x) = \\sum_i L_i(x) + a + b x`.
    
    .. seealso: :py:func:`lorentzians` for details.
    """
    return line(x, a, b) + lorentzians(x, *p)
    
# same with parabolic baseline
def lorentzians_parabola(x, a, b, c, *p):
    """Sum over variable number of Lorentzians on a quadratic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param variable p: Coefficients for Lorentzians. For each Lorentzian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Lorentzians :math:`f(x) = \\sum_i L_i(x) + a + b x + c x^2`.
    
    .. seealso: :py:func:`lorentzians` for details.
    """
    return parabola(x, a, b, c) + lorentzians(x, *p)
    
# same with cubic baseline
def lorentzians_cubic(x, a, b, c, d, *p):
    """Sum over variable number of Lorentzians on a cubic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :param variable p: Coefficients for Lorentzians. For each Lorentzian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Lorentzians :math:`f(x) = \\sum_i L_i(x) + a + b x + c x^2 + d x^3`.
    
    .. seealso: :py:func:`lorentzians` for details.
    """
    return cubic(x, a, b, c, d) + lorentzians(x, *p)

# same with quartic baseline
def lorentzians_quartic(x, a, b, c, d, e, *p):
    """Sum over variable number of Lorentzians on a quartic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :param float e: Quartic term.
    :param variable p: Coefficients for Lorentzians. For each Lorentzian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Lorentzians :math:`f(x) = \\sum_i L_i(x) + a + b x + c x^2 + d x^3 + e x^4`.
    
    .. seealso: :py:func:`lorentzians` for details.
    """
    return quartic(x, a, b, c, d, e) + lorentzians(x, *p)

# ====================================================================================================================
# dispersive Lorentzians (see Hoffman et al, J Phys Chem A 2014)
# ====================================================================================================================

# single dispersive lorentzian w/o baseline; A is intensity of 'real' part, B is intensity of dispersive part
def displor(x, A, B, x0, dx):
    """A *dispersive* Lorentzian without baseline / offset.
    
    :param array x: x-values.
    :param float A: Intensity of *real* part.
    :param float B: Intensity of *dispersive* part.
    :param float x0: Center along x-axis.
    :param float dx: FWHM.
    :returns: :math:`f(x) = \\frac{A + 2 B (x - x_0) / \\Delta x}{1.0 + 4 (x - x0)^2 / \\Delta x^2}`.
    
    .. seealso: Hoffman et al., *J. Phys. Chem. A* **118**, 4955 (2014) on how to use dispersive Lorentzians for analysis of dispersive lineshapes in FSRS.
    """
    return (A + B * (x - x0)*(2.0 / dx)) / (1.0 + ((x - x0)*(2.0 / dx))**2)

# sum of lorentzians w/o baseline
def displorentzians(x, *p):
    """A sum over a variable number of dispersive Lorentzians without baseline.
    
    :param array x: x-values.
    :param variable p: Coefficients for dispersive Lorentzians. For each Lorentzian provide
    
                       - A amplitude of real part,
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Lorentzians :math:`f(x) = \\sum_i L_i(x)`.
    
    .. seealso: :py:func:`displor` for mathematical formulation.
    """
    y = np.zeros(len(x))
    N = int(len(p)/4)
    for i in range(N):
        y += displor(x, *p[4*i:4*i+4])
    return y

# same with const. baseline
def displorentzians_const(x, a, *p):
    """A sum over a variable number of dispersive Lorentzians on a constant background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param variable p: Coefficients for dispersive Lorentzians. For each Lorentzian provide
    
                       - A amplitude of real part,                       
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Lorentzians :math:`f(x) = \\sum_i L_i(x) + a`.
    
    .. seealso: :py:func:`displorentzians` for details.
    """
    return const(x, a) + displorentzians(x, *p)

# same with linear baseline
def displorentzians_line(x, a, b, *p):
    """A sum over a variable number of dispersive Lorentzians on a linear background.
    
    :param array x: x-values.
    :param float a: y-intercept.
    :param float b: Slope.
    :param variable p: Coefficients for dispersive Lorentzians. For each Lorentzian provide
    
                       - A amplitude of real part,
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Lorentzians :math:`f(x) = \\sum_i L_i(x) + a + b x`.
    
    .. seealso: :py:func:`displorentzians` for details.
    """
    return line(x, a, b) + displorentzians(x, *p)
    
# same with parabolic baseline
def displorentzians_parabola(x, a, b, c, *p):
    """A sum over a variable number of dispersive Lorentzians on a quadratic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param variable p: Coefficients for dispersive Lorentzians. For each Lorentzian provide
    
                       - A amplitude of real part,
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Lorentzians :math:`f(x) = \\sum_i L_i(x) + a + b x + c x^2`.
    
    .. seealso: :py:func:`displorentzians` for details.
    """
    return parabola(x, a, b, c) + displorentzians(x, *p)
    
# same with cubic baseline
def displorentzians_cubic(x, a, b, c, d, *p):
    """A sum over a variable number of dispersive Lorentzians on a cubic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :param variable p: Coefficients for dispersive Lorentzians. For each Lorentzian provide
    
                       - A amplitude of real part,
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Lorentzians :math:`f(x) = \\sum_i L_i(x) + a + b x + c x^2 + d x^3`.
    
    .. seealso: :py:func:`displorentzians` for details.
    """
    return cubic(x, a, b, c, d) + displorentzians(x, *p)

# same with quartic baseline
def displorentzians_quartic(x, a, b, c, d, e, *p):
    """A sum over a variable number of dispersive Lorentzians on a quartic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :param float e: Quartic term.
    :param variable p: Coefficients for dispersive Lorentzians. For each Lorentzian provide
    
                       - A amplitude of real part,
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Lorentzians :math:`f(x) = \\sum_i L_i(x) + a + b x + c x^2 + d x^3 + e x^4`.
    
    .. seealso: :py:func:`displorentzians` for details.
    """
    return quartic(x, a, b, c, d, e) + displorentzians(x, *p)

# ====================================================================================================================
# Gaussians
# ====================================================================================================================

# single gaussian w/o baseline
def gauss(x, A, x0, dx):
    """A single Gaussian without baseline / offset.
    
    :param array x: x-values.
    :param float A: Amplitude.
    :param float x0: Center.
    :param float dx: FWHM.
    :returns: :math:`f(x) = A e^{-4 \\ln 2 (x - x_0)^2 / \\Delta x^2}`.
    """
    return A * np.power(16.0, -(x-x0)**2 / dx**2)

# sum of gaussians w/o baseline
def gaussians(x, *p):
    """Sum over a variable number of Gaussians without baseline.
    
    :param array x: x-values.
    :param variable p: Coefficients for Gaussians. For each Gaussian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Gaussians :math:`f(x) = \\sum_i G_i(x)`.
    
    .. seealso:: :py:func:`gauss` for the mathematical formulation.
    """
    y = np.zeros(len(x))
    N = int(len(p)/3)
    for i in range(N):
        y += gauss(x, *p[3*i:3*i+3])
    return y
    
# same with const. baseline
def gaussians_const(x, a, *p):
    """Sum over a variable number of Gaussians on a constant background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param variable p: Coefficients for Gaussians. For each Gaussian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Gaussians :math:`f(x) = \\sum_i G_i(x) + a`.
    
    .. seealso:: :py:func:`gaussians` for details.
    """
    return const(x, a) + gaussians(x, *p)

# same with linear baseline
def gaussians_line(x, a, b, *p):
    """Sum over a variable number of Gaussians on a linear background.
    
    :param array x: x-values.
    :param float a: y-intercept.
    :param float b: Slope.
    :param variable p: Coefficients for Gaussians. For each Gaussian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Gaussians :math:`f(x) = \\sum_i G_i(x) + a + b x`.
    
    .. seealso:: :py:func:`gaussians` for details.
    """
    return line(x, a, b) + gaussians(x, *p)

# same with parabolic baseline
def gaussians_parabola(x, a, b, c, *p):
    """Sum over a variable number of Gaussians on a quadratic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param variable p: Coefficients for Gaussians. For each Gaussian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Gaussians :math:`f(x) = \\sum_i G_i(x) + a + b x + c x^2`.
    
    .. seealso:: :py:func:`gaussians` for details.
    """
    return parabola(x, a, b, c) + gaussians(x, *p)

# same with cubic baseline
def gaussians_cubic(x, a, b, c, d, *p):
    """Sum over a variable number of Gaussians on a cubic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :param variable p: Coefficients for Gaussians. For each Gaussian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Gaussians :math:`f(x) = \\sum_i G_i(x) + a + b x + c x^2 + d x^3`.
    
    .. seealso:: :py:func:`gaussians` for details.
    """
    return cubic(x, a, b, c, d) + gaussians(x, *p)

# same with quartic baseline
def gaussians_quartic(x, a, b, c, d, e, *p):
    """Sum over a variable number of Gaussians on a quartic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :param float e: Quartic term.
    :param variable p: Coefficients for Gaussians. For each Gaussian provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over Gaussians :math:`f(x) = \\sum_i G_i(x) + a + b x + c x^2 + d x^3 + e x^4`.
    
    .. seealso:: :py:func:`gaussians` for details.
    """
    return quartic(x, a, b, c, d, e) + gaussians(x, *p)

# ====================================================================================================================
# dispersive Gaussians (similar to Hoffman et al, J Phys Chem A 2014)
# ====================================================================================================================    
    
def dispgauss(x, A, B, x0, dx):
    """A single *dispersive* Gaussian without baseline / offset.
    
    :param array x: x-values.
    :param float A: Amplitude of *real* part.
    :param float B: Amplitude of *dispersive* part.
    :param float x0: Center.
    :param float dx: FWHM.
    :returns: :math:`f(x) = (A + \\frac{2 \\sqrt{\\ln(2) \\pi} (x - x_0)}{\\Delta x} B) e^{-4 \\ln 2 (x - x_0)^2 / \\Delta x^2}`.
    
    .. note:: This dispersive Gaussian is defined in an analogous manner as the dispersive Lorentzian :py:func:`displor` with an extra factor :math:`\\sqrt{\\ln(2) \\pi}`, which makes the total area under the dispersive and real parts identical. 
    
    .. seealso:: See Hoffman et al., *J. Phys. Chem. A* **118**, 4955 (2014) on how to use dispersive Lorentzians for analysis of dispersive lineshapes in FSRS.
    """
    return (A + 2.0 * B * (x-x0) / dx) * np.power(16.0, -(x-x0)**2 / dx**2)
    

# sum of dispgaussians w/o baseline
def dispgaussians(x, *p):
    """Sum over a variable number of dispersive Gaussians without baseline.
    
    :param array x: x-values.
    :param variable p: Coefficients for dispersive Gaussians. For each Gaussian provide
    
                       - A amplitude of real part,
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Gaussians :math:`f(x) = \\sum_i G_i(x)`.
    
    .. seealso:: :py:func:`dispgauss` for the mathematical formulation.
    """
    y = np.zeros(len(x))
    N = int(len(p)/3)
    for i in range(N):
        y += dispgauss(x, *p[3*i:3*i+3])
    return y
    
# same with const. baseline
def dispgaussians_const(x, a, *p):
    """Sum over a variable number of dispersive Gaussians on a constant background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param variable p: Coefficients for dispersive Gaussians. For each Gaussian provide
    
                       - A amplitude of real part,
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Gaussians :math:`f(x) = \\sum_i G_i(x) + a`.
    
    .. seealso:: :py:func:`dispgaussians` for details.
    """
    return const(x, a) + dispgaussians(x, *p)

# same with linear baseline
def dispgaussians_line(x, a, b, *p):
    """Sum over a variable number of dispersive Gaussians on a linear background.
    
    :param array x: x-values.
    :param float a: y-intercept.
    :param float b: Slope.
    :param variable p: Coefficients for dispersive Gaussians. For each Gaussian provide
    
                       - A amplitude of real part,
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Gaussians :math:`f(x) = \\sum_i G_i(x) + a + b x`.
    
    .. seealso:: :py:func:`dispgaussians` for details.
    """
    return line(x, a, b) + dispgaussians(x, *p)

# same with parabolic baseline
def dispgaussians_parabola(x, a, b, c, *p):
    """Sum over a variable number of dispersive Gaussians on a quadratic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param variable p: Coefficients for dispersive Gaussians. For each Gaussian provide
    
                       - A amplitude of real part,
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Gaussians :math:`f(x) = \\sum_i G_i(x) + a + b x + c x^2`.
    
    .. seealso:: :py:func:`dispgaussians` for details.
    """
    return parabola(x, a, b, c) + dispgaussians(x, *p)

# same with cubic baseline
def dispgaussians_cubic(x, a, b, c, d, *p):
    """Sum over a variable number of dispersive Gaussians on a cubic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :param variable p: Coefficients for dispersive Gaussians. For each Gaussian provide
    
                       - A amplitude of real part,
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Gaussians :math:`f(x) = \\sum_i G_i(x) + a + b x + c x^2 + d x^3`.
    
    .. seealso:: :py:func:`dispgaussians` for details.
    """
    return cubic(x, a, b, c, d) + dispgaussians(x, *p)

# same with quartic baseline
def dispgaussians_quartic(x, a, b, c, d, e, *p):
    """Sum over a variable number of dispersive Gaussians on a quartic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :param float e: Quartic term.
    :param variable p: Coefficients for dispersive Gaussians. For each Gaussian provide
    
                       - A amplitude of real part,
                       - B amplitude of dispersive part,
                       - x0 center,
                       - dx FWHM.
    :returns: Sum over dispersive Gaussians :math:`f(x) = \\sum_i G_i(x) + a + b x + c x^2 + d x^3 + e x^4`.
    
    .. seealso:: :py:func:`dispgaussians` for details.
    """
    return quartic(x, a, b, c, d, e) + dispgaussians(x, *p)

    
# ====================================================================================================================
# Voigts
# ====================================================================================================================

# single voigt line (= lorentzian + gaussian) w/o baseline
# alpha = 0 = gaussian, 1 = lorentzian
def voigt(x, A, x0, dx, a):
    """Approximation to a Voigt profile based on a weighted sum of a Gaussian and a Lorentzian with the same parameters.
    
    :param array x: x-values.
    :param float A: Amplitude.
    :param float x0: Center.
    :param float dx: FWHM.
    :param float a: Relative weight of Lorentzian vs. Gaussian (0 = Gaussian, 1 = Lorentzian).
    :returns: :math:`f(x) = \\alpha L(x) + (1 - \\alpha) G(x)`.
    
    .. seealso: Refer to :py:func:`lor` and :py:func:`gauss` for the mathematical implementation of Lorentzian and Gaussian.
    """    
    alpha = np.minimum(1, np.maximum(0, a))
    return alpha * lor(x, A, x0, dx) + (1.0 - alpha) * gauss(x, A, x0, dx)

# sum of voigt lines w/o baseline
def voigts(x, *p):
    """Sum over variable number of approximate Voigt profiles without baseline / offset.
    
    :param array x: x-values.
    :param variable p: Coefficients for Voigt profiles. For each Voigt profile provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM,
                       - a relative weight of Lorentzian vs. Gaussian.
    :returns: Sum over Voigt profiles :math:`f(x) = \\sum_i V_i(x)`.
    
    .. seealso:: :py:func:`voigt` for mathematical implementation of Voigt lineshape.
    """
    y = np.zeros(len(x))
    N = int(len(p)/4)
    for i in range(N):
        y += voigt(x, *p[4*i:4*i+4])
    return y

# same with const. baseline
def voigts_const(x, a, *p):
    """Sum over variable number of approximate Voigt profiles on a constant background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param variable p: Coefficients for Voigt profiles. For each Voigt profile provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM,
                       - a relative weight of Lorentzian vs. Gaussian.
    :returns: Sum over Voigt profiles :math:`f(x) = \\sum_i V_i(x) + a`.
    
    .. seealso:: :py:func:`voigts` for more details.
    """
    return const(x, a) + voigts(x, *p)

# same with linear baseline
def voigts_line(x, a, b, *p):
    """Sum over variable number of approximate Voigt profiles on a linear background.
    
    :param array x: x-values.
    :param float a: y-intercept.
    :param float b: Slope.
    :param variable p: Coefficients for Voigt profiles. For each Voigt profile provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM,
                       - a relative weight of Lorentzian vs. Gaussian.
    :returns: Sum over Voigt profiles :math:`f(x) = \\sum_i V_i(x) + a + b x`.
    
    .. seealso:: :py:func:`voigts` for more details.
    """
    return line(x, a, b) + voigts(x, *p)
    
# same with parabolic baseline
def voigts_parabola(x, a, b, c, *p):
    """Sum over variable number of approximate Voigt profiles on a quadratic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param variable p: Coefficients for Voigt profiles. For each Voigt profile provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM,
                       - a relative weight of Lorentzian vs. Gaussian.
    :returns: Sum over Voigt profiles :math:`f(x) = \\sum_i V_i(x) + a + b x + c x^2`.
    
    .. seealso:: :py:func:`voigts` for more details.
    """
    return parabola(x, a, b, c) + voigts(x, *p)
    
# same with cubic baseline
def voigts_cubic(x, a, b, c, d, *p):
    """Sum over variable number of approximate Voigt profiles on a cubic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :param variable p: Coefficients for Voigt profiles. For each Voigt profile provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM,
                       - a relative weight of Lorentzian vs. Gaussian.
    :returns: Sum over Voigt profiles :math:`f(x) = \\sum_i V_i(x) + a + b x + c x^2 + d x^3`.
    
    .. seealso:: :py:func:`voigts` for more details.
    """   
    return cubic(x, a, b, c, d) + voigts(x, *p)

# same with quartic baseline
def voigts_quartic(x, a, b, c, d, e, *p):
    """Sum over variable number of approximate Voigt profiles on a quartic background.
    
    :param array x: x-values.
    :param float a: Offset.
    :param float b: Linear term.
    :param float c: Quadratic term.
    :param float d: Cubic term.
    :param float e: Quartic term.
    :param variable p: Coefficients for Voigt profiles. For each Voigt profile provide
    
                       - A amplitude,
                       - x0 center,
                       - dx FWHM,
                       - a relative weight of Lorentzian vs. Gaussian.
    :returns: Sum over Voigt profiles :math:`f(x) = \\sum_i V_i(x) + a + b x + c x^2 + d x^3 + e x^4`.
    
    .. seealso:: :py:func:`voigts` for more details.
    """   
    return quartic(x, a, b, c, d, e) + voigts(x, *p)
    
# ====================================================================================================================
# beam waist fitting
# ====================================================================================================================

def waist(z, A, y0, z0, w0):
    """Use this function to fit the integrated intensity profile obtained by translating a knife edge through the waist of a laser beam. Mathematically, this is an error function.
    
    :param array z: Position of knife edge perpendicular to the beam axis.
    :param float A: Total integrated intensity.
    :param float y0: Offset / constant background.
    :param float z0: Center of beam on axis perpendicular to beam axis.
    :param float w0: Beam waist defined as :math:`1 / e^2` - **radius** of intensity.
    :returns: Intensity profile obtained by knife-edge method: :math:`f(x) = y_0 + \\frac{A}{2} [ 1  + \\mathrm{erf}( \\sqrt{2} (z - z_0) / w_0 ) ]`.
    """
    return y0 + A / 2 * (1 + erf(np.sqrt(2) * (z - z0) / w0))
    
# extract beam waist from razor blade measurement of transmitted beam power (y) and step size dz
def fit_waist(y, dz):
    """Convenience function for extracting the beam waist from a knife-edge intensity profile using :py:func:`waist` and regularly spaced points.
    
    :param array y: Transmitted power.
    :param float dz: Change in z-position of knife edge between subsequent power readings.
    :returns: Fit results
    
              - A amplitude, total power,
              - y0 offset, background light level,
              - z0 center of beam,
              - w0 beam waist defined as :math:`1 / e^2` - **radius** of intensity.
    """
    # make z axis
    z = np.arange(len(y)) * dz  
    #fit
    popt, _ = curve_fit(waist, z, y, [np.amax(y)-np.amin(y), np.amin(y), (z[-1]+z[0])/2, (z[-1]-z[0])/2])   
    return popt
    
# fit Gaussian beam waists as function of on-axis position x
# x0 is offset along x axis, 
# xR is Rayleigh length
# w0 is spot size in focus
def rayleigh(x, x0, xR, w0):
    """Waist size of a Gaussian laser beam as function of position along axis.
    
    :param array x: x-position along axis.
    :param float x0: x-position of beam waist / focus.
    :param float xR: Rayleigh length.
    :param float w0: Beam waist / spot size in focus defined as :math:`1 / e^2` - **radius** of intensity.
    :returns: :math:`f(x) = w_0 \\sqrt{ 1 + (x - x_0)^2 / x_R^2 }`.
    """
    return w0 * np.sqrt(1.0 + (x-x0)**2 / xR**2)
    
# ====================================================================================================================
# saturation curve
# ====================================================================================================================
    
# A is saturated value
# x2 is value at which function reaches A/2
def saturation(x, A, x2):
    """Saturation curve from 0 to A.
    
    :param array x: x-values.
    :param float A: Saturation value (may be negative).
    :param float x2: x-value for which function reaches A/2.
    :returns: :math:`f(x) = \\frac{A x}{x + x_2}`.
    """
    return A * x / (x + x2)

# ====================================================================================================================
# step functions on linear podest
# ====================================================================================================================

def heaviside(x, x0):
    """Heaviside step function.
    
    :param array x: x-values.
    :param float x0: x-offset.
    :returns: - 0.0 for x < x0,
              - 0.5 for x == x0,
              - 1.0 for x > x0.
    """
    return 0.5 * (np.sign(x - x0) + 1.0)

# step function at x0 between lower value y0 and upper value y1; baseline slope is a
def step(x, x0, y0, y1, a):
    """Sharp step function on a linear background.
    
    :param array x: x-values.
    :param float x0: Center of step.
    :param float y0: Function value for x < x0.
    :param float y1: Function value for x > x0.
    :param float a: Slope of baseline.
    :returns: :math:`f(x) = (y_1 - y_0) \\mathrm{Heaviside}(x0 - x) + y_0 + a (x - x_0)`.
    """
    return (y1 - y0) * 0.5 * (np.sign(x0 - x) + 1.0) + y0 + a * (x - x0)

# same as above only smooth step with "smoothness" parameter k; if k >> 1/x, step is sharp, otherwise its smooth
def smoothstep(x, x0, y0, y1, a, k):
    """Smooth step function on a linear background.
    
    :param array x: x-values.
    :param float x0: Center of step.
    :param float y0: Function value for x < x0.
    :param float y1: Function value for x > x0.
    :param float a: Slope of baseline.
    :param float k: Sharpness of step (k << 1 / x0: smooth, k >> 1 / x0: sharp).
    :returns: :math:`f(x) = \\frac{y_1 - y_0}{1 + e^{-2 k (x_0 - x)}} + y_0 + a (x - x_0)`.
    """
    return (y1 - y0) / (1.0 + np.exp(-2.0 * k * (x0-x))) + y0 + a * (x - x0)
    
# ====================================================================================================================
# trigonometric
# ====================================================================================================================
    
# general, non-decaying cosine with constant offset
def cosine(x, a, b, c, d = 0.0):
    """A cosine with constant offset.
    
    :param array x: x-values.
    :param float a: Amplitude.
    :param float b: Frequency (rad).
    :param float c: Phase (rad).
    :param float d: Offset.
    :returns: :math:`f(x) = a \\cos(2 \\pi b x + c) + d`.
    """
    return a * np.cos(2.0 * np.pi * b * x + c) + d
    
# general, non-decaying cosine-squared with constant offset
def cosine2(x, a, b, c, d = 0.0):
    """A cosine squared.
    
    :param array x: x-values.
    :param float a: Amplitude.
    :param float b: Frequency (rad).
    :param float c: Phase (rad).
    :param float d: Offset.
    :returns: :math:`f(x) = a \\cos^2(2 \\pi b x + c) + d`.
    
    .. note:: This is not the same as (:py:func:`cosine`)**2!
    """
    return a * np.cos(2.0 * np.pi * b * x + c)**2 + d
    
# ====================================================================================================================
# LPSVD fitting
# ====================================================================================================================
    
# sum of exponentially decaying sinusoids
# parameters per sinusoid are ck (amplitude), bk (exp decay constant), f (oscillation frequency), phi (phase)
def sinusoids(x, *p):
    """A sum over a variable number of decaying sinusoids.
    
    :param array x: x-values.
    :param variable p: Coefficients of sinusoids. For each term provide
    
                       - a amplitude,
                       - b exponential decay rate ( = 1 / decay time),
                       - f0 oscillation frequency (rad),
                       - phi phase (rad).
    :returns: :math:`f(x) = \\sum_i a_i e^{-x b_i} \\cos( 2 \\pi f_0 x + \\phi )`.
    
    .. seealso:: This function is intended for use with :py:func:`lpsvd`.    
    """
    y = np.zeros(x.shape)
    K = int(len(p)/4)
    for i in range(K):
        y = y + p[i*4+0] * np.exp(-x * p[i*4+1]) * np.cos( 2 * np.pi * p[i*4+2] * x + p[i*4+3] )
    return y
    
# returns reconstructed frequency domain representation (from analytic Fourier transform of one sided decaying sinusoid)
# parameters are the same as above
# f is frequency axis in units corresponding to time axis
def sinusoidsFT(f, *p):
    """Returns reconstructed frequency domain representation from analytic Fourier transform of one sided decaying sinusoid for a variable number of sinusoids.
    
    :param array f: frequency axis in units corresponding to the time axis.
    :param variable p: Coefficients of sinusoids. For each term provide
    
                       - a amplitude,
                       - b exponential decay rate ( = 1 / decay time),
                       - f0 oscillation frequency (rad),
                       - phi phase (rad).
    :returns: :math:`F(f) = \\sum_i \\left| a_i \\frac{2 j \\pi f + b_i}{ (2 j \\pi f + b_i - 2 j \\pi f_{0,i}) ( 2 j \\pi f + b_i + 2 j \\pi f_{0, i} ) } \\right|`, where *j* is the imaginary unit.
    
    .. note:: The phase of the sinusoids is actually not used, however, it is included in the parameter list to provide consistency with :py:func:`sinusoids`.
    
    .. seealso:: This function is intended for use with :py:func:`lpsvd`.    
    """
    y = 0.0
    K = int(len(p)/4)
    for i in range(K):
        y = y + np.absolute(p[i*4+0] * (2 * 1j * np.pi * f + p[i*4+1]) / ((2 * 1j * np.pi * f + p[i*4+1] - 2 * 1j * np.pi * p[i*4+2])*(2 * 1j * np.pi * f + p[i*4+1] + 2 * 1j * np.pi * p[i*4+2])))
    return y
    
# LPSVD fitting function
# extract parameters for arbitrarily many exponentially decaying sinusoids from time domain data (x,y) without prior
# knowledge of parameters
# based on IEEE Trans. ASSP, ASSP-30, 833 (1982)
# adapted from Matlab code by Greg Reynolds, 2006 (from mathworks.com)
#
# x is time axis
# y is data
# eta is ratio of number of points in sample to number of points used for LP
# refine = True: run a standard least squares fitting to optimize the LPSVD parameters

# returns the parameter set in the form (A, dx, f, phi) per sinusoid
def lpsvd(x, Y, eta = 0.75, refine=False):
    """LPSVD fitting function based on Kumaresan and Tufts, *IEEE Transactions on Acoustics, Speech and Signal Processing* **ASSP-30**, 833 (1982) and the Matlab code by Greg Reynolds, 2006 (from mathworks.com).
    
    Use this function to extract the parameters of arbitrarily many exponentially decaying sinusoids from (noisy) time domain data (x,y) without prior knowledge of these parameters nor the exact number of oscillation frequencies.

    :param array x: x-values / time axis.
    :param array Y: Input data (same shape as x).
    :param float eta: Ratio of number of points in sample to number of points used for LP (< 1).
    :param bool refine: If True, run a standard least squares fitting after LPSVD to further refine the the LPSVD parameters.
    :returns: A set of coefficients for each sinusoid. The number of sinusoids is `len(res) / 4`. For each sinusoid the function returns
    
            - a amplitude,
            - b exponential decay rate,
            - f frequency (rad),
            - phi phase (rad).
            
    .. note:: The resulting array can be directly used as input to :py:func:`sinusoids` and :py:func:`sinusoidsFT`.
    
    .. seealso:: :py:func:`lpsvd2`.
    """
    dt = x[1] - x[0]
    y = Y - np.mean(Y)
    
    # extract size of data array and number of points used for linear prediction
    N = len(y)
    L = int(eta*N)

    # construct matrix and signal vector for primary linear prediction step
    A = np.conj(hankel(y[1:N+1-L], y[N-L:N]))
    h = y[0:N-L]
    
    # solve linear system using pinv (which comprises the svd functionality in a single step)
    b = np.dot(np.linalg.pinv(A), -h)

    # the solutions are given by the roots to the Lth order polynom
    qr = np.roots(np.append([1], b))

    # retain only those roots, which lie outside the unit circle (backward prediction, all others lie within)
    quse = qr[np.where(np.absolute(qr) >= 1.0)]
    q = -np.conj(np.log(quse))
    q = q[np.where(np.imag(q) > 0)] # retain only positive frequencies -> double amplitude afterwards
    
    # extract damping constants and frequencies
    dampings = -np.real(q) / dt
    frequencies = np.imag(q) / (2*np.pi) / dt

    # get number of sinusoids
    K = len(dampings)

    # use second linear prediction step to get amplitudes and phases
    # basis = np.exp(np.outer(x, (-dampings + 1j * 2.0 * np.pi * frequencies)))
    # ahat = np.dot(np.linalg.pinv(basis[0:N,:]), y)
    basis = np.exp(np.outer(x, q / dt))
    ahat = np.dot(np.linalg.pinv(basis), y)
    amplitudes = 2 * np.absolute(ahat)
    phases = np.angle(ahat)
    
    # construct resulting parameter array
    popt = np.array([amplitudes, dampings, frequencies, phases]).transpose().ravel()

    if(refine == True):
        popt, _ = curve_fit(sinusoids, x, y, popt, maxfev = 100000)
    
    return popt
    
# estimate the number of relevant singular values
# adapted from Matlab code by Greg Reynolds, 2006 (from mathworks.com)
# used by lpsvd and lpsvd2, not for direct use
def _estimate_model_order(s, N, Lp):
    mdl = []
    L = len(s)
    for k in range(L):
        val = -N * np.sum(np.log(s[k:L])) 
        val += N * (L-k) * np.log( np.sum(s[k:L])/float(L-k) ) 
        val += k * (2 * L - k) * np.log(N) / 2
        mdl.append(val)
    M = np.argmin(mdl)
    return M
    
# same as above; adopted from Hoffman's version, based on J. Mag. Res. 61, 465 (1985)
# seems to be more robust for noisy data
# M = model order = number of relevant singular values; if none is given, estimate automatically
# removeBias = True: subtract the mean value of noise-related singular values before stripping
def lpsvd2(x, Y, eta = 0.75, M = None, removeBias = False, refine = False):
    """Variation of LPSVD fitting function based on Barkhujisen et al, *Journal of Magnetic Resonance* **61**, 465 (1985), which seems to be more robust for noisy data.
    
    Use this function to extract the parameters of arbitrarily many exponentially decaying sinusoids from (noisy) time domain data (x,y) without prior knowledge of these parameters nor the exact number of oscillation frequencies.

    :param array x: x-values / time axis.
    :param array Y: Input data (same shape as x).
    :param float eta: Ratio of number of points in sample to number of points used for LP (< 1).
    :param int M: Model order = number of relevant singular values. If None, estimate automatically (default).
    :param bool removeBias: If True, subtract the mean value of noise-related singular values before stripping.
    :param bool refine: If True, run a standard least squares fitting after LPSVD to further refine the the LPSVD parameters.
    :returns: A set of coefficients for each sinusoid. The number of sinusoids is `len(res) / 4`. For each sinusoid the function returns
    
            - a amplitude,
            - b exponential decay rate,
            - f frequency (rad),
            - phi phase (rad).
            
    .. note:: The resulting array can be directly used as input to :py:func:`sinusoids` and :py:func:`sinusoidsFT`.
    
    .. seealso:: :py:func:`lpsvd`.
    """
    dt = x[1] - x[0]
    y = Y - np.mean(Y)
    
    # extract size of data array and number of points used for linear prediction
    if(eta > 0.75):
        eta = 0.75
    N = len(y)
    L = int(eta*N)

    # construct matrix and signal vector for primary linear prediction step
    A = np.conj(hankel(y[1:N+1-L], y[N-L:N]))
    h = y[0:N-L]
    
    # solve linear system using pinv (which comprises the svd functionality in a single step)
    U, s, V = np.linalg.svd(A)
    if(M == None):
        M = _estimate_model_order(s, N, L) + 8
    if(M > len(s)):
        M = len(s)
    print("Estimated model order:", M)
    
    # remove 'bias'
    if(removeBias and M < len(s)-1):
        s = s - np.mean(s[M:])
    
    # invert and create diagonal matrix
    Si = np.diag(1.0/s[0:M])

    # redimension the other two matrices U, V
    U = U[:,0:M]
    V = V[0:M,:]

    # calculate the LP coefficients
    b = -1 * np.dot(np.transpose(np.conj(V)), np.dot(Si, np.dot(np.transpose(np.conj(U)), h)))
    
    # the solutions are given by the roots to the Lth order polynom
    qr = np.roots(np.append([1], b))

    # retain only those roots, which lie outside the unit circle (backward prediction, all others lie within)
    quse = qr[np.where(np.absolute(qr) >= 1.0)]
    
    if(len(quse) == 0):
        print("Error: No roots found outside unit circle!")
        return []
        
    q = np.conj(np.log(quse))
    q = q[np.where(np.imag(q) > 0)] # only retain positive frequencies -> double amplitude!
    
    # extract damping constants and frequencies
    dampings = np.real(q) / dt
    frequencies = np.imag(q) / (2*np.pi) / dt
    
    # use second linear prediction step to get amplitudes and phases
    basis = np.exp(np.outer(x, -np.conj(q) / dt))
    ahat = np.dot(np.linalg.pinv(basis), y)
    amplitudes = 2 * np.absolute(ahat)
    phases = np.angle(ahat)
    
    # construct resulting parameter array
    popt = np.array([amplitudes, dampings, frequencies, phases]).transpose().ravel()

    if(refine == True):
        popt, _ = curve_fit(sinusoids, x, y, popt, maxfev = 100000)
    
    return popt 
    
# ====================================================================================================================
# Short Time Fourier Transform
# ===================================================================================================================
    
# data = a numpy array containing the signal to be processed
# fs = a scalar which is the sampling frequency of the data
# w = window size in units of time
# overlap_fac = percentage of overlap between consecutive windows; a value of 0.5 does not change the magnitude of the signal 
def stft(data, fs, w, overlap_fac = 0.5):
    """Short time (or moving window) Fourier transform using a Hanning window.
    Code was adapted from Kevin Nelson's code (https://github.com/KevinNJ/Projects).
        
    :param array data: 1d data to be processed.
    :param float fs: Sampling frequency of the data ( = 1 / dt ).
    :param float w: Size of the moving window in units of time.
    :param float overlap_fac: Percentage of overlap between consecutive windows. A value of 0.5 does not alter the magnitude of the signal.
    :returns: 2d spectrogram of frequency vs time.
    """    
    fft_size = np.int32(w * fs)
    hop_size = np.int32(np.floor(fft_size * (1-overlap_fac)))
    pad_end_size = fft_size          # the last segment can overlap the end of the data array by no more than one window size
    total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))
    t_max = len(data) / np.float32(fs)
     
    window = np.hanning(fft_size)  # our half cosine window
    inner_pad = np.zeros(fft_size) # the zeros which will be used to double each segment size
     
    proc = np.concatenate((data, np.zeros(pad_end_size)))              # the data to process
    result = np.empty((total_segments, fft_size), dtype=np.float32)    # space to hold the result
     
    for i in range(total_segments):                       # for each segment
        current_hop = hop_size * i                        # figure out the current segment offset
        segment = proc[current_hop:current_hop+fft_size]  # get the current segment
        windowed = segment * window                       # multiply by the half cosine function
        padded = np.append(windowed, inner_pad)           # add 0s to double the length of the data
        spectrum = np.fft.fft(padded) / fft_size          # take the Fourier Transform and scale by the number of samples
        autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
        result[i, :] = autopower[:fft_size]               # append to the results array
     
    return result
    
# ====================================================================================================================
# Lorentz Oscillator Model  - see http://de.wikipedia.org/wiki/Lorentzoszillator
# ====================================================================================================================

def lorOscRe(w, A, w0, dw):
    """Real part of a Lorentz oscillator model for the dielectric function.
    
    :param array w: Angular frequency.
    :param float A: Amplitude.
    :param float w0: Resonance frequency.
    :param float dw: Damping rate.
    :returns: :math:`f(w) = 1 + \\frac{A (\\omega_0^2 - \\omega^2)}{ (\\omega_0^2 - \\omega^2)^2 + \\Delta \\omega^2 \\omega^2 }`.
    """
    return 1.0 + A * (w0**2 - w**2) / ((w0**2 - w**2)**2 + dw**2 * w**2)

def lorOscIm(w, A, w0, dw):
    """Imaginary part of a Lorentz oscillator model for the dielectric function.
    
    :param array w: Angular frequency.
    :param float A: Amplitude.
    :param float w0: Resonance frequency.
    :param float dw: Damping rate.
    :returns: :math:`f(w) = \\frac{A \\Delta \\omega \\omega}{ (\\omega_0^2 - \\omega^2)^2 + \\Delta \\omega^2 \\omega^2 }`.
    """
    return A * dw * w / ((w0**2 - w**2)**2 + dw**2 * w**2)
        
# ====================================================================================================================
# Principal Component Analysis for deconvolution of mulitple species spectra
# see Kubista et al. Crit. Rev. Anal. Chem. 29, 1-28 (1999) and references therein
#
# matrix A is a data matrix with variable parameter in rows and wavelength / wavenumber in columns
# parameter can be concentration, temperature, time, etc
# ====================================================================================================================
    
# NIPALS - Algorithm for getting most significant principal components
# returns the first n most important target and projection vectors to reconstruct matrix A
# target vectors are eigenvectors of A A' with length being eigenvalues; they are orthogonal
# projection vectors are orthonormal
#
# np.dot(T, P) yields the approximation to the matrix A
# if n == -1: return all principal components
def nipals(A, n = -1):
    """Principal component analysis based on the NIPALS algorithm for deconvolution of spectra containing the signature of multiple species. The algorithm is based on Kubista et al., *Crit. Rev. Anal. Chem.* **29**, 1 (1999) and references therein.

    :param array A: Data matrix with variable parameter in rows and wavelength / wavenumber in columns. This parameter can be concentration, temperature, time, etc.
    :param int n: Number of principal components to return. If n = -1, return all.
    :returns: The n most important target and projection vectors that are needed to reconstruct matrix A. Target vectors T are eigenvectors of A A' with length being the eigenvalues. They are orthogonal. Projection vectors P are orthonormal. `np.dot(T, P)` yields the approximation to the matrix A.
    
    .. seealso:: :py:func:`getRdimer` and :py:func:`getR23mer`.
    """
    if n < 1:
        n = min(A.shape[0], A.shape[1])

    T = np.zeros((A.shape[0], n))
    P = np.zeros((n, A.shape[1]))
    
    for i in range(n):
        j = np.argmax(np.var(A, axis=0))        # find column with largest variance
        t1 = A[:,j]                             # copy column to vector t1
        it = 0
        while(1):
            p1 = np.dot(np.transpose(A), t1) / np.dot(t1, t1)       # calculate the projection vector
            p1 = p1 / np.sqrt(np.dot(p1, p1))       # normalize to unit length
            
            t1old = np.copy(t1)                     # copy original target vector to t1old
            t1 = np.dot(A, p1) / np.dot(p1, p1)     # project back
            
            d = np.sum(np.absolute(t1old - t1))     # check for convergence
            if(d < 1e-8 and it > 0):
                break
            it += 1
        
        T[:,i] = t1                             # copy results to matrices T and P
        P[i,:] = p1
        A = A - np.outer(t1, p1)                # form new matrix from the residual     
    
    return [T, P]
    
# calculate the transform matrix R using an estimate for the monomer <-> dimer equilibrium constant
# K is in units mol/L ( K = cm**2 / cd )
# c is list of concentrations, T is result from NIPALS and outR = True returns R, False returns residua
def getRdimer(Kp, c, T, outR = False):
    """Calculate the transform matrix R using an estimate for the monomer <-> dimer equilibrium constant Kp. See Kubista et al., *Crit. Rev. Anal. Chem.* **29**, 1 (1999) and references therein for details.

    :param float Kp: Equilibrium constant in units of mol/L ( K = cm**2 / cd ).
    :param array c: Concentration values as used in NIPALS.
    :param array T: Target vectors as obtained by NIPALS.
    :param bool outR: If True, return R; if False, return residua (default).
    :returns: Transform matrix R or residuals depending on outR.
    
    .. seealso:: :py:func:`nipals`.
    """
    Nc = 2
    
    K = np.power(10, Kp)
    
    # this is the physical constraint
    c1 = K/4.0 * (-1.0 + np.sqrt(1.0 + 8.0 * c / K))
    c2 = (c - c1) / 2
    
    # normalize
    c1 = c1 / c
    c2 = 2 * c2 / c
    
    # construct matrices and concatenated vectors for linear regression
    y = np.transpose(T).ravel()
    C = np.transpose(np.vstack((c1, c2)))               
    A = block_diag(C, C)
    
    # solve for R matrix
    R = np.linalg.lstsq(A, y)[0]
    R = np.transpose(np.reshape(R, (Nc,Nc)))

    # return results
    if(outR):
        return R
    else:
        return np.ravel(T - np.dot(C, R))
    
# calculate the transform matrix R using an estimate for the monomer <-> dimer and monomer <-> trimer equilibrium 
# constants (K2, K3)
# c is list of concentrations, T is result from NIPALS and outR = True returns R, False returns residua
def getR23mer(K, c, T, outR = False):
    """Calculate the transform matrix R using an estimate for the monomer <-> dimer and monomer <-> trimer equilibrium constants K2 and K3. See Kubista et al., *Crit. Rev. Anal. Chem.* **29**, 1 (1999) and references therein for details.

    :param tuple K: (K2, K3) Equilibrium constants for monomer <-> dimer and monomer <-> trimer equilibrium.
    :param array c: Concentration values as used in NIPALS.
    :param array T: Target vectors as obtained by NIPALS.
    :param bool outR: If True, return R; if False, return residua (default).
    :returns: Transform matrix R or residuals depending on outR.
    
    .. seealso:: :py:func:`nipals`.
    """
    Nc = 3
    
    K2, K3 = K
    K2 = np.power(10, K2)
    K3 = np.power(10, K3)
    
    # this is the physical constraint - dimer / trimer equilibrium
    c1 = np.zeros(len(c))
    for i in range(len(c)):
        c1[i] = np.amax(np.real(np.roots((3/K3, 2/K2, 1, -c[i]))))
    c2 = c1**2 / K2
    c3 = c1**3 / K3
    
    # normalize
    c1 = c1 / c 
    c2 = 2 * c2 / c
    c3 = 3 * c3 / c
    
    # construct matrices and concatenated vectors for linear regression
    y = np.transpose(T).ravel()
    C = np.transpose(np.vstack((c1, c2, c3)))               
    A = block_diag(C, C, C)
    
    # solve for R matrix
    R = np.linalg.lstsq(A, y)[0]
    R = np.transpose(np.reshape(R, (Nc,Nc)))

    # return results
    if(outR):
        return R
    else:
        return np.ravel((T - np.dot(C, R))**2)
    
# ====================================================================================================================
# dimer / monomer equilibrium fitting functions
# based on West and Pearce, J. Phys. Chem. 69, 1894 (1965)
# ====================================================================================================================

# use mixed spectrum to estimate dimer OD ratio between dimer max and monomer max by assuming dimer is symmetric
# y is mixed spectrum
# ym is 'pure' monomer spectrum
# id is index of dimer peak
# im is index of monomer peak
def estimateMonOD(y, ym, im, id, threshold = 1e-3):
    """Use mixed spectrum and pure monomer spectrum to estimate monomer optical density at position of dimer absorbance peak assuming that the dimer spectrum is symmetric. Part of dimer / monomer equilibrium fitting function based on West and Pearce, *J. Phys. Chem.* **69**, 1894 (1965).
    
    :param array y: Mixed spectrum.
    :param array ym: Pure monomer spectrum.
    :param float im: Index of monomer maximum absorbance.
    :param float id: Index of dimer maximum absorbance.
    :param float threshold: Iteration stopping threshold as relative change in monomer optical density between iterations.
    :returns: Optical density of monomer in mixed spectrum at position of dimer absorbance maximum.
    """
    # use pure monomer spectrum to get monomer OD ratio between monomer max and dimer max
    rm1 = ym[id] / ym[im]                       # dimer max
    rm2 = ym[2*id-im] / ym[im]                  # opposite wing of dimer max 

    # now start iterations
    ODd = y[id]
    rd = y[2*id-im] / ODd                       # estimated intensity ratio of dimer contribution
    ODm = y[im]                                 # first guess for monomer
    delta = 1.0
    while(abs(delta) > threshold):                   # stop when relative change drops below threshold
        delta = ODm 
        ODm = y[im] - ODd * rd                  # refine monomer OD
        ODd = y[id] - ODm * rm1                 # refine dimer OD
        rd = (y[2*id-im] - ODm * rm2) / ODd     # refine ratio
        delta -= ODm
        delta /= ODm
    
    return ODm
    
# calculate relative monomer abundance and pure dimer spectrum from a set of mixed spectra
# x is wavelength axis
# y is list of spectra for different monomer/dimer compositions that have been normalized for same total concentration
# ym is monomer spectrum
# wlm, wld is wavelength of monomer / dimer maximum
def analyseMonDim(x, y, ym0, wlm, wld):
    """Calculate relative monomer abundance and pure dimer spectrum from a set of mixed spectra. Part of dimer / monomer equilibrium fitting function based on West and Pearce, *J. Phys. Chem.* **69**, 1894 (1965).
    
    :param array x: Wavelength axis.
    :param array y: List of mixed spectra for different monomer/dimer compositions that have been normalized for same total concentration (same shape as x).
    :param array ym0: Pure monomer spectrum (same shape as x).
    :param float wlm: Wavelength of monomer absorbance peak (same units as x).
    :param float wld: Wavelength of dimer absorbance peak (same units as x).
    :returns: List of relative monomer abundances (same length as y), extrapolated dimer spectrum and extrapolated monomer spectrum (which should be identical to ym0).
    """
    # indices of monomer and dimer peaks
    im = np.argmin(np.absolute(x - wlm))
    id = np.argmin(np.absolute(x - wld))
    
    # peak monomer OD
    OD0 = ym0[im]
    
    # estimate relative monomer abundance via OD
    A = np.zeros(len(y))
    for i in range(len(y)):
        ODm = estimateMonOD(y[i], ym0, im, id)      # estimate monomer OD
        A[i] = ODm/OD0                              # calculate alpha
    # A is not necessarily between 0 and 1 if there is some concentration (e.g. solvent) change between monomer spectrum
    # and mixed spectra
    
    # extract pure dimer and monomer spectra
    z = np.transpose(y)             # [alpha, wl] -> [wl, alpha]
    yd = np.zeros(len(x))
    ym = np.zeros(len(x))
    for i in range(len(x)):
        popt, _ = curve_fit(ft.line, A, z[i])
        yd[i] = popt[0]
        ym[i] = popt[1] + popt[0]
    
    # recalculate alpha with extracted monomer spectrum
    for i in range(len(y)):
        ODm = estimateMonOD(y[i], ym, im, id)       # estimate monomer OD
        A[i] = ODm/OD0                              # calculate alpha
    
    return [A, yd, ym]