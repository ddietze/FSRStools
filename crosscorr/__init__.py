"""
.. module: FSRStools.crosscorr
   :platform: Windows
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu>

A collection of functions to process and analyze Kerr-gate cross-correlation data.
Data are analyzed by fitting each spectral slice of the cross-correlation with a
Gaussian pulse shape to extract chirp, temporal offset and FWHM.

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
import pylab as pl
import scipy.optimize as sp
import FSRStools.raman as fs


# load cross correlation data
# the first column is the time delay in fs
# the other columns correspond to the wavelengths of the
# spectrograph
# returns [time, data]
def loadXC(filename):
    """Load cross-correlation data, e.g. saved with PyFSRS.
    Cross-correlation data has the shape of a two-dimensional matrix with the
    first column being the time delay between the two pulses in femtoseconds. The
    subsequent columns represent the detected intensity at each spectral position as function of time.

    :param str filename: Filename of the cross-correlation file to load.
    :returns: Time and 2d cross-correlation data I(time, wavelength).
    """
    d = np.loadtxt(filename)
    t = np.transpose(d)[0]
    d = d[:, 1:]
    return [t, d]


def plotXC(t, d, filename=None, center=None, fwhm=None):
    """Convenience function to make a 2d plot of the cross correlation data.

    :param array t: Time axis.
    :param array d: Cross-correlation data I(time, wavelength).
    :param str filename: Filename in case the final plot should be saved (default: None).
    :param array center: Draw a line indicating the center positions as returned by :py:func:`fitXC` (default: None).
    :param array fwhm: Draw two lines indicating the FWHM of the gate pulse. fwhm is a list returned by :py:func:`fitXC` (default: None).
    """
    pl.figure()
    pl.imshow(np.flipud(d), aspect="auto", extent=[0, d.shape[1], t[0], t[-1]])

    if(center is not None):
        pl.plot(np.arange(d.shape[1]), center, color='k')
    if(fwhm is not None):
        pl.plot(np.arange(d.shape[1]), center - fwhm / 2, linewidth=0.5, color='w')
        pl.plot(np.arange(d.shape[1]), center + fwhm / 2, linewidth=0.5, color='w')

    pl.xlabel("WL (arb. units)")
    pl.ylabel("time (fs)")
    if(filename is not None):
        pl.savefig(filename)


def fitXC(t, d):
    """Analyze cross-correlation data eby fitting each spectral slice with a Gaussian
    to extract amplitude, center, and fwhm.

    :param array t: Time axis.
    :param array d: Cross-correlation data I(time, wavelength).
    :returns: List of amplitudes, center positions and widths (FWHM) of resulting Gaussians. Additionally, prints information about total chirp and average pulse width to stdout.
    """
    # define fitting functions
    g = lambda t, t0, dt, A, y0: y0 + A * 16.0**(-(t - t0)**2 / dt**2)
    l = lambda t, a, m: a + m * t

    amplitude = np.zeros(d.shape[1])
    center = np.zeros(d.shape[1])
    fwhm = np.zeros(d.shape[1])
    for i in range(d.shape[1]):
        c = d[:, i]
        popt, pcov = sp.curve_fit(g, t, c, [t[np.argmax(c)], 10, np.amax(c) - c[0], c[0]])
        amplitude[i] = popt[2]
        center[i] = popt[0]
        fwhm[i] = popt[1]

    # linear fit to the center positions to get the chirp
    popt, pcov = sp.curve_fit(l, np.arange(d.shape[1]), center, [center[0], center[1] - center[0]])

    print("FIT RESULTS:")
    print("total chirp (max-min): %.1ffs" % (np.amax(center) - np.amin(center)))
    print("total chirp (fitted): %.1ffs" % (np.absolute(popt[1]) * d.shape[1]))
    if(popt[1] < 0):
        print("chirp is positive")
    else:
        print("chirp is negative")

    print("min fwhm: %.1ffs" % np.amin(fwhm))
    print("max fwhm: %.1ffs" % np.amax(fwhm))
    print("mean fwhm: %.1ffs" % (np.sum(fwhm) / float(len(fwhm))))

    return [amplitude, center, fwhm]


def plotXCfit(a, c, f, cFilename=None, aFilename=None):
    """Convenience function to plot amplitude, center and fwhm of the cross correlation as function of index (i.e. wavelength).

    :param array a: Amplitudes of Gaussians as returned by :py:func:`fitXC`.
    :param array c: Center positions of Gaussians as returned by :py:func:`fitXC`.
    :param array f: Widths (FWHM) of Gaussians as returned by :py:func:`fitXC`.
    :param str cFilename: Filename for saving center and width plot (default: None).
    :param str aFilename: Filename for saving amplitude plot (default: None).
    """
    fig = pl.figure()
    px = fig.add_subplot(111)
    px.plot(c, color=(0, 0, 0), label="center")
    px.set_xlabel("Wavelength (arb. units)")
    px.set_ylabel("Center (fs)")
    py = px.twinx()
    py.plot(f, color=(1, 0, 0), label="FWHM")
    py.set_ylabel("FWHM (fs)")

    py.spines['right'].set_color('red')
    py.tick_params(axis='y', colors='red')
    py.yaxis.label.set_color('red')

    if(cFilename is not None):
        pl.savefig(cFilename)

    pl.figure()
    pl.plot(a)
    pl.xlabel("Wavelength (arb. units)")
    pl.ylabel("Amplitude (arb. units)")
    if(aFilename is not None):
        pl.savefig(aFilename)


# -------------------------------------------------------------------------------------------------------------------
# time zero functions - use in conjunction with cross correlation data

def shift_data(x, y, dx):
    """Smoothly translate data by an arbitrary amount along the x-axis using Fourier transformation. See :py:mod:`FSRStools.raman` for details.
    """
    return fs.shift_data(x, y, dx)


def correct_t0(t, d, c):
    """Shift all frequency columns in d along time axis t to set t0 (given in c) to 0. See :py:mod:`FSRStools.raman` for details.
    """
    return fs.correct_t0(t, d, c)
