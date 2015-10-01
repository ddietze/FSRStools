"""
.. module: FSRStools.uvvis
   :platform: Windows
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu>

Load and process UV/VIS spectra and related formats. Supports PerkinElmers old Lambda format.
Provides also some short cuts to useful functions from the :py:mod:`FSRStools.raman` module.

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

   Copyright 2015 Daniel Dietze <daniel.dietze@berkeley.edu>.
"""
import numpy as np
import pylab as pl
import glob
import struct
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import FSRStools.fitting as ft
import FSRStools.raman as fs

# ----------------
# Function shortcuts to raman module..
def cut(x, y, x0, x1):
    """Cut a slice from x and y using the values in x. See :py:mod:`FSRStools.raman` for details.
    """
    return fs.cut(x, y, x0, x1)

def at(x, y, x0):
    """Return the value of y with x-coordinate closest to x0. See :py:mod:`FSRStools.raman` for details.
    """
    return fs.at(x, y, x0)

def denoise(y):
    """Optimized smoothing of data based on Appl. Spectrosc. 62, 1160 (2008). See :py:mod:`FSRStools.raman` for details.
    """
    return fs.denoise(y)

def get_OD(wl, spectrum, wl0):
    """Returns the interpolated value of a spectrum at a wavelength wl0 using the wavelength axis wl.
    """
    return interp1d(wl, spectrum, 'cubic')(wl0)

def getOD(wl, spectrum, wl0):
    """Same as :py:func:`get_OD`.
    """
    return get_OD(wl, spectrum, wl0)

# returned binned version of x, y where d is the bin width (odd number)
# each bin is assigned the mean value of y, also returns the std over the bin width
# keeps the first and last point
def binning(x, y, d):
    """Return a binned version of x and y, where the new y values are calculated as the average over the bin width at the center of the bin.

    :param array x: x-axis.
    :param array y: Data (same shape as x).
    :param int d: Bin width in indices. Should be an odd number.
    :returns: bin centers, binned value and standard deviation over the binned values.
    """
    if d % 2 == 0:
        d += 1

    N = len(x)
    NN = int(N/d)
    xtmp = np.zeros(NN+2)
    ytmp = np.zeros(NN+2)
    etmp = np.zeros(NN+2)
    xtmp[0] = x[0]
    ytmp[0] = y[0]
    xtmp[-1] = x[-1]
    ytmp[-1] = y[-1]
    for i in range(NN):
        xtmp[i+1] = np.mean(x[d*i:d*(i+1)])
        ytmp[i+1] = np.mean(y[d*i:d*(i+1)])
        etmp[i+1] = np.std(y[d*i:d*(i+1)])
    return [xtmp, ytmp, etmp]

# returned smoothed version of x, y
# using binning to generate interpolation points
# d is the bin width (odd number)
def binsmooth(x, y, d):
    """Returned smoothed version of x, y based on binning and cubic interpolation.

    :param array x: x-axis.
    :param array y: Data (same shape as x).
    :param int d: Bin width in indices. Should be an odd number.
    :returns: Smoothed data array (same shape as x).
    """
    xtmp, ytmp, _ = binning(x, y, d)
    return interp1d(xtmp, ytmp, 'cubic')(x)

# ----------------
# convenience functions to read numeric values from binary file at given position relative to beginning of file stream
def readCHAR(stream, position):
    stream.seek(position, 0)
    return int.from_bytes(stream.read(1), byteorder = 'little', signed = True)

def readUCHAR(stream, position):
    stream.seek(position, 0)
    return int.from_bytes(stream.read(1), byteorder = 'little', signed = False)

def readINT(stream, position):
    stream.seek(position, 0)
    return int.from_bytes(stream.read(2), byteorder = 'little', signed = False)

def readUINT(stream, position):
    stream.seek(position, 0)
    return int.from_bytes(stream.read(2), byteorder = 'little', signed = True)

def readDINT(stream, position):
    stream.seek(position, 0)
    return int.from_bytes(stream.read(4), byteorder = 'little', signed = False)

def readUDINT(stream, position):
    stream.seek(position, 0)
    return int.from_bytes(stream.read(4), byteorder = 'little', signed = True)

def readLINT(stream, position):
    stream.seek(position, 0)
    return int.from_bytes(stream.read(8), byteorder = 'little', signed = False)

def readULINT(stream, position):
    stream.seek(position, 0)
    return int.from_bytes(stream.read(8), byteorder = 'little', signed = True)

def readFLOAT(stream, position):
    stream.seek(position, 0)
    return struct.unpack("<f", stream.read(4))

def readDOUBLE(stream, position):
    stream.seek(position, 0)
    return struct.unpack("<d", stream.read(8))

# actual load function
# very simple, just read out the data and create a x-axis according to xmin and xmax values
# if filename contains wildcards, return the average over all matching files
def load_sp(filename, lmin = -1, lmax = -1, smooth=True):
    """Basic support for PerkinElmer's Lambda-series UV/VIS file format (`*.sp`).
    Reads out a single spectrum and generates a x-axis according to the xmin and xmax values stored in the file.
    No support for multiple spectra per file.

    :param str filename: Filename(s) of the file(s) to load. Supports wildcards via glob. If several files match the pattern, the returned data is the average over these files.
    :param float lmin: Set minimum wavelength of returned data (same units as stored wavelength). Return all when -1 (default).
    :param float lmax: Set maximum wavelength of returned data (same units as stored wavelength). Return all when -1 (default).
    :param bool smooth: If True, apply :py:func:`denoise` to data before returning (default).
    :returns: Wavelength axis and spectrum.
    """
    files = glob.glob(filename)
    if(len(files) == 0):
        print("ERROR: No file found!")
        return np.array([[],[]])

    elif(len(files) == 1):
        fp = open(filename, "rb")
        # read header information
        numPoints = readUDINT(fp, 10)
        xMax = readUDINT(fp, 14) / 100.0
        xMin = readUDINT(fp, 18) / 100.0
        dataInterval = readUINT(fp, 22) / 100.0
        scaleFactor = readUDINT(fp, 28)
        scaleDef = readUDINT(fp, 252)

        fp.seek(512, 0)
        raw = fp.read()
        fp.close()

        # extract data
        data = np.array(struct.unpack("<" + "i" * int((len(raw) / 4)), raw), dtype='float') * scaleDef / 100.0 / scaleFactor
        data = np.nan_to_num(data)
        if smooth:
            data = denoise(data)

        # make x-axis
        xaxis = np.linspace(xMin, xMax, len(data))

        # slicing?
        if lmax < 0:
            lmax = np.amax(x) + 1

        return cut(xaxis, data, lmin, lmax)

    else:
        data = 0
        for f in files:
            x, tmp = load_sp(f, lmin, lmax, denoise)
            data = data + tmp
        data = data / float(len(files))

        return x, data

def load(filename, lmin=-1, lmax=-1, smooth=True, delim=None):
    """Loads a spectrum / set of spectra from ASCII file(s). Expects the first column to be wavelength. Text lines are ignored, i.e., this function can be used to load Ocean Optics text files, for instance.

    :param str filename: Filename(s) of the file(s) to load. Supports wildcards via glob. If several files match the pattern, the returned data is the average over these files.
    :param float lmin: Set minimum wavelength of returned data (same units as stored wavelength). Return all when -1 (default).
    :param float lmax: Set maximum wavelength of returned data (same units as stored wavelength). Return all when -1 (default).
    :param bool smooth: If True, apply :py:func:`denoise` to data before returning (default).
    :param str delim: Delimiter argument to be passed to :py:func:`numpy.loadtxt`. Set to None to use any whitespace (default).
    :returns: Wavelength axis and spectrum.
    """
    files = glob.glob(filename)
    if len(files) > 1:
        out = 0
        for f in files:
            x, y = load(f, lmin, lmax, smooth, delim)
            out = out + y
        out = out / float(len(files))
        return x, y

    elif len(files) == 1:

        try:
            # try to load data with loadtxt, if there are text lines, this will fail
            tmp = np.loadtxt(filename, unpack=True, delimiter=delim)
            x = tmp[0, :]
            y = tmp[1:, :]

        except:
            # there are text lines, so load file manually and convert line by line
            # maybe there is a faster way using numpy native functions??

            tmp = []
            fp = open(filename, "r")
            for line in fp:
                try:
                    if delim is not None:           # str.split works slightly different than np.loadtxt regarding the delimiter
                        c = line.split(delim)
                    else:
                        c = line.split()
                    c = map(lambda x: float(x), c)  # convert str to float; this function raises an exception on a text line
                    tmp.append(c)                   # append as new line
                except:
                    pass
            fp.close()

            # convert to numpy array
            tmp = np.array(tmp)
            x = tmp[0, :]
            y = tmp[1:, :]

        if y.shape[0] == 1:
            y = y[0]

        if smooth:
            y = denoise(y)

        if lmax < 0:
            lmax = np.amax(x) + 1

        return cut(x, y, lmin, lmax)

    # if no file is found, return an empty array
    print("ERROR: No file found!")
    return np.array([]), np.array([])

# ##################################################################################################################
# shortcut to fit a single peak spectrum with a single gaussian
# returns the fit parameters
# if show=True, displays the results
# if delta=True, returns parameters and errors in second list
def fit_OO_spectrum(wl, spectrum, show=False, delta=False):
    """Fit a single peaked spectrum with a gaussian and a constant offset.

    :param array wl: Wavelength axis.
    :param array spectrum: Spectrum to fit (same shape as wl).
    :param bool show: If True, show the fitted spectrum in a new figure (False by default).
    :param bool delta: If True, return the errors along with the fit parameters.
    :returns: Fit parameters [offset, amplitude, center, fwhm].
    """
    popt = [ np.amin(spectrum), np.amax(spectrum) - np.amin(spectrum), wl[np.argmax(spectrum)], 5 ]
    popt, pcov = curve_fit(ft.gaussians_const, wl, spectrum, popt)

    if show:
        pl.figure()
        pl.plot(wl, spectrum, "r-")
        pl.plot(wl, ft.gaussians_const(wl, *popt), ":k")
        pl.xlabel("Wavelength (nm)")
        pl.ylabel("Spectrum (arb. units)")

    if delta:
        err = np.sqrt(np.diag(pcov))
        return popt, err

    return popt

# ##################################################################################################################

def get_concentration(spectrum, epspeak, cuvL = 0.1, refspectrum = None, maxOD = 3, retfit = False, withoffset = True):
    """Extract concentration of solute from UV/VIS spectrum and peak absorptivity of solute.
    If a reference spectrum is given, apply peak epsilon to this spectrum and then fit it to the data (this method is suited for very small and very large ODs, especially exceeding the dynamic range of the instrument).

    :param array spectrum: UV/VIS data to analyze (OD). If spectrum is list of spectra, get concentration for every one.
    :param float epspeak: Peak molar extinction (cm-1 M-1).
    :param float cuvL: Length of cuvette (cm) (default 1mm).
    :param array refspectrum: Reference spectrum (optional).
    :param float maxOD: Maximum OD that is taken into account for fitting to reference spectrum (default = 3.0).
    :param bool retfit: If True, return also fit parameters and uncertainties (default = False).
    :param bool withoffset: If True, use a constant offset/background when fitting to the reference spectrum (default).
    :returns: Solute concentration (and fit parameters and errors if retfit = True).
    """
    if refspectrum != None:
        cref = np.amax(refspectrum) / (epspeak * cuvL)

    N = len(spectrum)
    if withoffset:
        fitfunc = lambda x, a, b: np.minimum(np.absolute(a) * refspectrum + b, np.ones(N) * maxOD)
    else:
        fitfunc = lambda x, a: np.minimum(np.absolute(a) * refspectrum, np.ones(N) * maxOD)
    x = np.arange(len(spectrum))
    fitpars = []
    fitsigma = []

    if(isinstance(spectrum, list) or spectrum.ndim > 1):
        c = []
        for y in spectrum:
            if refspectrum == None:
                c.append(np.amax(y) / (epspeak * cuvL))
            else:
                if withoffset:
                    popt = [(np.amax(y)-np.amin(y))/(np.amax(refspectrum)-np.amin(refspectrum)), np.amin(y)-np.amin(refspectrum)]
                else:
                    popt = [(np.amax(y)-np.amin(y))/(np.amax(refspectrum)-np.amin(refspectrum))]
                popt, pcov = curve_fit(fitfunc, x, np.minimum(y, np.ones(N) * maxOD), popt)
                c.append(cref * abs(popt[0]))
                fitpars.append(popt)
                try:
                    fitsigma.append(np.sqrt(np.diag(pcov)))
                except:
                    fitsigma.append(np.zeros(len(popt)))

        if not isinstance(spectrum, list):
            c = np.array(c)
    else:
        if refspectrum == None:
            c = np.amax(spectrum) / (epspeak * cuvL)
        else:
            if withoffset:
                popt = [(np.amax(spectrum)-np.amin(spectrum))/(np.amax(refspectrum)-np.amin(refspectrum)), np.amin(spectrum)-np.amin(refspectrum)]
            else:
                popt = [(np.amax(spectrum)-np.amin(spectrum))/(np.amax(refspectrum)-np.amin(refspectrum))]
            popt, pcov = curve_fit(fitfunc, x,  np.minimum(spectrum, np.ones(N) * maxOD), popt)
            c = cref * abs(popt[0])
            fitpars = popt
            try:
                fitsigma = np.sqrt(np.diag(pcov))
            except:
                fitsigma = np.zeros(len(popt))

    if(retfit == True):
        return [c, fitpars, fitsigma]
    else:
        return c
