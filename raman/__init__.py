"""
.. module: FSRStools.raman
   :platform: Windows
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu>

A collection of functions to process and analyze Raman spectra. These functions work both for spontaneous Raman as well as femtosecond stimulated Raman data. The import functions for spontaneous Raman are optimized for Princeton Instruments' WinSpec ASCII files, while those for FSRS expect the files in the output format of pyFSRS or David Hoffman's LabView FSRS.

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
from scipy.optimize import curve_fit, differential_evolution
from scipy.interpolate import interp1d
from scipy.integrate import simps
import glob
import matplotlib.ticker as ticker
import sys
import FSRStools.fitting as ft

# -------------------------------------------------------------------------------------------------------------------
# some axis label shortcuts

talabel = "Absorption (OD)"
tamlabel = "Absorption (mOD)"
glabel = "Raman Gain (OD)"
gmlabel = "Raman Gain (mOD)"
rslabel = "Raman Shift (cm$-1$)"
wnlabel = "Wavenumber (cm$-1$)"


# -------------------------------------------------------------------------------------------------------------------
# axis calibration functions

def Raman2WL(lambda0, lines):
    """Convert Raman shift (cm-1) to wavelength (nm) for given pump wavelength lambda0.

    :param float lambda0: Pump wavelength (nm).
    :param array lines: Raman shifts (cm-1) to convert to wavelengths.
    :returns: Array of wavelengths (nm).
    """
    return 1e7 / (1e7 / lambda0 - lines)


def WL2Raman(lambda0, lines):
    """Convert wavelength (nm) to Raman shift (cm-1) for given pump wavelength lambda0.

    :param float lambda0: Pump wavelength (nm).
    :param array lines: Wavelengths (nm) to convert to Raman shifts (cm-1).
    :returns: Array of Raman shifts (cm-1).
    """
    return 1e7 * ((lines - lambda0) / (lines * lambda0))


def get_closest_maximum(x, y, x0):
    """Returns the position and value of the (local) maximum closest to x0.

    First, the function gets the local maximum in y closest to x0 by following the gradient. Once the function value does not increase any more, it stops. Second, a parabolic approximation is used on this value and its two neighbours to get a more exact estimate.

    .. note:: This function is not noise resistant. In order to get good results, use a x0-value as close as possible to the peak maximum.

    :param array x: x-values.
    :param array y: y-values / data (same shape as x).
    :param float x0: x-position of a point close to the maximum.
    :returns: (x', y') Position and value of (local) maximum closest to x0.
    """
    i0 = np.argmin(np.absolute(x-x0))
    #   get closest maximum
    if(y[i0+1]>y[i0]):
        i1 = i0 + 1
        while(y[i1] > y[i0]):
            i0 = i1
            i1 = i0 + 1
    else:
        i1 = i0 - 1
        while(y[i1] > y[i0]):
            i0 = i1
            i1 = i0 - 1

    #   now i0 is the index of the maximum
    a = (- y[i0+1] - 4.0 * x[i0] * y[i0] + 2.0 * x[i0] * y[i0-1] + 2.0 * x[i0] * y[i0+1] + y[i0-1]) / (2.0 * (-2.0 * y[i0] + y[i0-1] + y[i0+1]))

    b = -y[i0] + y[i0-1]/2.0 + y[i0+1]/2.0

    c = -(16.0 * y[i0]**2 - 8 * y[i0] * y[i0-1] - 8 * y[i0] * y[i0+1] + y[i0-1]**2 - 2.0 * y[i0-1] * y[i0+1] + y[i0+1]**2)/(8.0 * (-2.0 * y[i0] + y[i0-1] + y[i0+1]))

    return [a, c]

# interpolate data y on wavenumber axis wn
# returns new wavenumber axis and data interpolated data with equidistant sampling points
def interpolate(x, y, N=0, kind='linear'):
    """Convenience wrapper around SciPy's :py:func:`~scipy.interpolate.interp1d` function.

    :param array x: x-axis with non-equidistant sampling points
    :param array y: data sampled on x (same shape as x)
    :param int N: Number of *additional* points that should be added per interval (default = 0).
    :param str kind: Type of interpolation to be used (default = 'linear'). This parameter is directly passed to :py:func:`~scipy.interpolate.interp1d`.
    :returns: (x', y') x- and y-arrays sampled equidistantly.
    """
    if(x[-1] < x[0]):
        x = np.flipud(x)
        y = np.flipud(y)
    x1 = np.linspace(x[0], x[-1], (len(x)-1) * (N+1) + 1)
    y1 = interp1d(x, y, kind)(x1)
    return [x1, y1]

# get vibrational frequencies of common used standards for calibration
# selection of lines is adapted for the red (blue) table
# if sorted is true, return in order of largest peak to smallest peak
def mol_lines(mol = "chex", window = (600, 2000), sorted = False):
    """Returns a list of vibrational frequencies for commonly used standards.

    :param str mol: Identifier of solvent. Currently supported solvents are:

                    - Cyclohexane (`chex`, default)
                    - Benzene (`benzene`)
                    - Methanol (`meoh`)
                    - Isopropanol / isopropyl alcohol (`iso`)
                    - Chloroform (`chloroform`)
    :param tuple window: Select the wavenumber interval of interest in cm-1 (default = (600, 2000)).
    :param bool sorted: If True, return sorted by amplitude (highest first); otherwise by wavenumber (default).
    :returns: Array of Stokes shifts in cm-1.
    """
    # a list of solvent lines; [[wavenumber], [amplitude]]
    spectra = {'chex' : [[384.1, 426.3, 801.3, 1028.3, 1157.6, 1266.4, 1444.4, 2664.4, 2852.9, 2923.8, 2938.3], [2, 3, 95, 15, 6, 14, 12, 8, 100, 58, 67]],
    'benzene':[[605.6, 848.9, 991.6, 1178, 1326, 1595, 3046.8, 3061.9], [2.2, 1.0, 11.7, 2.7, 0.1, 4.5, 8.1, 18.0]],
    'meoh':[[1037, 1453, 2835, 2945], [48, 18, 96, 71]],
    'iso':[[820, 955, 1132, 1454, 2881, 2919, 2938, 2972], [95, 19, 10, 18, 45, 46, 44, 41]],
    'chloroform':[[3178.8, 685.7, 366.7, 1261.8, 781.6, 263.7],[46.7, 14.3, 6.31, 3.58, 9.32, 4.44]] }

    # check first whether solvent exists
    if mol not in spectra.keys():
        print("ERROR: Solvent not found! Choose one of", spectra.keys())
        return np.array([])

    # select proper wavenumber window
    lines = np.array(spectra[mol])
    lines = np.compress((lines[0,:] >= window[0]) & (lines[0,:] <= window[1]), lines, axis=1)

    # return properly sorted array
    if sorted:
        return np.array(lines[0])[np.flipud(np.argsort(lines[1]))]
    else:
        return np.array(lines[0])[np.argsort(lines[0])]

def calibrate(y, lambda0, peaks, mol, show=False):
    """Returns a calibrated x-axis in wavenumbers using a calibration spectrum and a list of vibrational modes.

    :param array y: Calibration spectrum.
    :param float lambda0: Pump wavelength (nm).
    :param array peaks: List of estimated peak positions in indices.
    :param array mol: List of **associated** vibrational frequencies (cm-1). Does not have to have same shape as `peaks`. If shape is different, use the first `min(len(peaks), len(mol))` entries.
    :param bool show: If True, show a plot of pixel index vs wavelength, False by default.
    :returns: Wavenumber axis with same shape as y (non-equidistantly sampled).

    .. seealso:: If the wavenumber axis should be sampled equidistantly, use :py:func:`interpolate`.

    Example::

        import FSRStools.raman as fs

        lmbda0 = 795.6      # pump wavelength
        showCal = True      # display calibration results

        # load the calibration spectrum
        chex, _, _ = loadFSRS("chex")

        # generate wavenumber axis
        wn = fs.calibrate(chex, lmbda0, fs.find_peaks(chex, sorted=True), fs.mol_lines('chex', sorted=True), show=showCal)
    """
    mappx = np.array([])
    mapwl = np.array([])

    # get a list of calibration points (px vs wl)
    line = lambda x, a, m: a * x + m
    x = np.arange(y.size)
    N = min(len(peaks), len(mol))
    for i in range(N):
        x1, y1 = get_closest_maximum(x, y, peaks[i])
        mappx = np.append(mappx, x1)
        mapwl = np.append(mapwl, Raman2WL(lambda0, mol[i]))

    # fit with a line
    popt, _ = curve_fit(line, mappx, mapwl, (1, 0))

    # and convert new wavelength axis to wavenumber axis
    wn = WL2Raman(lambda0, line(x, *popt))

    # plot a nice figure when show == True
    if(show == True):
        pl.figure()
        xtmp = np.linspace(0, len(y), 10)
        pl.plot(xtmp, line(xtmp, *popt), color='k')
        pl.plot(mappx, mapwl, "o")
        pl.xlabel("Pixel")
        pl.ylabel("Wavelength (nm)")

    return wn


# -------------------------------------------------------------------------------------------------------------------
# file loading functions

def loadFSRS(basename, wn = None, timesteps = None, excstr = "exc*", filteroutliers = False):
    """Load and average all FSRS files matching the basename (e.g. using wildcards).

    :param mixed basename: If basename is a `str`, use :py:mod:`glob` to load all matching files using wildcards. If basename is a list, use this list directly without wildcards (only works when no time steps are given).
    :param array wn: If a wavenumber axis is given, the first column is interpolated over this axis (default = None).
    :param array timesteps: If not None, load excited state data using the given time points. The time value is converted to a string and inserted into a `%s` tag in basename, or appended if no `%s` tag is found. Ground state data is not loaded.
    :param str excstr: String that gets appended to the basename to indicate excited state spectra (default = 'exc*').
    :param bool filteroutliers: If True, outliers, i.e., invalid data sets, are removed before averaging. An outlier is defined as a spectrum whose sum over amplitude squared deviates from the mean by more than two standard deviations.
    :returns: When no time steps are given, return the three columns of the FSRS file averaged over all input spectra, i.e. (Raman gain, probe with pump on, probe with pump off). When time steps are given, return three 2d arrays corresponding to the three columns with timesteps along axis 0 and data along axis 1.
    """
    if(timesteps is not None):
        alldata = []

        # load excited state
        for t in timesteps:
            if(t <= 0):
                tstr = "m%d%s" % (abs(t), excstr)
            else:
                tstr = "p%d%s" % (t, excstr)
            data = 0
            if("%s" in basename):
                print("load", basename % tstr)
                files = glob.glob(basename % tstr)              # get all matching files
            else:
                print("load", basename + tstr)
                files = glob.glob(basename + tstr)              # get all matching files

            if(len(files) < 1):
                print("ERROR: No files found!")
                return []

            tmp = []
            for f in files:
                tmp.append(np.loadtxt(f, unpack=True))
            tmp = np.array(tmp)

            # [#][column][data]
            if filteroutliers:
                data = np.mean(tmp[filter_outliers(tmp[:,0,:])],axis=0)
            else:
                data = np.mean(tmp, axis=0)

            if(wn is not None):
                x, data[0] = interpolate(wn, data[0])
            alldata.append(data)

        return np.rollaxis(np.array(alldata), 1)    # rotate data such that its ordered as [column][timepoint][data]
    else:
        data = 0
        if isinstance(basename, str):
            files = glob.glob(basename)             # get all matching files
        else:
            files = basename

        if(len(files) < 1):
            print("ERROR: No files found!")
            return []

        tmp = []
        for f in files:
            tmp.append(np.loadtxt(f, unpack=True))
        tmp = np.array(tmp)

        # [#][column][data]
        if filteroutliers:
            data = np.mean(tmp[filter_outliers(tmp[:,0,:])],axis=0)
        else:
            data = np.mean(tmp, axis=0)

        if(wn is not None):
            x, data[0] = interpolate(wn, data[0])
        return data

#
def loadTA(basename, wl = [], timesteps = [], excstr = "*", filteroutliers = False):
    """Convenience function for loading transient absorption data. The only difference to :py:func:`loadFSRS` is the naming convention for TA excited state data compared to FSRS data.

    .. seealso:: :py:func:`loadFSRS` for list of parameters.
    """
    return loadFSRS(basename, wl, timesteps, excstr, filteroutliers)

# use glob to support wildcards and regExp
# loads Raman spectra and returns the averaged data
def loadRaman(filename):
    """Load ASCII Raman (or other) data.

    :param str filename: Filename / base name of Raman spectra to load. Supports wildcards and regular expression via :py:mod:`glob`. If filename matches multiple files, these are averaged.
    :returns: 2d array containing the (possibly averaged) columns of the input file(s).
    """
    files = glob.glob(filename)             # get all matching files
    if(len(files) < 1):
        print("ERROR: No files found!")
        return []

    tmp = []
    for f in files:
        tmp.append(np.loadtxt(f, unpack=True))
    tmp = np.array(tmp)

    data = np.mean(tmp, axis=0)
    return data

# -------------------------------------------------------------------------------------------------------------------
# Data filtering / noise improvements

def filter_outliers(y, eps = 2.0):
    """Returns a list of indices for datasets that NOT outliers. An outlier is defined as a spectrum whose sum over amplitudes squared deviates from the ensemble average by more than eps times the ensemble standard deviation.

    :param array y: A 2d array of spectra, where the first axis is the number of the dataset.
    :param float eps: Threshold that defines an outlier in units of standard deviations (default = 2).
    :returns: Array of indices of spectra in y that are NOT outliers.
    """
    mean = np.mean(y, axis=0)
    chi2 = np.sum((y - mean)**2, axis=1)
    meanchi2 = np.mean(chi2)
    stdchi2 = np.std(chi2)
    return np.nonzero(np.absolute(chi2 - meanchi2) <= eps * stdchi2)

# taken from http://www.scipy.org/Cookbook/SavitzkyGolay
def savitzky_golay(y, window_size, order, deriv=0):
    """Implementation of the Savitzky Golay filter. Code adapted from http://www.scipy.org/Cookbook/SavitzkyGolay.

    :param array y: Data array to be smoothed.
    :param int window_size: Length of the smoothing window. Must be an odd integer.
    :param int order: Order of the smoothing polynomial. Must be less than window_size - 1.
    :param int deriv: Order of the derivative to compute (default = 0).
    :returns: Smoothed signal or its derivative. Same shape as y.
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order+1)
    half_window = (window_size-1) // 2 # divide and round off

    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]

    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve( m, y, mode='valid')

def noise_estimate(y):
    """Estimate the standard deviation of Gaussian white noise contained in a spectrum. Code is based on Schulze et al., *Appl. Spectrosc.* **60**, 820 (2006).

    :param array y: Input spectrum.
    :returns: Noise estimate for input spectrum (float).
    """
    ws = 21
    e0 = 1e8
    for i in range(int(y.size / ws)):
        e = np.var( np.roll(y[i*ws:(i+1)*ws], -2)[0:-2] - 2.0 * np.roll(y[i*ws:(i+1)*ws], -1)[0:-2] + (y[i*ws:(i+1)*ws])[0:-2] ) / 3.0
        if(e < e0):
            e0 = e
    return np.sqrt(e0)

def denoise(y):
    """Fully automatic optimized smoothing algorithm. Code is based on Schulze et al. *Appl. Spectrosc.* **62**, 1160 (2008).
    The algorithm uses a repeated application of a 3-pixel zero-order Savitzky-Golay filter until a stopping criterion is fulfilled. This stopping criterion is equivalent to a notable distortion of the signal due to smoothing.

    :param array y: Input spectrum. If y is a `list` or a 2d array of spectra, denoise every spectrum in that list.
    :returns: Filtered spectrum with same shape as y.
    """
    if(isinstance(y, list) or y.ndim > 1):
        out = []
        for sp in y:
            out.append(denoise(sp))
        return np.array(out)
    else:
        N = float(y.size)
        s = noise_estimate(y)   # get stddev of input data
        m = np.copy(y)
        while(True):
            m = savitzky_golay(m, 3, 0) # apply smoothing
            if(np.sum((y - m)**2 / s**2) > N):
                break
        return m

def FT_denoise(y, cutoff = 1, filter = 'rect'):
    """Apply a Fourier low pass filter to the data to remove high frequency noise.

    :param array y: Input spectrum. If y is a `list` or 2d-array of spectra, filter every spectrum.
    :param int cutoff: Low pass cutoff position taken from the high frequency side in array indices (0 means no filtering, default = 1).
    :param str filter: Type of step function to apply as filter. Currently supported are:

                       - 'cos2' - A cosine squared.
                       - 'linear' - A linear onset.
                       - 'rect' - A rectangular step function (default).
    :returns: Filtered spectrum with same shape as y.
    """
    if(isinstance(y, list) or y.ndim > 1):
        out = []
        for sp in y:
            out.append(FT_denoise(sp, cutoff, filter))
        return np.array(out)
    else:
        # get FFT - use padding to reduce edge effects
        ypad = np.pad(y, len(y), mode='reflect', reflect_type = 'odd')
        FT = np.fft.rfft(ypad * np.hanning(len(ypad)))

        xtmp = np.arange(0, cutoff)
        if filter == 'cos2':
            FT[-cutoff:] = FT[-cutoff] * np.cos(4.0 * np.pi * xtmp / cutoff)**2
        elif filter == 'linear':
            FT[-cutoff:] = FT[-cutoff] / cutoff * np.flipud(xtmp)
        else:
            FT[-cutoff:] = np.zeros(cutoff)

        return np.fft.irfft(FT)[len(y):-len(y)]

# -------------------------------------------------------------------------------------------------------------------
# baseline correction and smoothing functions

def rayleigh_correction(y):
    """Remove the baseline arising from the Rayleigh line by fitting a Gauss-Lorentz curve to the data.
    The position of the Rayleigh line (left or right end) is chosen by the amplitude of the spectrum. To reduce the effect of huge Raman peaks, the second order derivative is subtracted from the spectrum before fitting the baseline.

    :param array y: Input spectrum. If y is a `list` or 2d-array of spectra, filter every spectrum.
    :returns: Spectrum without Rayleigh baseline, same shape as y.
    """
    if(isinstance(y, list) or y.ndim > 1):
        out = []
        for sp in y:
            out.append(rayleigh_correction(sp))
        return np.array(out)
    else:
        # generate pixel axis
        x = np.arange(len(y))

        # partially remove the peaks
        ytmp = y - np.gradient(np.gradient(y))

        # fit stripped spectrum by Voigt profile
        popt = [ytmp[0], np.amax(ytmp) - np.amin(ytmp), (x[[0,-1]])[np.argmax(ytmp[[0,-1]])], (x[0]-x[-1])/10, 0.9]
        popt, _ = curve_fit(ft.voigts_const, x, ytmp, popt, maxfev=10000)

        # return residuum
        return y - ft.voigts_const(x, *popt)

def interpolated_bl_correction(x, y, px, py = None, usedatay = True):
    """Remove a baseline obtained by interpolating a set of fulcrums.

    :param array x: x-axis.
    :param array y: Input spectrum, same shape as x. If y is a `list` or 2d-array of spectra, filter every spectrum.
    :param array px: - If `py != None`: list of x-coordinates of interpolation points.
                     - If `py == None` and `usedatay == True`: list of x-coordinates of interpolation points.
                     - If `py == None` and `usedatay == False`: list of x- and y-coordinates of interpolation points in the form [x0, y0, x1, y1, x2, y2, ...].
    :param array py: List of y-coordinates of interpolation points (optional).
    :param bool usedatay: If True, use the y-data at px for py (default).
    :returns: Baseline corrected spectrum.
    """
    if(isinstance(y, list) or y.ndim > 1):
        out = []
        for sp in y:
            out.append(interpolated_bl_correction(x, sp, px, py, usedatay))
        return np.array(out)
    else:
        # get interpolation points
        if usedatay and py is None:
            x0 = px
            y0 = interp1d(x, y, 'linear')(x0)
        elif not usedatay and py is None:
            x0 = p0[::2]
            y0 = p0[1::2]
        else:
            x0 = px
            y0 = py

        # make sure the end points are contained in the interpolation
        if(np.amin(x0) > np.amin(x)):
            x0 = np.append(x0, np.amin(x))
            y0 = np.append(y0, y[np.argmin(x)])
        if(np.amax(x0) < np.amax(x)):
            x0 = np.append(x0, np.amax(x))
            y0 = np.append(y0, y[np.argmax(x)])

        # make sure the points are correctly sorted for interpolation
        s = np.argsort(x0)
        x0 = x0[s]
        y0 = y0[s]

        # return the spectrum minus the interpolated baseline
        return y - interp1d(x0, y0, 'cubic')(x)

def baseline_correction(y, n0 = 2, verbose = False, iterate = True):
    """Automated baseline removal algorithm. Code is based on Schulze et al., *Appl. Spectrosc.* **65**, 75 (2011).

    Works better if baseline is more flat at the beginning and end of the spectrum. If divisor is too high, there will be some ringing of the baseline. Sometimes it is better to start with a higher value for n0 to get a good baseline removal, especially when the baseline is wavy and there are strong Raman lines.

    :param array y: Input spectrum. If y is a list or a 2d-array of spectra, apply correction to each one.
    :param int n0: Initial divisor for window size, i.e. initial window size is size of spectrum divided by n0. Must be at least 1 (default = 2).
    :param bool verbose: If True, print final number of iterations and final divisor at the end (default = False).
    :param bool iterate: If True, automatically increase the order of the divisor until optimal baseline removal is achieved. If False, just use the value given by n0.
    :returns: Baseline corrected spectrum with same shape as y.
    """
    if(isinstance(y, list) or y.ndim > 1):
        out = []
        for sp in y:
            out.append(baseline_correction(sp, n0, verbose, iterate))
        return np.array(out)
    else:
        if(n0 < 1 or n0 > y.size/3):
            print("n0 is out of range (1, %d)! Set to 1." % (y.size/3))
            n0 = 1

        s0 = noise_estimate(y)                                      # get stddev of input data
        Npad = len(y)                                               # number of points for padding
        ypad = np.pad(y, Npad, mode='reflect', reflect_type = 'odd') # create padded spectrum to reduce edge effects
        N = ypad.size
        sblbest = np.sum(ypad**2)                                   # estimate for baseline chi2 best value
        blbest = np.zeros(N)                                        # store best baseline estimate

        wndiv = n0
        while(wndiv < int(N/3)):                                    # window size reduction

            # prepare filter window size
            wn = int(N / wndiv)
            if(wn % 2 == 0):
                wn += 1

            y0 = np.copy(ypad)                                      # copy original spectrum
            bl = np.zeros(N)                                        # initialize empty baseline
            sblold = np.sum(y0**2)                                  # estimate for baseline chi2

            cbl = 0
            while(cbl < 1000):                                      # baseline estimation
                ys1 = denoise(y0)                                   # autosmoothing step
                cint = 0
                while(cint < 1000):                                 # peak stripping
                    blint = savitzky_golay(ys1, wn, 0)              # intermediate baseline estimate
                    ys2 = np.where(blint + 2.0 * s0 < ys1, blint, ys1)  # replace values bigger than baseline + 2x stdev by baseline

                    if(np.sum((ys2 - ys1)**2 / s0**2) < N):         # stripping changes are below noise level
                        break                                       # break inner loop
                    else:
                        ys1 = np.copy(ys2)                          # else: proceed with partially stripped spectrum
                        cint += 1                                   # increase counter

                sbl = np.sum(blint**2)
                if(sbl >= sblold and cbl > 2):                      # chi2 of intermediate baseline has not been reduced after > 2 iterations
                    break                                           # break second loop
                else:
                    y0 -= blint                                     # set new starting spectrum by subtracting intermediate baseline
                    bl += blint                                     # add intermediate baseline to total baseline
                    sblold = np.copy(sbl)                           # store old value
                    cbl += 1                                        # increase counter

            if not iterate:                                         # stop here if iterate = False
                break

            if(sblold < sblbest or wndiv == n0):                    # could reduce best chi2 or completed just first iteration
                sblbest = np.copy(sblold)                           # new intermediate bl is flatter than previous best intermediate bl
                wndiv += 1                                          # reduce window size
                blbest = np.copy(bl)                                # store new best baseline estimate
            else:
                break

        if verbose == True:
            print("finished with divisor %d after %d iterations" % (wndiv, cbl))

        # return baseline corrected spectrum
        return (ypad - blbest)[Npad:-Npad]

def FT_baseline_correction(y, cutoff = None, filter = 'rect'):
    """Automatic baseline correction based on a Fourier high pass filter after removal of regression line. The stopping criterion for the automatic cutoff search is that the incremental change in the sum over squares should be at least one percent.

    :param array y: Input spectrum. If y is a list or a 2d array of spectra, correct all spectra.
    :param int cutoff: Cutoff frequency for high pass filter:

                       - If `cutoff == None`, display the Fourier transform of the (first) input spectrum and stop the script.
                       - If `cutoff > 0`, use cutoff directly for high pass filter.
                       - If `cutoff == -1`, do an automatic determination of the optimal cutoff.
    :param str filter: Type of filter function to use. Currently supported values are:

                       - 'rect' (default): a rectangular step function.
                       - 'cos2': a cosine squared.
                       - 'linear': linear interpolation.
    :returns: Baseline corrected spectrum with same shape as y.
    """
    if(isinstance(y, list) or y.ndim > 1):
        out = []
        for sp in y:
            out.append(FT_baseline_correction(sp, cutoff, filter))
        return np.array(out)
    else:
        wnx = np.arange(0, len(y))

        # subtract regression line
        line = lambda x, a, m: a * x + m
        popt, _ = curve_fit(line, wnx, y, [1, 0])
        y = y - line(wnx, *popt)

        # get FFT - use padding to reduce edge effects
        ypad = np.pad(y, len(y), mode='reflect', reflect_type = 'odd')
        FT = np.fft.rfft(ypad * np.hanning(len(ypad)))

        if(cutoff == None):
            pl.figure()
            pl.plot(np.absolute(FT))
            pl.show()
            sys.exit()
        elif(cutoff == 0):
            c = 10
            chi20 = 1e8
            chi2 = np.sum(y**2)
            while(abs(chi20 - chi2)/chi20 > 1e-2 and c < len(y)):

                c += 1
                xtmp = np.arange(0, c)

                if filter == 'cos2':
                    FT[0:c] = FT[c] * np.sin(4.0 * np.pi * xtmp / c)**2
                elif filter == 'linear':
                    FT[0:c] = FT[c] / c * xtmp
                else:
                    FT[0:c] = np.zeros(c)

                y1 = np.fft.irfft(FT)[len(y):2*len(y)]

                chi20 = chi2
                chi2 = np.sum(y1**2)
            print(c, "iterations, dchi =", abs(chi20 - chi2)/chi20)
            return y1
        else:
            xtmp = np.arange(0, cutoff)
            if filter == 'cos2':
                FT[0:cutoff] = FT[cutoff] * np.sin(4.0 * np.pi * xtmp / cutoff)**2
            elif filter == 'linear':
                FT[0:cutoff] = FT[cutoff] / cutoff * xtmp
            else:
                FT[0:cutoff] = np.zeros(cutoff)

            return np.fft.irfft(FT)[len(y):2*len(y)]

# -------------------------------------------------------------------------------------------------------------------
# solvent/ground state subtraction functions

def solvent_subtract_chi2(y, solvent, scaling='const', shiftx = False):
    """Subtract a solvent or ground state spectrum from a spectrum or list of spectra. The optimum scaling of the spectrum to subtract is found by minimizing the sum over the residual spectrum squared. This function works good if the residual spectrum has weaker peaks than the solvent / ground state spectrum.

    :param array y: Input spectrum. If y is a list or 2d array of spectra, apply solvent subtraction to each spectrum.
    :param array solvent: Spectrum to subtract (same shape as y).
    :param str scaling: Type of scaling function ('const', default, or 'linear'). Linear scaling is used to account for self-absorption effects.
    :param bool shiftx: If True, the spectrum to subtract can also be translated along the x-axis (default = False).
    :returns: Solvent / ground state corrected spectrum with same shape as y.
    """
    # fitting functions
    if(shiftx == False):
        f1 = lambda x, a, f0: solvent * a + f0
        f2 = lambda x, a, m, f0: solvent * (m * x + a) + f0
    else:
        f1 = lambda x, a, f0, dx: shift_data(x, solvent, dx) * a + f0
        f2 = lambda x, a, m, f0, dx: shift_data(x, solvent, dx) * (m * x + a) + f0

    if(isinstance(y, list) or y.ndim > 1):
        out = []
        for sp in y:
            out.append(solvent_subtract_chi2(sp, solvent, scaling, shiftx))
        return np.array(out)
    else:
        x = np.arange(len(y))
        if(scaling == 'const'):
            if(shiftx == False):
                popt, _ = curve_fit(f1, x, y, [1.0, 0.0])
            else:
                popt, _ = curve_fit(f1, x, y, [1.0, 0.0, 0.0])
            out = y - f1(x, *popt)
        else:
            if(shiftx == False):
                popt, _ = curve_fit(f2, x, y, [0.0, 1.0, 0.0])
            else:
                popt, _ = curve_fit(f2, x, y, [0.0, 1.0, 0.0, 0.0])
            out = y - f2(x, *popt)
        print("solvent subtract fit results: ", popt)
        return out


def solvent_subtract(y, solvent, peaks, scaling='const', type='lor'):
    """Subtract solvent or ground state spectrum from a spectrum or a list of spectra. The optimum scaling of the spectrum to subtract is found by fitting one or more solvent peaks using a Lorentzian or Gaussian. This function works well with spectra having peaks of similar intensity as the solvent spectrum.

    :param array y: Input spectrum. If y is a list or 2d array of spectra, apply solvent subtraction to each spectrum.
    :param array solvent: Spectrum to subtract (same shape as y).
    :param array peaks: Array of tuples, one for each solvent peak, giving the data interval in indices used for fitting. Each peak is fit by a single Lorentzian or Gaussian.
    :param str scaling: Type of scaling function ('const', default, or 'linear'). Linear scaling is used to account for self-absorption effects and requires at least two solvent peaks.
    :param str type: Type of fit function ('lor', default, or 'gauss').
    :returns: Solvent / ground state corrected spectrum with same shape as y.
    """
    if(isinstance(y, list) or y.ndim > 1):
        out = []
        for sp in y:
            out.append(solvent_subtract_lor(sp, solvent, peaks, scaling, type))
        return np.array(out)
    else:
        x = np.arange(len(y))
        if(type == 'lor'):
            func = ft.lorentzians_line
        else:
            func = ft.gaussians_line
        areas = np.zeros(len(peaks))
        positions = np.zeros(len(peaks))

        for i in range(len(peaks)):
            poptspe, _ = curve_fit(func, x[peaks[i][0]:peaks[i][1]], y[peaks[i][0]:peaks[i][1]], [ y[peaks[i][0]], 0, 1, x[int((peaks[i][0]+peaks[i][1])/2)], x[peaks[i][0]]-x[peaks[i][1]]])
            poptsol, _ = curve_fit(func, x[peaks[i][0]:peaks[i][1]], solvent[peaks[i][0]:peaks[i][1]], [ solvent[peaks[i][0]], 0, 1, x[int((peaks[i][0]+peaks[i][1])/2)], x[peaks[i][0]]-x[peaks[i][1]]])

            positions[i] = poptsol[3]
            areas[i] = np.absolute((poptspe[2] * poptspe[4]) / (poptsol[2] * poptsol[4]))

        if(scaling == 'const'):
            popt, _ = curve_fit(ft.const, positions, areas, [1.0])
            out = y - ft.const(x, *popt) * solvent
        else:
            popt, _ = curve_fit(ft.line, positions, areas, [1.0, 0.0])
            out = y - ft.line(x, *popt) * solvent

        print("solvent subtract fit results: ", popt)
        return out

# -------------------------------------------------------------------------------------------------------------------
# time zero functions - use in conjunction with cross correlation data

# translate 1d data by amount dx
# uses padding to reduce edge effects
def shift_data(x, y, dx):
    """Smoothly translate data by an arbitrary amount along the x-axis using Fourier transformation.

    :param array x: x-axis.
    :param array y: Data, same shape as x.
    :param float dx: Delta value.
    :returns: Translated data, same shape as y.

    .. note:: This function uses padding to reduce edge artefacts. While the output has the same shape as the input, strictly speaking, the `dx / (x[1] - x[0])` first values on the left or right edge (depending on sign of dx) are invalid.
    """
    sx = x[1]-x[0]
    Npad = max(int(abs(2 * dx / sx)), 1) * 2
    ypad = np.pad(y, Npad, mode='reflect', reflect_type = 'odd')

    w = np.fft.rfftfreq(len(ypad), d=sx) * 2.0 * np.pi
    ypad = np.fft.irfft(np.fft.rfft(ypad) * np.exp(-1j * w * dx))

    return ypad[Npad:len(y)+Npad]

# shift all frequency columns along time axis to set t0 (given in c) to 0
# data d is assumed to be in the format [time][wl]
def correct_t0(t, d, c):
    """Shift all frequency columns in d along the time axis to correct t0 to 0.

    :param array t: Time axis.
    :param array d: 2d array containing spectra vs time.
    :param array c: Time t0 for each wavelength / wavenumber, e.g. obtained by cross-correlation.
    :returns: Shifted spectra with time t0 at zero, same shape as d.
    """
    A = np.copy(d)
    # iterate over all frequency columns
    for i in range(d.shape[1]):
        tmp = shift_data(t, d[:,i], -c[i])
        A[:,i] = tmp
    return A

# -------------------------------------------------------------------------------------------------------------------
# normalization / data selection functions

# normalize data in y
# mode = max: divide by maximum
# mode = 01: shift minimum to zero before dividing by max
# mode = area: normalize by area
def norm(y, mode='max'):
    """Normalize spectrum.

    :param array y: Spectral data.
    :param str mode: Type of normalization:

                     - 'max' - Divide by maximum.
                     - 'area' - Divide by sum over absolute value.
                     - '01' - Scale to interval [0, 1] by subtracting offset before dividing by maximum.
    :returns: Normalized spectrum with same shape as y.
    """
    if mode == 'max':
        return y / np.amax(y)
    if mode == '01':
        y = y - np.amin(y)
        return y / np.amax(y)
    if mode == 'area':
        return y / np.sum(np.absolute(y))

def cut(x, y, x0, x1):
    """Cut out a subarray from x and y according to the *closed* interval [xfrom, xto].

    :param array x: x-axis.
    :param array y: Data (same shape as x; if 2d array, same shape as x along axis 1).
    :param float x0: Lower bound of interval in same units as x.
    :param float x1: Upper bound of interval in same units as x.
    :returns: Sliced arrays x and y.

    .. note:: This function does not perform any interpolation but rather allows slicing of arrays using physical values rather than array indices.
    """
    if x0 > x1:
        x0, x1 = x1, x0
    if x0 == x1:
        x1 = x0 + x[1] - x[0]
    u = np.compress((x >= x0) & (x <= x1), x)
    if y.ndim > 1:
        v = np.compress((x >= x0) & (x <= x1), y, axis=1)
    else:
        v = np.compress((x >= x0) & (x <= x1), y)
    return u, v

def at(x, y, x0):
    """Return the value of y with x-coordinate closest to x0.

    :param array x: x-axis.
    :param array y: Data (same shape as x).
    :param float x0: x-coordinate of desired data point. If x0 is a list or array, return an array of y values.
    :returns: Data point with x-coordinate closest to x0.
    """
    if isinstance(x0, list) or isinstance(x0, np.array):
        out = []
        for xp in x0:
            out.append(at(x, y, xp))
        return np.array(out)
    else:
        return y[np.argmin(np.absolute(x - x0))]

# -------------------------------------------------------------------------------------------------------------------
# TA analysis functions

def bandintegral(x, y, x0, x1):
    """Calculate the band integral over time dependent data.

    :param array x: x-axis (wavelength / wavenumber).
    :param array y: 2d array containing spectra vs time.
    :param float x0: Left boundary of the band integral.
    :param float x1: Right boundary of the band integral.
    :returns: Band integral as function of time (= axis 1 of y).
    """
    i0 = np.argmin(np.absolute(x-x0))
    i1 = np.argmin(np.absolute(x-x1))
    if i0 > i1:
        i0, i1 = i1, i0

    return simps(y[:,i0:i1], x[i0:i1], axis=1)

# -------------------------------------------------------------------------------------------------------------------
# peak fitting functions


def find_peaks(y, wnd = 9, ath = 0.01, sth = 1.0, bl = 25, show = False, sorted=False):
    """Automated peak finding algorithm based on zero crossings of smoothed derivative as well as on slope and amplitude thresholds.

    :param array y: Spectrum.
    :param int wnd: Smoothing window size. Has to be an odd integer (default = 9).
    :param float ath: Amplitude threshold in units of maximum amplitude (default = 0.01).
    :param float sth: Slope threshold in units of estimated noise standard deviation (default = 1.0).
    :param in bl: Initial window size divisor for automated baseline removal. If None, no baseline removal is performed (default = 25).
    :param bool show: If True, make a plot showing the extracted peak positions in the spectrum.
    :param bool sorted: If True, return sorted by amplitude (highest first). If False, returns sorted by frequency.
    :returns: A list of peak position *INDICES*, not wavenumbers.
    """
    g = np.copy(y)

    # baseline removal
    if bl is not None:
        g = baseline_correction(g, bl)

    # get derivative of smoothed data
    dg = np.gradient(savitzky_golay(g, wnd, 0))

    # get noise estimate for thresholds
    s0 = noise_estimate(g)
    gmin = ath * np.amax(g)
    smin = sth * s0

    # find peaks
    peaks = []
    gs = []
    for i in range(1, dg.size-1):   # leave out end points
        # apply criteria: local max, min slope and min amp
        if(dg[i] > 0 and dg[i+1] < 0 and dg[i]-dg[i+1] > smin and max(g[i], g[i+1]) > gmin):
            if(g[i] > g[i+1]):
                peaks.append(i)
                gs.append(g[i])
            else:
                peaks.append(i+1)
                gs.append(g[i+1])
    peaks = np.array(peaks)

    # sort for peak amplitude
    if sorted:
        peaks = peaks[np.flipud(np.argsort(gs))]

    if show == True:
        pl.figure()
        pl.plot(g, color='k')
        for i in range(len(peaks)):
            pl.plot(peaks[i], g[peaks[i]], "o", color="r")
            pl.annotate(str(i), (peaks[i], g[peaks[i]]))

    return peaks

# utility function to quickly extract data x and y coordinates by double clicking in the plot
# after the function is executed, the script terminates
# these functions are not to be called directly but act as event handler
pick_peaks_lastclickx = 0
pick_peaks_lastclicky = 0
def pick_peaks_onclick_x_y(event):
    global pick_peaks_lastclickx, pick_peaks_lastclicky
    if(pick_peaks_lastclickx == event.x and pick_peaks_lastclicky == event.y):
        pick_peaks_lastclickx = 0
        pick_peaks_lastclicky = 0
        print ("%f, %f" % (event.xdata, event.ydata))
    else:
        pick_peaks_lastclickx = event.x
        pick_peaks_lastclicky = event.y

def pick_peaks_onclick(event):
    global pick_peaks_lastclickx, pick_peaks_lastclicky
    if(pick_peaks_lastclickx == event.x and pick_peaks_lastclicky == event.y):
        pick_peaks_lastclickx = 0
        pick_peaks_lastclicky = 0
        print ("%f," % (event.xdata))
    else:
        pick_peaks_lastclickx = event.x
        pick_peaks_lastclicky = event.y

def pick_peaks(x, y=None):
    """Utility function to quickly extract x coordinates of points from data by double clicking in a plot window.
    For each double click, the x coordinates are printed on stdout. After the function is executed, the script terminates.

    Use this function to get peak positions as input for :py:func:`get_peak_estimates` or :py:func:`calibrate`, for example.

    :param array x: x-axis or data if y is None.
    :param array y: Data (optional). If None, the function displays the data vs indices.

    .. seealso: :py:func:`pick_peaks_x_y` for extracting x and y coordinates.
    """
    pl.figure()
    if(y != None):
        pl.plot(x, y)
    else:
        pl.plot(x)
    pl.gcf().canvas.mpl_connect('button_press_event', pick_peaks_onclick)
    pl.show()
    sys.exit()

# if only x is given, it takes the role of y
def pick_peaks_x_y(x, y=None):
    """Utility function to quickly extract x and y coordinates of points from data by double clicking in a plot window.
    For each double click, the x and y coordinates are printed on stdout. After the function is executed, the script terminates.

    Use this function to get interpolation points as input for :py:func:`interpolated_bl_correction`, for example.

    :param array x: x-axis or data if y is None.
    :param array y: Data (optional). If None, the function displays the data vs indices.

    .. seealso: :py:func:`pick_peaks` for extracting just x coordinates.
    """
    pl.figure()
    if(y != None):
        pl.plot(x, y)
    else:
        pl.plot(x)
    pl.gcf().canvas.mpl_connect('button_press_event', pick_peaks_onclick_x_y)
    pl.show()
    sys.exit()

# get a list of parameters A, x0, dx for each peak whose x coordinate is listed in peaks
# peaks is in same units as x
# x, y is data
def get_peak_estimate(x, y, peaks):
    """Returns a list of estimated peak parameters from a spectrum for each peak whose approximate x-coordinate is listed in peaks. The output can be directly used in the Lorentzian or Gaussian fitting functions or in :py:func:`fit_peaks`.

    :param array x: x-axis.
    :param array y: Data, same shape as x.
    :param array peaks: List with peak positions in same units as x.
    :returns: A list of estimated parameters for each peak:

                - A amplitude,
                - x0 center,
                - dx FWHM.
    """
    # list for results
    est = []
    i0 = []
    imin = []
    imax = []

    # get gradient to better pronounce the peaks
    dy = np.gradient(y)

    # iterate over all guesses and get indices of peaks
    for i in range(len(peaks)):
        i0.append( np.argmin(np.absolute(x - peaks[i])) )
        if(i != 0):
            imin.append( np.argmin(np.absolute(x - (peaks[i] + peaks[i-1]) / 2) ) )
        if(i != len(peaks)-1):
            imax.append( np.argmin(np.absolute(x - (peaks[i] + peaks[i+1]) / 2) ) )

        if(i == 0):
            imin.append(np.maximum(0, 2 * i0[-1] - imax[-1]))
        if(i == len(peaks)-1):
            imax.append(np.maximum(0, 2 * i0[-1] - imin[-1]))

    # now get the estimates
    for i in range(len(peaks)):

        i1 = np.argmin( dy[imin[i]:imax[i]] - (dy[imax[i]] - dy[imin[i]])/(x[imax[i]] - x[imin[i]]) * x[imin[i]:imax[i]] )+imin[i]
        i2 = np.argmax( dy[imin[i]:imax[i]] - (dy[imax[i]] - dy[imin[i]])/(x[imax[i]] - x[imin[i]]) * x[imin[i]:imax[i]] )+imin[i]
        delta = np.maximum(np.absolute(i0[i] - i1), np.absolute(i0[i]  - i2))

        i12 = np.maximum(0, i0[i] - 2 * delta)
        i22 = np.minimum(len(x)-1, i0[i] + 2 * delta)

        est.append(y[i0[i]] - 0.5 * (y[i12] + y[i22]))
        est.append(x[i0[i]])
        est.append(np.absolute( x[i2] - x[i1] ))

    return np.array(est)

def fit_peaks(x, y, popt0, fit_func, bounds = None, estimate_bl = False, use_popt0 = True, inc_blorder = 1, max_blorder = -1, global_opt = True):
    """Try to fit Raman peaks with (automatic) baseline fitting.

    :param array x: x-axis.
    :param array y: Data. If y is a list or 2d array of spectra, fit every spectrum.
    :param array popt0: Initial parameters for fitting, for example obtained from :py:func:`get_peak_estimate`.
    :param function fit_func: Fitting function to be used to fit the data. If `estimate_bl` is True, this function should only fit the peaks, not the baseline.
    :param array bounds: Array of tuples giving the minimum and maximum allowed value for each parameter in popt0. If not None, use :py:func:`scipy.optimize.differential_evolution` to fit the peaks. (default = None).
    :param bool estimate_bl: If True, attempt an automatic baseline fitting using n-th order polynomials. This baseline is added to any basline contained in fit_func.
    :param bool use_popt0: If fitting multiple spectra, control whether the initial parameters are always the ones provided in `popt0` (True, default) or the ones from the last spectrum (False).
    :param int inc_blorder: When fitting the baseline, the order of the polynomial is increased by `inc_blorder` until no further improvement can be achieved (default = 1).
    :param int max_blorder: Maximum permitted order of the baseline polynomial. Set to -1 for no restriction (default).
    :param bool global_opt: If True, attempt a simultaneous fit of baseline AND peaks once a decent fit for each has been found.
    :returns: The fitted spectrum, same shape as y, and the final peak fit parameters (popt0).
    """
    popt_p0 = np.copy(popt0)

    if(isinstance(y, list) or y.ndim > 1):
        out = []
        outpopt = []
        for sp in y:

            tmp, popt = fit_peaks(x, sp, popt_p0, fit_func, bounds, estimate_bl, use_popt0, inc_blorder, global_opt, max_blorder)

            out.append(tmp)
            outpopt.append(popt)

            if(use_popt0 == False):
                popt_p0 = np.copy(popt)

        return [np.array(out), np.array(outpopt)]
    else:

        N = len(x)

        bl = np.zeros(N)
        blorder = 1
        popt_bl = np.array([0])

        func = lambda p, x, y: (fit_func(x, *p) - y)**2
        func_glob = lambda x, *p: fit_func(x, *p[0:len(popt0)]) + ft.poly(x, *p[len(popt0):])

        if(estimate_bl == True):

            err1 = 1e8
            while(1):

                y1 = np.copy(y)
                y1old = np.zeros(N)

                while( np.sum((y1old - y1)**2)/N > 1e-7 ):

                    popt_p = np.copy(popt_p0)
                    y1old = np.copy(y1)

                    # fit baseline
                    popt_bl, pcov_bl = curve_fit(ft.poly, x, y1, np.zeros(blorder))
                    bl = ft.poly(x, *popt_bl)
                    y1 = y - bl

                    # fit peaks
                    try:
                        if bounds is None:
                            popt_p, pcov_p = curve_fit(fit_func, x, y1, popt_p)
                        else:
                            popt_p = differential_evolution(func, bounds, args=(x, y1)).x
                    except:
                        pass
                    y1 = y - fit_func(x, *popt_p)

                # save error for next order
                y1 = fit_func(x, *popt_p) + bl

                err = np.sum((y - y1)**2)/N

                if(err < err1 and ((max_blorder == -1) or blorder < max_blorder)):
                    err1 = err
                    blorder += inc_blorder
                else:
                    break

            # try a global optimization of baseline and peaks simultaneously
            if global_opt:
                popt, pcov = curve_fit(func_glob, x, y, np.append(popt_p0, popt_bl))
                popt_p = popt[0:len(popt0)]
                pcov_p = pcov[0:len(popt0), 0:len(popt0)]
                popt_bl = popt[len(popt0):]
                pcov_bl = pcov[len(popt0):, len(popt0):]
            elif bounds is not None:
                pcov_p = np.zeros((len(popt0), len(popt0)))
        else:
            popt_p, pcov_p = curve_fit(fit_func, x, y, popt_p)

        print("Peak fit parameters:")
        perr = np.sqrt(np.diag(pcov_p))
        for i in range(len(popt_p)):
            print(i, ": ", popt_p[i], "+-", perr[i])

        if estimate_bl:
            print("Baseline fit parameters:")
            print("polynomial order = ", blorder)
            perr = np.sqrt(np.diag(pcov_bl))
            for i in range(len(popt_bl)):
                print(i, ": ", popt_bl[i], "+-", perr[i])

        return [fit_func(x, *popt_p) + bl, popt_p]

def peak_remove(y, peaks):
    """Remove (narrow) peaks like single pixel artefacts from Raman data by linear interpolation.

    :param array y: Spectrum.
    :param array peaks: List of tuples containing the first and last index of the interval to remove.
    :returns: Stripped spectrum with same shape as y.
    """
    ytmp = np.copy(y)
    for p in peaks:
        ytmp[p[0]+1:p[1]] = (ytmp[p[1]] - ytmp[p[0]]) * (np.arange(p[0]+1, p[1]) - p[0])/(p[1] - p[0]) + ytmp[p[0]]
    return ytmp

def peak_area(y, peaks):
    """Returns the area under the spectrum in a given interval after removing a linear baseline.

    :param array y: Spectrum.
    :param array peaks: List of tuples containing the first and last index of the interval to integrate over.
    :returns: List of peak areas.
    """
    areas = np.zeros(len(peaks))
    for i in range(len(peaks)):
        p = peaks[i]
        areas[i] = simps(y[p[0]+1:p[1]] - ((y[p[1]] - y[p[0]]) * (np.arange(p[0]+1, p[1]) - p[0])/(p[1] - p[0]) + y[p[0]]))
    return areas
