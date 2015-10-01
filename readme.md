Welcome to FSRStools' documentation!
=====================================

FSRStools is a compilation of python modules for the script-based analysis of scientific data.
It was developed with application towards spontaneous and time-resolved Raman spectra in mind, but can be used quite general.

Package overview:

- **crosscorr**: Analysis of Kerr-gate cross-correlation data.
- **fitting**: Compilation of useful functions for curve-fitting including mathematical functions, LPSVD, NIPALS, etc.
- **profilometer**: Tools for optical profilometry and thin film measurements.
- **raman**: Analysis of spontanoues and time-resolved Raman spectra.
- **refind**: Everything related to refractive indices.
- **rraman**: Anne Myers resonance Raman Fortran code translated to python.
- **uvvis**: Useful functions for analysis of UV-VIS spectra.
- **yeh_tm**: Yeh's 4x4 transfer matrix formalism including electric field calculations and the Needle algorithm.

Documentation
=============

A detailed documentation of all modules and functions with example codes can be found on my github pages: <http://ddietze.github.io/FSRStools>.

Installation
============

There is no installation script. Just copy the FSRStools folder somewhere where
python will find it or add the path to your PYTHONPATH variable.

Prerequisites for FSRStools are NumPy, SciPy and Matplotlib.

Licence
=======

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
