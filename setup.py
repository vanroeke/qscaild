#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#  Copyright (C) 2016-2019 Ambroise van Roekeghem <ambroise.vanroekeghem@gmail.com>
#  Copyright (C) 2016-2019 Jesús Carrete Montaña <jcarrete@gmail.com>
#  Copyright (C) 2016-2019 Natalio Mingo Bisquert <natalio.mingo@cea.fr>
#
#  This file is part of qSCAILD.
#
#  qSCAILD is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  qSCAILD is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with qSCAILD.  If not, see <https://www.gnu.org/licenses/>.

import numpy
from distutils.core import setup
from distutils.extension import Extension

# Add the location of the "spglib/spglib.h" to this list if necessary.
# Example: INCLUDE_DIRS=["/home/user/local/include"]
INCLUDE_DIRS = ["/home/sousaw/.local/include"]
# Add the location of the spglib shared library to this list if necessary.
# Example: LIBRARY_DIRS=["/home/user/local/lib"]
LIBRARY_DIRS = ["/home/sousaw/.local/lib"]

# Set USE_CYTHON to True if you want include the cythonization in your build
# process.
USE_CYTHON = True

ext = ".pyx" if USE_CYTHON else ".c"

extensions = [
    Extension(
        "thirdorder_core", ["thirdorder_core" + ext],
        include_dirs=[numpy.get_include()] + INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=LIBRARY_DIRS,
        libraries=["symspg"])
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(name="thirdorder", ext_modules=extensions)
