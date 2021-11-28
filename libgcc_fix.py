"""
Fixes weird `libgcc_s.so.1 must be installed for pthread_cancel to work` error during debugging.

This error only shows up when importing stuff from this repo into another directory.

Make sure you import this first in the other files.  SEE: https://stackoverflow.com/a/65908383/11163122
>>> import libgcc_fix
"""

import ctypes

libgcc_s = ctypes.CDLL("libgcc_s.so.1")
