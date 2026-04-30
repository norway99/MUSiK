"""
MUSiK: Multi-transducer Ultrasound Simulations in K-wave

A Python library for simulating multi-transducer ultrasound systems using the k-Wave toolbox.
It provides tools for creating phantoms, configuring transducers, running simulations, and processing results.
"""

import os
import sys

__version__ = "0.1.0"
__author__ = "Trevor Chan, Aparna Nair-kanneganti"
__email__ = "tjchan@seas.upenn.edu"
__license__ = "MIT"
__description__ = "Multi-transducer Ultrasound Simulations in K-wave"
__url__ = "https://github.com/norway99/MUSiK"

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current)

__all__ = [
    file[:-3]
    for file in os.listdir(current)
    if file.endswith(".py") and file != "__init__.py"
]

kwave = os.path.dirname(os.path.realpath(f"{current}/../k-wave-python/kwave"))
sys.path.append(kwave)
