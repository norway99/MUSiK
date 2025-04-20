import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current)

__all__ = [
    file[:-3]
    for file in os.listdir(current)
    if file.endswith(".py") and file != "__init__.py"
]

kwave = os.path.dirname(os.path.realpath(f"{current}/../k-wave-python/kwave"))
sys.path.append(kwave)
