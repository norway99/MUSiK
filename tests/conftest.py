import os
import sys
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")

# MUSiK vendors k-Wave as a git submodule rather than declaring it as a normal
# package.  Make a source checkout testable without requiring an editable
# install of the submodule.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KWAVE_ROOT = REPOSITORY_ROOT / "k-wave-python"
if KWAVE_ROOT.is_dir():
    sys.path.insert(0, str(KWAVE_ROOT))

