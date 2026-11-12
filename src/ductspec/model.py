"""The base model, named in one place.

Training, prediction and the token measurements all have to load the same
model, and the id was written out in five scripts. `DUCTSPEC_BASE_MODEL`
points them at a local copy instead of the Hub: a Kaggle model mount, or a
directory on a machine with no network. It changes where the weights are read
from, not which weights they are, so `HUB_ID` is what a run reports.
"""

from __future__ import annotations

import os

HUB_ID = "CohereLabs/tiny-aya-global"
BASE_MODEL = os.environ.get("DUCTSPEC_BASE_MODEL") or HUB_ID
