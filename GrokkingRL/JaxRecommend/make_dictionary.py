import glob
import json
import os
from typing import Any, Dict, Tuple
from absl import app, flags, logging
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS
_PLAYLISTS = flags.DEFINE_string("playlists", None, "Playlist json glob.")
