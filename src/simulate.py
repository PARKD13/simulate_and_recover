import numpy as np
import pandas as pd
import time
from src.ez_diffusion import EZ_diffusion

class simulationResult:
    """Class to store simulation results."""
    def __init__(self, history, time_steps):
        self.history = history
        self.time_steps = time_steps