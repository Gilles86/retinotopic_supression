import psychopy
psychopy.prefs.hardware['audioLib'] = ['ptb', 'pyo', 'pygame']
from exptools2.core import Session, PylinkEyetrackerSession
import numpy as np
import os.path as op
import yaml
from pathlib import Path
from psychopy import visual, core
from IPython import embed

from psychopy import sound
from psychopy.sound import Sound 


print(psychopy.__version__)

soundfile = str(Path(__file__).parent / "beep.wav")
beep = Sound(str(soundfile))
beep.setSound(800, secs=0.1)
# self.beep = Sound('A', secs=0.1)
beep.play()
        