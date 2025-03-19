from psychopy.core import Clock
from psychopy import visual
from exptools2.core import Trial
from stimuli import TargetStimulusArray, FixationStimulus, SweepingBarStimulus
from psychopy import event
from psychopy import logging
import numpy as np
import os


win = visual.Window([1200, 1200], fullscr=False, monitor='testMonitor', units='deg', allowStencil=True)
logging.console.setLevel(logging.DEBUG)

stimulus_array = TargetStimulusArray(win, 'red')

bar_stimulus = SweepingBarStimulus(win, speed=5, fov_size=18)

fixation = FixationStimulus(win)

start_time = Clock()

last_phase_shift = start_time.getTime()

# Show stimulus_array for 5 seconds
while start_time.getTime() < 60:

    # if start_time.getTime() - last_phase_shift > 1:
    #     last_phase_shift = start_time.getTime()
    #     bar_stimulus.phase_forward()

    bar_stimulus.draw()
    stimulus_array.draw()
    fixation.draw()
    win.flip()



win.close()

