import sys
sys.path.append('C:\\Users\\SL_flicker\\AppData\\Roaming\\Python\\Python311\\site-packages')

from psychopy.core import Clock
from psychopy import visual
from exptools2.core import Trial
from stimuli import TargetStimulusArray, FixationStimulus, SweepingBarStimulus
from psychopy import event
from psychopy import logging
import numpy as np
import os
from utils import get_output_dir_str, get_settings
from session import SingletonSession
import argparse

def main(subject, session, run, settings='default', calibrate_eyetracker=False, most_likely_distractor_location=5):


    output_dir, output_str = get_output_dir_str(subject, session, 'estimation_task', run)
    settings_fn, use_eyetracker = get_settings(settings)
    use_eyetracker = True
    
    run_session = SingletonSession(output_str=output_str, subject=subject, session=session,
                          output_dir=output_dir, settings_file=settings_fn, 
                          run=run, eyetracker_on=use_eyetracker,
                          calibrate_eyetracker=calibrate_eyetracker)

    run_session.create_trials(most_likely_distractor_location=most_likely_distractor_location)
    run_session.run()

#if __name__ == "__main__":
#    argparser = argparse.ArgumentParser()
#    argparser.add_argument('subject', type=str, help='Subject nr')
#    argparser.add_argument('session', type=str, help='Session')
#    argparser.add_argument('run', type=int, help='Run')
#    argparser.add_argument('most_likely_distractor_location', type=int, help='Where the distractor most likely appears', choices=[1, 3, 5, 7])
#    argparser.add_argument('--settings', type=str, help='Settings label', default='default')
#    argparser.add_argument('--calibrate_eyetracker', action='store_true', dest='calibrate_eyetracker')
#
#    args = argparser.parse_args()
#
#    main(args.subject, args.session, args.run, args.settings, calibrate_eyetracker=args.calibrate_eyetracker, most_likely_distractor_location=args.most_likely_distractor_location)
#    
if __name__ == "__main__":
    # Define the subject/session/run manually since PsychoPy does not pass command-line arguments
    subject = "1"  # Change as needed
    session = "A"  # Change as needed
    run = 1
    most_likely_distractor_location = 5
    settings = "default"
    calibrate_eyetracker = True  # Change to True if needed
    eyetracker_on = True  

    # Run the main experiment function
    main(subject, session, run, settings=settings, calibrate_eyetracker=calibrate_eyetracker, most_likely_distractor_location=most_likely_distractor_location)
    
#main(subject=1, session=1, run=1, most_likely_distractor_location=5, settings=default, calibrate_eyetracker=calibrate_eyetracker)
#main(subject="1", session="1", run=A)
#