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


def main(subject, session, run, settings="default", use_eyetracker=True):
    """
    start the experiment by providing three variables:
    subject number (between 1 and infinity)
    session number (1 = practice, 2 = scanning, 3 = scanning)

    thats all folks!
    we will automatically find what HP condition to use

    cab also specify settings document to use
    as well as whether to use eyetracker.
    Automatically calibrates every block if used.
    """
    eyetracker_on = True
    calibrate_eyetracker = True

    hp_list = [10, 10, 10, 10]
    # hp_list = [3,7] ## for our pilot experiments

    if session > 1:
        hp_list = (
            [1, 5, 3, 7],
            [1, 5, 7, 3],
            [3, 7, 5, 1],
            [5, 1, 7, 3],
            [3, 7, 1, 5],
            [7, 3, 1, 5],
            [5, 1, 3, 7],
            [7, 3, 5, 1],
        )[subject % 8 - 1]
        eyetracker_on = False
        calibrate_eyetracker = False

    most_likely_distractor_location = hp_list[(
        run - 1) // 3 + (session - 2) * 2]
    output_dir, output_str = get_output_dir_str(
        subject, session, "ret_sup", run)
    if os.path.exists(output_dir):
        raise ValueError(
            f"""\n
            ================================\n
            =========WATCH OUT!!!===========\n
            Output directory already exists.\n 
            ====Please check your input.====\n
            ================================\n"""
        )
    settings_fn, use_eyetracker = get_settings(settings)
    include_instructions = False
    print(most_likely_distractor_location)
    # if (session == 1) & (run ==1):
    #     include_instructions = True
    if run == 1:
        include_instructions = True

    run_session = SingletonSession(
        output_str=output_str,
        subject=subject,
        session=session,
        output_dir=output_dir,
        settings_file=settings_fn,
        run=run,
        eyetracker_on=eyetracker_on,
        calibrate_eyetracker=calibrate_eyetracker,
    )

    run_session.create_trials(
        most_likely_distractor_location=most_likely_distractor_location,
        include_instructions=include_instructions,
    )
    run_session.run()
    print(f"Eyemovements: {run_session.beep_count} trials")


# if __name__ == "__main__":
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('subject', type=str, help='Subject nr')
#     argparser.add_argument('session', type=str, help='Session')
#     argparser.add_argument('run', type=int, help='Run')
#     argparser.add_argument('most_likely_distractor_location', type=int, help='Where the distractor most likely appears', choices=[1, 3, 5, 7])
#     argparser.add_argument('--settings', type=str, help='Settings label', default='default')
#     # argparser.add_argument('--calibrate_eyetracker', action='store_true', dest='calibrate_eyetracker')

#     args = argparser.parse_args()

#     main(args.subject, args.session, args.run, args.settings, calibrate_eyetracker=True, most_likely_distractor_location=args.most_likely_distractor_location)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("subject", type=int, help="Subject nr")
    argparser.add_argument("session", type=int, help="Session")
    argparser.add_argument("run", type=int, help="run")
    argparser.add_argument(
        "--settings", type=str, help="Settings label", default="default"
    )
    argparser.add_argument(
        "--use_eyetracker", action="store_true", help="Enable eyetracker"
    )

    args = argparser.parse_args()

    main(args.subject, args.session, args.run,
         args.settings, args.use_eyetracker)
