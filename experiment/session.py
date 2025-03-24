from exptools2.core import Session, PylinkEyetrackerSession
from stimuli import CueStimulusArray, SweepingBarStimulus, FixationStimulus, TargetStimulusArray
import numpy as np
import os.path as op
import yaml
from pathlib import Path
from trial import InstructionTrial, SingletonTrial
from psychopy import visual, core

class SingletonSession(PylinkEyetrackerSession):

    def __init__(self, output_str, subject=None, session=None, output_dir=None, settings_file=None, run=None, eyetracker_on=False, calibrate_eyetracker=False,
                 n_trials=10):

        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on)

        self.show_eyetracker_calibration = calibrate_eyetracker

        self.instructions = yaml.safe_load((Path(__file__).parent / "instructions.yml").read_text())
        print(self.instructions)

        self.settings['subject'] = subject
        self.settings['session'] = session
        self.settings['run'] = run
        # self.settings['n_trials'] = n_trials

        self.eccentricity_stimuli = self.settings['experiment'].get('eccentricity_stimulus', 5)
        self.size_stimuli = self.settings['experiment'].get('size_stimuli', 1)
        self.radius_bar_aperture = self.eccentricity_stimuli - self.size_stimuli


        self.fixation_dot = FixationStimulus(self.win, size=self.settings['experiment']['size_fixation'])
        self.sweeping_bars = SweepingBarStimulus(self.win, session=self,
                                                 speed=self.settings['bar_stimulus']['speed'], fov_size=self.radius_bar_aperture * 2,
                                                 bar_width=self.settings['bar_stimulus']['bar_width'],)

        self.target_stimuli = TargetStimulusArray(self.win, eccentricity=self.eccentricity_stimuli, stimulus_size=self.size_stimuli)
        self.cue_stimuli = CueStimulusArray(self.win, self.eccentricity_stimuli, self.size_stimuli)

        self.correct_stimulus = visual.TextStim(self.win, text='v', color='green', height=self.settings['experiment']['size_fixation'])
        self.error_stimulus = visual.TextStim(self.win, text='x', color='red', height=self.settings['experiment']['size_fixation']*2.)

        self.rt_clock = core.Clock()

    def run(self):
        """ Runs experiment. """
        if self.eyetracker_on and self.show_eyetracker_calibration:
            self.calibrate_eyetracker()

        self.start_experiment()

        if self.eyetracker_on:
            self.start_recording_eyetracker()
        for trial in self.trials:
            trial.run()

        self.close()


    def create_trials(self, most_likely_distractor_location, include_instructions=False):
        """Create trials."""


        instruction_trials = [InstructionTrial(self, 0, self.instructions['intro'].format(run=self.settings['run']))]

        
        self.trials = instruction_trials

        if not include_instructions:
            self.trials = []

        possible_itis = self.settings['durations']['iti']
        n_trials = self.settings['design']['n_trials']

        # Assert n_trials is multiple of possible_itis
        assert n_trials % len(possible_itis) == 0, 'n_trials should be multiple of possible itis'
        itis = np.tile(possible_itis, n_trials // len(possible_itis))
        np.random.shuffle(itis)



        for ix, iti in enumerate(itis):
            self.trials.append(SingletonTrial(self, ix+1, iti=iti, most_likely_distractor_location=most_likely_distractor_location))