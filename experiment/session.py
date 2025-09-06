from psychopy.sound import Sound
from psychopy import sound
from IPython import embed
from psychopy import visual, core
from trial import (
    InstructionTrial,
    SingletonTrial,
    SingletonTrial_training,
    BlankTrial,
    DummyWaiterTrial,
    WaitStartTriggerTrial,
    OutroTrial,
)
from pathlib import Path
import yaml
import os.path as op
import numpy as np
from stimuli import (
    CueStimulusArray,
    SweepingBarStimulus,
    FixationStimulus,
    TargetStimulusArray,
    BackgroundCircle,
)
from exptools2.core import Session, PylinkEyetrackerSession
import psychopy

psychopy.prefs.hardware["audioLib"] = ["ptb", "pyo", "pygame"]


class SingletonSession(PylinkEyetrackerSession):

    def __init__(
        self,
        output_str,
        subject=None,
        session=None,
        output_dir=None,
        settings_file=None,
        run=None,
        eyetracker_on=False,
        calibrate_eyetracker=False,
        n_trials=10,
    ):

        super().__init__(
            output_str,
            output_dir=output_dir,
            settings_file=settings_file,
            eyetracker_on=eyetracker_on,
        )
        self.mri_trigger = "t"
        self.show_eyetracker_calibration = calibrate_eyetracker
        self.stimulus_shift = self.settings["experiment"]["stimulus_shift"]

        self.instructions = yaml.safe_load(
            (Path(__file__).parent / "instructions.yml").read_text()
        )
        print(self.instructions)

        self.settings["subject"] = subject
        self.settings["session"] = session
        self.settings["run"] = run
        # self.settings['n_trials'] = n_trials

        self.eccentricity_stimuli = self.settings["experiment"].get(
            "eccentricity_stimulus", 5
        )
        self.size_stimuli = self.settings["experiment"].get("size_stimuli", 1)
        self.radius_bar_aperture = self.eccentricity_stimuli - self.size_stimuli / 1.8

        self.fixation_dot = FixationStimulus(
            self.win,
            size=self.settings["experiment"]["size_fixation"],
            position=(0, self.stimulus_shift),
        )
        self.sweeping_bars = SweepingBarStimulus(
            self.win,
            session=self,
            speed=self.settings["bar_stimulus"]["speed"],
            fov_size=self.radius_bar_aperture * 2,
            #  bar_width=self.settings['bar_stimulus']['bar_width'],)
            bar_width=(self.radius_bar_aperture * 2) / 8,
            rest_duration=self.settings["bar_stimulus"]["rest_duration"],
            break_duration=self.settings["bar_stimulus"]["break_duration"],
        )
        self.backgroundcircle = BackgroundCircle(
            self.win,
            session=self,
            fov_size=self.radius_bar_aperture * 2,
        )

        self.target_stimuli = TargetStimulusArray(
            self.win,
            eccentricity=self.eccentricity_stimuli,
            stimulus_size=self.size_stimuli,
            stimulus_shift=self.stimulus_shift,
        )
        self.cue_stimuli = CueStimulusArray(
            self.win, self.eccentricity_stimuli, self.size_stimuli
        )

        self.correct_stimulus = visual.TextStim(
            self.win,
            text="v",
            color="green",
            height=self.settings["experiment"]["size_fixation"],
            pos=(0, self.stimulus_shift),
        )
        self.error_stimulus = visual.TextStim(
            self.win,
            text="x",
            color="red",
            height=self.settings["experiment"]["size_fixation"] * 2.0,
            pos=(0, self.stimulus_shift),
        )

        self.rt_clock = core.Clock()

        self.soundfile = str(Path(__file__).parent / "beep.wav")
        self.beep = Sound(str(self.soundfile))
        self.beep.setSound(800, secs=0.02)
        # self.beep = Sound('A', secs=0.1)
        self.beep.play()
        core.wait(0.5)
        self.beep.stop()
        self.beep.play()
        core.wait(0.02)
        self.beep.stop()
        self.beep_count = 0

    def run(self):
        """Runs experiment."""
        if self.eyetracker_on and self.show_eyetracker_calibration:
            self.calibrate_eyetracker()

        self.start_experiment()

        if self.eyetracker_on:
            self.start_recording_eyetracker()

        for trial in self.trials:
            trial.run()

        self.close()

    def create_trials(self, most_likely_distractor_location, include_instructions=True):
        """Create trials."""

        def resolve_image_path(path):
            if path is None:
                return None
            # Make absolute path relative to the current script
            base_dir = op.dirname(op.abspath(__file__))
            abs_path = op.join(base_dir, path)
            return abs_path if op.exists(abs_path) else None

        if include_instructions:
            if most_likely_distractor_location == 10:
                instruction_entries = [
                    self.instructions["intro"],
                    self.instructions["example1"],
                    self.instructions["example2"],
                    self.instructions["example3"],
                    self.instructions["fix"],
                    self.instructions["summary"],
                    self.instructions["reminder"],
                ]

                instruction_trials = []
                for i, entry in enumerate(instruction_entries):
                    # Handle either plain string or dict with 'text' and optional 'image'
                    if isinstance(entry, dict):
                        text = entry["text"].format(run=self.settings["run"])
                        image_path = resolve_image_path(entry.get("image", None))
                    else:
                        text = entry.format(run=self.settings["run"])
                        image_path = None

                    if entry == self.instructions["example1"]:
                        instruction_trials.append(
                            InstructionTrial(
                                self,
                                i,
                                txt=text,
                                image_path=image_path,
                                bottom_txt="Press left to continue",
                                keys=["left"],
                            )
                        )
                    elif entry == self.instructions["example2"]:
                        instruction_trials.append(
                            InstructionTrial(
                                self,
                                i,
                                txt=text,
                                image_path=image_path,
                                bottom_txt="Press up to continue",
                                keys=["up"],
                            )
                        )
                    else:
                        instruction_trials.append(
                            InstructionTrial(self, i, txt=text, image_path=image_path)
                        )

                self.trials = instruction_trials
            else:
                instruction_entries = [
                    self.instructions["return"],
                ]

                instruction_trials = []
                for i, entry in enumerate(instruction_entries):
                    # Handle either plain string or dict with 'text' and optional 'image'
                    if isinstance(entry, dict):
                        text = entry["text"].format(run=self.settings["run"])
                        image_path = resolve_image_path(entry.get("image", None))
                    else:
                        text = entry.format(run=self.settings["run"])
                        image_path = None

                    instruction_trials.append(
                        InstructionTrial(self, i, txt=text, image_path=image_path)
                    )

                self.trials = instruction_trials
        else:
            self.trials = []

        possible_itis = self.settings["durations"]["iti"]
        n_trials = self.settings["design"]["n_trials"]

        indices = [1, 3, 5, 7, 10]

        # for the dot version
        if most_likely_distractor_location < 10:
            indices.remove(most_likely_distractor_location)
            indices.insert(0, most_likely_distractor_location)
            t_d_locs = [
                (t, d)
                for t in [0, 0, 0, 0, 1, 2, 3]
                for d in [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]
                if t != d
            ] + [(t, d) for t in [0, 1, 2, 3] for d in [4]] * 3
        else:
            t_d_locs = [
                (t, d)
                for t in [0, 1, 2, 3]
                for d in [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
                if t != d
            ] + [(t, d) for t in [0, 1, 2, 3] for d in [4]] * 3

        # #for present/absent version
        # t_present_d_present = [(t,d) for t in [0,1,2,3] for d in [0,0,0,0,0,0,0,1,2,3] if t != d]
        # t_present_d_absent = [(t,4) for t in [0,1,2,3]] * 2
        # t_absent_d_present = [(4,d) for d in [0,0,0,0,0,0,0,1,2,3]] * 2
        # t_absent_d_absent = [(4,4)] * 2
        # t_d_locs = t_present_d_present + t_present_d_absent + t_absent_d_present + t_absent_d_absent

        np.random.shuffle(t_d_locs)

        # Assert n_trials is multiple of possible_itis
        assert (
            n_trials % len(possible_itis) == 0
        ), "n_trials should be multiple of possible itis"
        assert (
            n_trials % len(t_d_locs) == 0
        ), "n_trials should be multiple of possible tar/dist combinations"

        itis = np.tile(possible_itis, n_trials // len(possible_itis))
        np.random.shuffle(itis)

        print("???", self.settings["durations"].get("blank", 1))

        # If practice, use different trial function which includes eye deviation beeps
        if self.settings["session"] == 1:
            self.trials.append(
                OutroTrial(
                    session=self,
                    trial_nr=0,
                    phase_durations=[
                        15,
                        0.10,
                    ],
                    phase_names=["intro_dummy_scan", "start_exp"],
                    draw_each_frame=False,
                )
            )
            for ix, iti in enumerate(itis):
                self.trials.append(
                    SingletonTrial_training(
                        self,
                        ix + 1,
                        iti=iti,
                        distractor_location=indices[t_d_locs[ix][1]],
                        target_location=indices[t_d_locs[ix][0]],
                        most_likely_distractor_location=most_likely_distractor_location,
                    )
                )

            self.trials.append(
                OutroTrial(
                    session=self,
                    trial_nr=ix + 2,
                    phase_durations=[
                        self.settings["durations"]["blank"],
                        0.10,
                    ],
                    phase_names=["outro_dummy_scan", "end_exp"],
                    draw_each_frame=False,
                )
            )

            # show either a break screen or the end of experiment screen. Assumes 6 runs per session
            if self.settings["run"] == 6:
                entry = self.instructions["fin"]
                text = entry.format(run=self.settings["run"])
                self.trials.append(
                    InstructionTrial(self, ix + 3, txt=text, image_path=None)
                )

            else:
                entry = self.instructions["break"]
                text = entry.format(run=self.settings["run"])
                self.trials.append(
                    InstructionTrial(self, ix + 3, txt=text, image_path=None)
                )

        else:
            dummy_trial = DummyWaiterTrial(
                session=self,
                trial_nr=0,
                phase_durations=[np.inf, self.settings["durations"]["blank"]],
                phase_names=["start_exp", "intro_dummy_scan"],
                draw_each_frame=False,
            )

            start_trial = WaitStartTriggerTrial(
                session=self,
                trial_nr=0,
                phase_durations=[np.inf],
                draw_each_frame=False,
            )
            self.trials.append(dummy_trial)
            self.trials.append(start_trial)
            for ix, iti in enumerate(itis):
                self.trials.append(
                    SingletonTrial(
                        self,
                        ix + 1,
                        iti=iti,
                        distractor_location=indices[t_d_locs[ix][1]],
                        target_location=indices[t_d_locs[ix][0]],
                        most_likely_distractor_location=most_likely_distractor_location,
                    )
                )

            self.trials.append(
                OutroTrial(
                    session=self,
                    trial_nr=ix + 2,
                    phase_durations=[
                        self.settings["durations"]["blank"],
                        0.10,
                    ],
                    phase_names=["outro_dummy_scan", "end_exp"],
                    draw_each_frame=False,
                )
            )
