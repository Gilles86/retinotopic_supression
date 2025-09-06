from exptools2.core import Trial
from psychopy.visual import TextStim, ImageStim
import numpy as np
from collections import deque
from psychopy import core
import os.path as op
from psychopy.core import getTime


class InstructionTrial(Trial):

    def __init__(
        self,
        session,
        trial_nr,
        txt,
        bottom_txt=None,
        image_path=None,
        keys=None,
        phase_durations=None,
        phase_names=None,
        **kwargs,
    ):

        self.keys = keys

        if phase_durations is None:
            phase_durations = [0.5, np.inf]

        if phase_names is None:
            phase_names = ["instruction"] * len(phase_durations)

        super().__init__(
            session,
            trial_nr,
            phase_durations=phase_durations,
            phase_names=phase_names,
            **kwargs,
        )

        txt_height = self.session.settings["various"].get("text_height")
        txt_width = self.session.settings["various"].get("text_width")
        txt_color = self.session.settings["various"].get("text_color")

        if image_path:
            self.text = TextStim(
                session.win,
                txt,
                pos=(-4.0, 0.0),
                height=txt_height,
                wrapWidth=txt_width,
                color=txt_color,
            )
        else:
            self.text = TextStim(
                session.win,
                txt,
                pos=(0.0, 0.0),
                height=txt_height,
                wrapWidth=txt_width,
                color=txt_color,
            )

        print(self.text)
        print(txt)

        if bottom_txt is None:
            bottom_txt = "Press any button to continue"

        self.text2 = TextStim(
            session.win,
            bottom_txt,
            pos=(0.0, -6.0),
            height=txt_height,
            wrapWidth=txt_width,
            color=txt_color,
        )

        self.image = None
        print(f"Trying to load image: {image_path}")
        if image_path:
            print("Exists?", op.exists(image_path))
        else:
            print("No image provided.")

        if image_path is not None and op.exists(image_path):
            self.image = ImageStim(
                session.win, image=image_path, pos=(6, 0), size=(10, 10), units="deg"
            )

    def get_events(self):

        events = Trial.get_events(self)

        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()

    def draw(self):
        if self.image:
            self.image.draw()
        self.text.draw()
        self.text2.draw()


class SingletonTrial(Trial):

    def __init__(
        self,
        session,
        trial_nr,
        iti,
        distractor_location=None,
        target_location=None,
        distractor_color=None,
        target_orientation=None,
        dot_presence=None,
        most_likely_distractor_location=None,
        **kwargs,
    ):

        trial_start_duration = session.settings["durations"].get(
            "trial_start", 1)
        cue_duration = session.settings["durations"].get("trial_wait", 1)
        target_duration = session.settings["durations"].get("target", 1)
        feedback_duration = session.settings["durations"].get("feedback", 1)

        phase_durations = [
            trial_start_duration,
            cue_duration,
            target_duration,
            feedback_duration,
            iti,
        ]
        phase_names = ["trial_start", "pre-target",
                       "target", "feedback", "iti"]

        super().__init__(
            session,
            trial_nr,
            phase_durations=phase_durations,
            phase_names=phase_names,
            **kwargs,
        )

        self.parameters["distractor_color"] = (
            np.random.choice(["red", "green"])
            if distractor_color is None
            else distractor_color
        )
        self.parameters["target_orientation"] = (
            np.random.choice([0.0, 90])
            if target_orientation is None
            else target_orientation
        )

        if distractor_location is None:
            locations = range(1, 8, 2)
            most_likely_distractor_p = self.session.settings["design"].get(
                "most_likely_distractor_p", 0.66
            )
            probabilities = [(1 - most_likely_distractor_p) / 3] * 4
            probabilities[locations.index(most_likely_distractor_location)] = (
                most_likely_distractor_p
            )
            self.parameters["distractor_location"] = np.random.choice(
                locations, p=probabilities
            )
        else:
            self.parameters["distractor_location"] = distractor_location

        if dot_presence is None:
            # to ensure always and only 2 dots on the possible target locations
            dot_presence = [False] * 8
            for i in np.random.choice([0, 2, 4, 6], 2, replace=False):
                dot_presence[i] = True
            for i in np.random.choice([1, 3, 5, 7], 2, replace=False):
                dot_presence[i] = True
            self.parameters["dot_presence"] = dot_presence

            # self.parameters['dot_presence'] = ([True] * 4) + ([False] * 4)
            # np.random.shuffle(self.parameters['dot_presence'])
        else:
            self.parameters["dot_presence"] = dot_presence

        if target_location is None:
            self.parameters["target_location"] = np.random.choice(
                [
                    i
                    for i in range(1, 8, 2)
                    if i != self.parameters["distractor_location"]
                ]
            )
        else:
            self.parameters["target_location"] = target_location

        self.parameters["correct"] = np.nan
        if most_likely_distractor_location == 10:
            self.parameters["HPL_distractor"] = None
        else:
            self.parameters["HPL_distractor"] = most_likely_distractor_location
        self.responded = False

        # for the dot version
        # print(self.parameters['dot_presence'])
        # print(self.parameters['target_location'])
        self.parameters["correct_response"] = self.parameters["dot_presence"][
            self.parameters["target_location"]
        ]

        # for present/absent version
        # if self.parameters['target_location'] < 9:
        #     self.parameters['correct_response'] = True
        # else:
        #     self.parameters['correct_response'] = False

        self.stimulus_onset = None

    def draw(self):

        # #if cue that search is about to start
        # if self.phase == 0:
        #     self.session.fixation_dot.color = 'blue'
        # elif self.phase == 1:
        #     self.session.fixation_dot.color = 'white'

        self.session.fixation_dot.color = "white"

        if self.phase == 2:
            if self.stimulus_onset is None:
                self.stimulus_onset = self.session.clock.getTime()

            self.session.target_stimuli.draw()

        self.session.sweeping_bars.draw()

        if self.phase == 3:
            if (not self.responded) or (not self.parameters["correct"]):
                self.session.error_stimulus.draw()
            else:
                if self.session.settings["experiment"].get(
                    "show_correct_feedback", False
                ):
                    self.session.correct_stimulus.draw()
                else:
                    self.session.fixation_dot.draw()
        else:
            self.session.fixation_dot.draw()

    def run(self):
        self.setup_trial_stimuli()
        super().run()

    def setup_trial_stimuli(self):
        self.session.target_stimuli.setup(
            self.parameters["distractor_color"],
            self.parameters["target_orientation"],
            self.parameters["distractor_location"],
            self.parameters["target_location"],
            self.parameters["dot_presence"],
        )

    def get_events(self):
        events = super().get_events()
        keys = self.session.settings["experiment"]["keys"]

        if self.phase == 2:
            for key, t in events:
                if (not self.responded) and (
                    key in self.session.settings["experiment"]["keys"]
                ):
                    self.parameters["response"] = key
                    self.parameters["rt"] = t - self.stimulus_onset
                    self.parameters["correct"] = (
                        bool(keys.index(self.parameters["response"]))
                        == self.parameters["correct_response"]
                    )
                    self.responded = True


class SingletonTrial_training(SingletonTrial):
    def __init__(
        self,
        session,
        trial_nr,
        iti,
        distractor_location=None,
        target_location=None,
        distractor_color=None,
        target_orientation=None,
        dot_presence=None,
        most_likely_distractor_location=1,
        **kwargs,
    ):

        super().__init__(
            session,
            trial_nr,
            iti,
            distractor_location=distractor_location,
            target_location=target_location,
            distractor_color=distractor_color,
            target_orientation=target_orientation,
            dot_presence=dot_presence,
            most_likely_distractor_location=most_likely_distractor_location,
            **kwargs,
        )
        self.audio_played = False
        self.trial_frame_count = 0
        self.gaze_x = deque(maxlen=60)
        self.gaze_y = deque(maxlen=60)
        self.gaze_time = deque(maxlen=60)
        self.drift_x = deque(maxlen=60)
        self.drift_y = deque(maxlen=60)
        self.drift_times = deque(maxlen=60)
        self.drift_collecting = False

    def drift_correction_step(self):
        """Call this every frame during phase 1 to incrementally collect gaze for drift correction."""
        now = core.getTime()

        if not self.drift_collecting:
            self.drift_collecting = True
            self.drift_start_time = now
            self.drift_x.clear()
            self.drift_y.clear()
            self.drift_times.clear()
            return

        if now - self.drift_start_time < 0.2:  # still collecting
            el_smp = self.session.tracker.getNewestSample()
            if el_smp is None:
                return

            if el_smp.isLeftSample():
                gaze = el_smp.getLeftEye().getGaze()
            elif el_smp.isRightSample():
                gaze = el_smp.getRightEye().getGaze()
            else:
                return

            if gaze is not None:
                self.drift_x.append(gaze[0])
                self.drift_y.append(gaze[1])
                self.drift_times.append(now)
            return

        # Finished collection
        if len(self.drift_x) < 5:
            screen_center = np.array(self.session.win.size) / 2
            screen_center[1] -= self.session.pix_stimulus_shift
            self.drift = tuple(screen_center)
        else:
            self.drift = (np.mean(self.drift_x), np.mean(self.drift_y))

        self.drift_collecting = False  # mark done

    def check_fixation_windowed(self):
        if self.trial_frame_count % 2 != 0:
            return True
        el_smp = self.session.tracker.getNewestSample()
        if el_smp is None:
            return True

        if el_smp.isLeftSample():
            sample = el_smp.getLeftEye().getGaze()
        elif el_smp.isRightSample():
            sample = el_smp.getRightEye().getGaze()
        else:
            return True

        now = core.getTime() * 1000  # ms
        self.gaze_x.append(sample[0])
        self.gaze_y.append(sample[1])
        self.gaze_time.append(now)

        # Convert only necessary portion
        times = np.array(self.gaze_time)
        d_times = times - now
        idx = np.where(d_times > -30)[0]

        if idx.size < 2:
            return True

        x_ = np.array(self.gaze_x)[idx] - self.drift[0]
        y_ = np.array(self.gaze_y)[idx] - self.drift[1]
        angles = np.hypot(x_, y_) / self.session.pix_per_deg

        if np.all(angles > self.session.settings["various"]["gaze_threshold_deg"]):
            return False
        return True

    def draw(self):
        self.trial_frame_count += 1
        if self.phase == 0:
            self.session.fixation_dot.color = "white"
        elif self.phase == 1:
            self.session.fixation_dot.color = "white"
            if self.session.eyetracker_on:
                self.drift_correction_step()

        if self.phase == 2:
            if self.stimulus_onset is None:
                self.stimulus_onset = core.getTime()

            self.session.target_stimuli.draw()

            if (
                self.session.eyetracker_on
                and self.session.settings["various"]["eyemovements_alert"]
            ):
                fix_ok = self.check_fixation_windowed()
                if not fix_ok and not self.audio_played:
                    self.session.beep.play()
                    core.wait(0.03)
                    self.session.beep.stop()
                    self.audio_played = True
                    self.session.beep_count += 1

        self.session.sweeping_bars.draw()

        if self.phase == 3:
            if (not self.responded) or (not self.parameters["correct"]):
                self.session.error_stimulus.draw()
            else:
                if self.session.settings["experiment"].get(
                    "show_correct_feedback", False
                ):
                    self.session.correct_stimulus.draw()
                else:
                    self.session.fixation_dot.draw()
        else:
            self.session.fixation_dot.draw()


class BlankTrial(Trial):

    def __init__(self, session, trial_nr, **kwargs):

        blank_duration = session.settings["durations"].get("blank", 1)
        phase_durations = [blank_duration]
        phase_names = ["blank"]

        super().__init__(
            session,
            trial_nr,
            phase_durations=phase_durations,
            phase_names=phase_names,
            **kwargs,
        )

        self.parameters["correct"] = np.nan
        self.responded = False
        self.stimulus_onset = None

    def draw(self):
        self.session.backgroundcircle.draw()
        self.session.fixation_dot.color = "white"
        self.session.fixation_dot.draw()

    def run(self):
        super().run()


class DummyWaiterTrial(Trial):
    """Simple trial with text (trial x) and fixation."""

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations=None,
        phase_names=None,
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )

    def draw(self):
        self.session.backgroundcircle.draw()
        self.session.fixation_dot.color = "white"
        self.session.fixation_dot.draw()
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)
        if events:
            for key, t in events:
                if key == self.session.mri_trigger:
                    print(f"Got trigger: {key}")
                    print(self.session.mri_trigger)
                    print("phase: ", self.phase)
                    if self.phase == 0:
                        print("Got trigger in phase 0")
                        self.stop_phase()
                        #####################################################
                        # TRIGGER HERE
                        #####################################################
                        self.session.experiment_start_time = getTime()


class WaitStartTriggerTrial(Trial):
    def __init__(
        self,
        session,
        trial_nr,
        phase_durations=[np.inf],
        phase_names=["waiting_start_trigger"],
        draw_each_frame=False,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            draw_each_frame=draw_each_frame,
        )

    def draw(self):
        self.session.backgroundcircle.draw()
        self.session.fixation_dot.color = "white"
        self.session.fixation_dot.draw()
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)
        if events:
            for key, t in events:
                print(f"Got trigger: {key}")
                print(self.session.mri_trigger)
                print("phase: ", self.phase)
                if key == self.session.mri_trigger:
                    self.stop_phase()
                    self.session.experiment_start_time = getTime()


class OutroTrial(Trial):
    """Simple trial with only fixation cross."""

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )

    def draw(self):
        self.session.backgroundcircle.draw()
        self.session.fixation_dot.color = "white"
        self.session.fixation_dot.draw()
        self.session.win.flip()
