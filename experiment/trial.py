from exptools2.core import Trial
from psychopy.visual import TextStim
import numpy as np
from psychopy import core


class InstructionTrial(Trial):

    def __init__(self, session, trial_nr, txt, bottom_txt=None, keys=None, phase_durations=None, 
                 phase_names=None, **kwargs):

        self.keys = keys

        if phase_durations is None:
            phase_durations = [.5, np.inf]

        if phase_names is None:
            phase_names = ['instruction'] * len(phase_durations)

        super().__init__(session, trial_nr, phase_durations=phase_durations, phase_names=phase_names, **kwargs)

        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')
        txt_color = self.session.settings['various'].get('text_color')

        self.text = TextStim(session.win, txt,
                             pos=(0.0, 0.0), height=txt_height, wrapWidth=txt_width, color=txt_color)


        print(self.text)
        print(txt)

        if bottom_txt is None:
            bottom_txt = "Press any button to continue"

        self.text2 = TextStim(session.win, bottom_txt, pos=(
            0.0, -6.0), height=txt_height, wrapWidth=txt_width,
            color=txt_color)

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
        self.text.draw()
        self.text2.draw()


class SingletonTrial(Trial):

    def __init__(self, session, trial_nr, iti,
                distractor_location=None,
                target_location=None,
                distractor_color=None,
                target_orientation=None,
                dot_presence=None,
                  **kwargs):


        trial_start_duration = session.settings['durations'].get('trial_start', 1)
        cue_duration = session.settings['durations'].get('cue', 1)
        target_duration = session.settings['durations'].get('target', 1)
        feedback_duration = session.settings['durations'].get('feedback', 1)


        phase_durations = [trial_start_duration, cue_duration, target_duration, feedback_duration, iti]
        phase_names = ['trial_start', 'cue', 'target', 'feedback', 'iti']

        super().__init__(session, trial_nr, phase_durations=phase_durations, phase_names=phase_names, **kwargs) 

        self.parameters['distractor_color'] = np.random.choice(['red', 'green']) if distractor_color is None else distractor_color
        self.parameters['target_orientation'] = np.random.choice([0.0, 90]) if target_orientation is None else target_orientation
        self.parameters['distractor_location'] = np.random.choice(range(8)) if distractor_location is None else distractor_location
        self.parameters['dot_presence'] = np.random.choice([True, False], 8) if dot_presence is None else dot_presence

        if target_location is None:
            self.parameters['target_location'] = np.random.choice([i for i in range(8) if i != self.parameters['distractor_location']])
        else:
            self.parameters['target_location'] = target_location

        self.parameters['correct'] = np.nan
        self.responded = False
        self.parameters['correct_response'] = self.parameters['dot_presence'][self.parameters['target_location']]
        self.stimulus_onset = None

    
    def draw(self):

        if self.phase == 1:
            self.session.cue_stimuli.draw()
        elif self.phase == 2:
            if self.stimulus_onset is None:
                self.stimulus_onset = core.getTime()

            self.session.target_stimuli.draw()

        self.session.sweeping_bars.draw()

        if self.phase == 3:
            if (not self.responded) or (not self.parameters['correct']):
                self.session.error_stimulus.draw()
            else:
                self.session.correct_stimulus.draw()
        else:
            self.session.fixation_dot.draw()

    def run(self):
        self.setup_trial_stimuli()
        super().run()
    
    def setup_trial_stimuli(self):
        self.session.target_stimuli.setup(self.parameters['distractor_color'], self.parameters['target_orientation'],
                                          self.parameters['distractor_location'], self.parameters['target_location'],
                                          self.parameters['dot_presence'])

    def get_events(self):
        events = super().get_events()
        keys = self.session.settings['experiment']['keys']

        if self.phase == 2:
            for key, t in events:
                if (not self.responded) and (key in self.session.settings['experiment']['keys']):
                    self.parameters['response'] = key
                    self.parameters['rt'] = t - self.stimulus_onset
                    self.parameters['correct'] = bool(keys.index(self.parameters['response'])) == self.parameters['correct_response']
                    self.responded = True