from psychopy import visual
import numpy as np

class TargetStimulus(object):

    def __init__(self, win, target, pos, size, color, ori):
        self.win = win
        self.target = target  # True or False
        self._pos = pos  # Store the position in _pos (used in setter)
        self._size = size  # Store size in _size (used in setter)
        self._color = color  # Store color in _color (used in setter)
        self._ori = ori  # Store orientation in _ori (used in setter)

        # Create visual elements
        self.rectangle = visual.Rect(
            win=win, pos=self._pos, width=self._size, height=self._size / 3.0,
            fillColor=self._color, lineColor=None, ori=self._ori
        )
        self.dot = visual.Circle(
            win=win, pos=self._pos, radius=self._size / 8, fillColor='black', lineColor=None
        )

    def update(self, target=None, color=None, ori=None, pos=None, size=None):
        """
        Dynamically updates stimulus properties without recreating the object.

        Parameters (all optional):
        - target: Whether the stimulus is the target (True/False).
        - color: The new color ('red' or 'green').
        - ori: The new orientation (0 or 90).
        - pos: The new position (tuple of x, y).
        - size: The new size (float, in degrees).
        """
        if target is not None:
            self.target = target
        if color is not None:
            self.color = color
        if ori is not None:
            self.ori = ori
        if pos is not None:
            self.pos = pos
        if size is not None:
            self.size = size

    def draw(self):
        """Draws the rectangle and dot (if target) on the screen."""
        self.rectangle.draw()
        if self.target:
            self.dot.draw()

    # Property for orientation
    @property
    def ori(self):
        return self._ori

    @ori.setter
    def ori(self, value):
        self._ori = value
        self.rectangle.ori = value  # Update visual property

    # Property for color
    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value
        self.rectangle.fillColor = value  # Update visual property

    # Property for position
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        self.rectangle.pos = value  # Update visual property
        self.dot.pos = value  # Keep dot aligned with the rectangle

    # Property for size
    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value
        self.rectangle.width = value
        self.rectangle.height = value / 3.0
        self.dot.radius = value / 8  # Scale dot size with the rectangle

class TargetStimulusArray(object):

    n_objects = 8

    def __init__(self, win, eccentricity=9, stimulus_size=1):
        self.win = win
        self.eccentricity = eccentricity
        self.stimulus_size = stimulus_size
        self.stimuli = []
        self.positions = self.get_positions(eccentricity)

        # Initialize stimuli with default settings
        for ix in range(self.n_objects):
            self.stimuli.append(TargetStimulus(win, target=False, pos=self.positions[ix], 
                                               size=stimulus_size, color='green', ori=np.pi/2.))

    def setup(self, distractor_color, target_orientation, distractor_location, target_location,
              dot_presence):
        """
        Dynamically updates the stimuli configuration.

        Parameters:
        - distractor_color: The color of the distractor ('red' or 'green').
        - target_orientation: The orientation of the target (0 or 90).
        - distractor_location: The index (0-7) of the distractor.
        - target_location: The index (0-7) of the target.
        """
        assert distractor_color in ['red', 'green']
        assert 0 <= distractor_location < self.n_objects
        assert 0 <= target_location < self.n_objects

        other_color = 'red' if distractor_color == 'green' else 'green'
        other_orientation = 90. if target_orientation == 0.0 else 0.0

        print(f'Setting up trial: distractor={distractor_location},\ntarget={target_location},\ndistractor_color={distractor_color},\ntarget_orientation={target_orientation}\n other_color={other_color},\nother_orientation={other_orientation}')

        for ix, stimulus in enumerate(self.stimuli):
            if ix == distractor_location:
                stimulus.update(target=dot_presence[ix], color=distractor_color, ori=other_orientation)
            elif ix == target_location:
                stimulus.update(target=dot_presence[ix], color=other_color, ori=target_orientation)
            else:
                stimulus.update(target=dot_presence[ix], color=other_color, ori=other_orientation)

    def get_positions(self, eccentricity):
        """Returns evenly spaced positions in a circular layout."""
        positions = []
        for i in range(self.n_objects):
            angle = i * 360 / self.n_objects + (360 / (2 * self.n_objects))
            x = eccentricity * np.cos(np.radians(angle))
            y = eccentricity * np.sin(np.radians(angle))
            positions.append((x, y))
        return positions

    def draw(self):
        """Draws all stimuli on the screen."""
        for stimulus in self.stimuli:
            stimulus.draw()

from psychopy import visual

class FixationStimulus:
    def __init__(self, win, position=(0, 0), size=0.5, color='black', cross_color='white', cross_thickness=2):
        """
        A fixation stimulus with a dot and optional cross.

        :param win: PsychoPy window object
        :param position: (x, y) position in deg
        :param size: Diameter of the central dot in deg
        :param color: Color of the central dot (default 'black')
        :param cross_color: Color of the cross lines (default 'white')
        :param cross_thickness: Thickness of the cross lines
        """
        self.win = win
        self.position = position
        self.size = size
        self.color = color

        # Central fixation dot
        self.dot = visual.Circle(
            win=win, pos=self.position, radius=self.size / 2, fillColor=self.color, lineColor=None
        )

        # Cross lines for better visibility
        cross_length = self.size * 2  # Length of the cross arms
        self.h_line = visual.Line(
            win=win, start=(position[0] - cross_length / 2, position[1]), 
            end=(position[0] + cross_length / 2, position[1]),
            lineColor=cross_color, lineWidth=cross_thickness
        )
        self.v_line = visual.Line(
            win=win, start=(position[0], position[1] - cross_length / 2), 
            end=(position[0], position[1] + cross_length / 2),
            lineColor=cross_color, lineWidth=cross_thickness
        )

    def draw(self, cross=True):
        """
        Draw the fixation stimulus.

        :param cross: Whether to draw the cross lines (default: True)
        """
        if cross:
            self.h_line.draw()
            self.v_line.draw()
        self.dot.draw()


from psychopy import visual, core, event
import numpy as np
from psychopy import plugins
plugins.loadPlugin('psychopy_visionscience')


from psychopy import visual, core, event

class SweepingBarStimulus:
    def __init__(self, win, fov_size=20, bar_width=2, speed=2, rest_duration=2):
        """
        Creates a sweeping checkerboard bar stimulus that properly rotates for vertical motion.

        :param win: PsychoPy window
        :param fov_size: Diameter of the circular aperture (deg)
        :param bar_width: Width of the sweeping bar (deg)
        :param speed: Speed of bar movement (deg/sec)
        :param rest_duration: Duration (sec) of rest between sweeps
        """
        self.win = win
        self.fov_size = fov_size
        self.bar_width = bar_width
        self.speed = speed
        self.rest_duration = rest_duration
        self.clock = core.Clock()
        self.flicker_clock = core.Clock()
        self.contrast = 1.0

        # ✅ Define paradigm: movement directions + rest periods
        self.directions = ["right", "rest", "left", "rest", "down", "rest", "up", "rest"]
        self.current_direction_index = 0  # Start with the first direction
        self.sweep_clock = core.Clock()  # Keeps track of how long we've been in the current sweep

        # ✅ Compute how long a full sweep takes
        self.sweep_duration = fov_size / speed  # Ensure full traversal of FOV

        # ✅ Create the **rectangular bar** with a checkerboard pattern
        self.bar = visual.GratingStim(
            win, tex="sqrXsqr", mask=None,
            size=(bar_width, fov_size), sf=self.bar_width * 1.5, contrast=self.contrast,
            interpolate=False, units="deg", ori=0  # Default orientation
        )

        # ✅ Create a **circular aperture**
        self.aperture = visual.Aperture(win, size=fov_size)
        self.aperture.enabled = False  # Ensures it applies only when needed

        self.background_circle = visual.Circle(
            win, radius=fov_size / 2.0, fillColor=None, lineColor='darkgray', lineWidth=5.0, pos=(0, 0)
        )


    def switch_direction(self):
        """ Switches to the next movement direction in the paradigm. """
        self.current_direction_index = (self.current_direction_index + 1) % len(self.directions)
        self.sweep_clock.reset()  # Reset the clock for the new sweep

    def update_position(self):
        """ Moves the bar across the screen OR enters a rest period. """
        direction = self.directions[self.current_direction_index]

        if direction == "rest":
            self.bar.opacity = 0  # Hide the bar during rest
        else:
            self.bar.opacity = 1  # Show the bar

            t = self.sweep_clock.getTime() * self.speed

            if direction == "right":
                self.bar.pos = (-self.fov_size / 2 + self.bar_width / 2 + t, 0)
                self.bar.ori = 0  # ✅ Keep bar horizontal
            elif direction == "left":
                self.bar.pos = (self.fov_size / 2 - self.bar_width / 2 - t, 0)
                self.bar.ori = 0  # ✅ Keep bar horizontal
            elif direction == "down":
                self.bar.pos = (0, self.fov_size / 2 - self.bar_width / 2 - t)
                self.bar.ori = 90  # ✅ Rotate bar 90° for vertical movement
            elif direction == "up":
                self.bar.pos = (0, -self.fov_size / 2 + self.bar_width / 2 + t)
                self.bar.ori = 90  # ✅ Rotate bar 90° for vertical movement

        # ✅ Check if it's time to switch direction
        if (direction != "rest" and self.sweep_clock.getTime() >= self.sweep_duration) or \
           (direction == "rest" and self.sweep_clock.getTime() >= self.rest_duration):
            self.switch_direction()

    def flicker(self):
        """ ✅ Inverts the contrast of the checkerboard to create proper flicker. """
        if self.flicker_clock.getTime() >= 1 / (2 * 8):  # 8 Hz flicker
            self.contrast *= -1  # Flip contrast
            self.bar.contrast = self.contrast  # Apply new contrast
            self.flicker_clock.reset()

    def draw(self):
        """ Draws the sweeping bar within the aperture. """
        self.update_position()
        self.flicker()

        self.aperture.enabled = True  # ✅ Enable aperture just for this stimulus
        self.bar.draw()
        self.aperture.enabled = False  # ✅ Disable aperture so it doesn’t affect other stimuli

        self.background_circle.draw()

class CueStimulusArray:
    
    n_objects = 8  # Default number of stimuli

    def __init__(self, win, eccentricity=9, size=1.0, stim_type="circle"):
        """
        Creates an array of cue stimuli arranged in a circular pattern.
        Stimuli can be either white circles or rotated squares.

        :param win: PsychoPy window
        :param eccentricity: Distance from center (in degrees)
        :param size: Size of each stimulus (in degrees)
        :param stim_type: "circle" or "square" (rotated by 90 degrees)
        """
        assert stim_type in ["circle", "square"], "stim_type must be 'circle' or 'square'"

        self.win = win
        self.eccentricity = eccentricity
        self.size = size
        self.stim_type = stim_type

        positions = self.get_positions(eccentricity)

        # ✅ Create an array of stimuli (either circles or rotated squares)
        self.stimuli = [
            self.create_stimulus(pos)
            for pos in positions
        ]

    def create_stimulus(self, pos):
        """ Creates a single stimulus, either a circle or a rotated square. """
        if self.stim_type == "circle":
            return visual.Circle(self.win, radius=self.size / 2, fillColor=None, lineColor="white", pos=pos,
                                 lineWidth=5)
        else:  # Square case
            return visual.Rect(self.win, width=self.size, height=self.size, fillColor=None, lineColor="white", 
                               lineWidth=5,
                               pos=pos, ori=45)  # Rotated 90° (actually 45° for a diamond shape)

    def get_positions(self, eccentricity):
        """ Compute evenly spaced positions around a circular array. """
        positions = []

        for i in range(self.n_objects):
            angle = i * 360 / self.n_objects + (360 / (2 * self.n_objects))
            x = eccentricity * np.cos(np.radians(angle))
            y = eccentricity * np.sin(np.radians(angle))
            positions.append((x, y))

        return positions

    def draw(self):
        """ Draw all stimuli. """
        for stimulus in self.stimuli:
            stimulus.draw()
