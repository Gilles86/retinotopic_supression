# Singleton Distractor Task

This experiment implements a version of the **Additional Singleton Paradigm**, using PsychoPy and `exptools2`.

Participants are asked to report whether a small dot is present **inside the target shape** on each trial.  
The **target** is defined as the only shape with an opposite orientation compared to the rest (e.g., vertical among horizontals), and it **never has the distractor color**.

---

## 🧠 Task Overview

On each trial:

- A display of 8 shapes appears.
- One shape has a different orientation — this is the **target**.
- Some shapes may have a black dot (the **distractor**).
- Your task is to decide whether the **target contains a dot** and press the appropriate key.

Distractors are most likely to appear at one of the locations (configurable), allowing measurement of spatial attentional biases.

---

## ▶️ Running the Experiment

To start the experiment, run the following command:

```bash
python main.py SUBJECT SESSION RUN MOST_LIKELY_DISTRACTOR_LOCATION [--settings SETTINGS_NAME] [--calibrate_eyetracker]
```

### Required Arguments

| Argument                      | Description                                                |
|------------------------------|------------------------------------------------------------|
| `SUBJECT`                    | Subject ID (e.g., `01`)                                    |
| `SESSION`                    | Session label (e.g., `A`)                                  |
| `RUN`                        | Run number (integer)                                       |
| `MOST_LIKELY_DISTRACTOR_LOCATION` | Location where the distractor is most likely to appear (1, 3, 5, or 7) |

### Optional Flags

| Flag                      | Description                              |
|--------------------------|------------------------------------------|
| `--settings`             | Settings YAML file label (default: `default`) |
| `--calibrate_eyetracker` | Run eyetracker calibration at the start  |

### Example

```bash
python main.py 01 A 1 5 --settings default --calibrate_eyetracker
```

---

## 🎮 Response Keys

Participants respond using the keyboard:

- Press **`F`** if the **dot is present** inside the target.
- Press **`J`** if the **dot is absent** inside the target.

These keys can be changed in the settings file (`settings/default.yml`) under the `experiment -> keys` section.

---

## 📋 Editing Instructions

Instructions shown before the experiment are stored in [`instructions.yml`](./instructions.yml).

To edit them:

1. Open `instructions.yml`.
2. Modify the text under the `intro` block.
3. You can use `{run}` as a placeholder that will be replaced by the current run number.

Example:

```yaml
intro: |
  - INSTRUCTIONS -

  Your task is to detect whether the unique shape contains a dot.

  Press 'F' if the dot is present.
  Press 'J' if the dot is absent.

  --- press space to start ---
```

---

## ⚙️ Customizing Settings

Settings for the experiment are stored in YAML files in the `settings/` folder.

To create your own config:

1. Duplicate `settings/default.yml` and rename it (e.g., `mysettings.yml`)
2. Pass it to the script via `--settings mysettings`

### Common Options in `default.yml`

#### Number of Trials

```yaml
design:
  n_trials: 30
```

#### Distractor Probability

```yaml
design:
  likely_distractor_probability: 0.67
```

#### Stimulus Parameters

```yaml
experiment:
  size_stimuli: 2.0
  eccentricity_stimulus: 5.0
  size_fixation: 0.25
  keys: ['f', 'j']
```

#### Timing (in seconds)

```yaml
durations:
  trial_start: 0.5
  target: 1.5
  feedback: 1.0
  iti: [1, 3, 5]  # jittered
```

#### Display Settings

```yaml
window:
  size: [1200, 1200]
  fullscr: False
monitor:
  distance: 50  # in cm
```

---

## 🧪 Output

All data and logs are saved to a structured output directory:

```
output/sub-01_ses-A_task-estimation_run-1/
```

Contents include:

- Trial-by-trial `.csv` file
- PsychoPy `.log` file
- Eyetracker data (if used)

---

## 📦 Dependencies

Install required Python packages with:

```bash
pip install psychopy numpy pyyaml git+https://github.com/VU-Cog-Sci/exptools2.git
```

---

## 📁 Project Structure

```
.
├── main.py                 # Entry point for the experiment
├── session.py              # Manages trial structure and eyetracker
├── trial.py                # Defines instruction and main trials
├── stimuli/                # Visual stimulus definitions
├── settings/
│   └── default.yml         # Default configuration file
├── instructions.yml        # Participant instructions
├── utils.py                # Helper utilities
└── output/                 # Data output per subject/session/run
```

---

Feel free to reach out if you want to extend this task, e.g., add conditions, log additional trial variables, or support fMRI synchronization.
