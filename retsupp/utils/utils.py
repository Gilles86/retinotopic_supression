import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def create_animation(stimulus, frametimes=None, interval=250, display=True):
    """
    Create an animation from a 3D stimulus array.

    Parameters:
    - stimulus: 3D numpy array where the first dimension is time (frames).
    - frametimes: Optional array of frame times.
    - interval: Delay between frames in milliseconds.

    Returns:
    - HTML object to display the animation in a Jupyter notebook.
    """
    if frametimes is None:
        frametimes = range(stimulus.shape[0])

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        ax.set_title(f"Frame: {frame}, Time: {frametimes[frame]}")
        ax.imshow(stimulus[frame, :, :], cmap='gray', vmin=0, vmax=1, origin='lower')
        return ax

    ani = FuncAnimation(fig, update, frames=len(frametimes), interval=interval)

    # Show the animation in notebook
    plt.close(fig)  # Prevents static display of the last frame
    if display:
        return HTML(ani.to_jshtml())
    else:
        return ani