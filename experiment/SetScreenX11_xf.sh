# Scanning screen setup
# DP-6 = BOLD screen
# DP-2 = inverted iiyama screen
# DP-1 = iiyama screen

# forcing resolution and framerate on DP-1, making DP-1 the primary screen
xrandr --output DP-1 --primary --mode 1920x1080 --rate 120

# forcing resolution and framerate on DP-1, WITHOUT making DP-1 the primary screen
# xrandr --output DP-1 --mode 1920x1080 --rate 120

# forcing resolution and framerate on DP-2, invert
xrandr --output DP-2 --mode 1920x1080 --rate 120 --rotate inverted

# forcing resolution and framerate on DP-6, invert, set to the right of DP-1, clone onto DP-2
xrandr --output DP-6 --mode 1920x1080 --rate 120 --rotate inverted --right-of DP-1 --output DP-2 --same-as DP-6

# forcing resolution and framerate on DP-6, invert, set to the right of DP-1, AND make it the primary screen, clone onto DP-2
# xrandr --output DP-6 --primary --mode 1920x1080 --rate 120 --rotate inverted --right-of DP-1 --output DP-2 --same-as DP-6
