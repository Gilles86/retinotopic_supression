from psychopy import visual, core

win = visual.Window(size=(800, 600))

# Create three rectangles with different orientations
rect_0 = visual.Rect(win, width=0.5, height=0.2, fillColor='red', ori=0)    # No rotation
rect_45 = visual.Rect(win, width=0.5, height=0.2, fillColor='green', ori=45)  # 45-degree rotation
rect_90 = visual.Rect(win, width=0.5, height=0.2, fillColor='blue', ori=90)  # 90-degree rotation

rect_0.pos = (-0.5, 0)  # Left
rect_45.pos = (0, 0)    # Center
rect_90.pos = (0.5, 0)  # Right

while True:
    rect_0.draw()
    rect_45.draw()
    rect_90.draw()
    win.flip()
