

# ── GUI Theme ────────────────────────────────────────────────────────

class Colors:
    """Centralized GUI color palette."""
    TITLE         = (230, 89, 89)      # Section titles (both panels)
    DESCRIPTION   = (74, 101, 176)     # Automaton description text
    FPS_LABEL     = (230, 120, 120)    # Live FPS counter
    LIVE_STATE    = (120, 230, 120)    # Live automaton state text
    BACKGROUND    = (0, 0, 0)          # Screen background


class FontSizes:
    """Centralized GUI font sizes."""
    TITLE       = 17    # Section titles
    SUBTITLE    = 14    # Smaller titles ("Automaton controls:")
    LABEL       = 13    # FPS counter, live automaton state
    HELP        = 13    # Help / controls body text


# ── Help Strings ─────────────────────────────────────────────────────

INTERFACE_HELP = {
"title": "General controls",
"content": \
"""H: show/hide this help
SPACE: start/stop the simulation
S: (while stopped) one simulation step
R: toggle recording of simulation
P: screenshot the simulation
Q: quit the simulation
CTRL+WHEEL: Zoom in/out
CTRL+DRAG: Move the camera
C: reset the camera position and zoom"""
}
