import mss
import numpy as np
import pygetwindow as gw
import time

class Capture:
    def __init__(self, window_name="BlueStacks"):
        self.sct = mss.mss()
        self.window_name = window_name
        self.region = self._get_window_region()

        if not self.region:
            raise RuntimeError("Finestra non trovata")

    def _get_window_region(self):
        windows = gw.getWindowsWithTitle(self.window_name)
        if not windows:
            return None

        w = windows[0]
        if w.isMinimized:
            w.restore()
            time.sleep(0.2)

        return {#taglio l'UI di bluestacks
            "top": w.top+30,
            "left": w.left,
            "width": w.width-30,
            "height": w.height-30
        }

    def grab(self):
        img = np.array(self.sct.grab(self.region))
        return img[:, :, :3]  # BGR