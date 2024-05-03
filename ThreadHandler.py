import time
from PyQt5.QtCore import *

from Parameters import camera_update_time


class SimpleThread(QThread):
    signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread_Active = None
        self.camera_update_time = camera_update_time

    def run(self):
        self.thread_Active = True

        while self.thread_Active:
            time.sleep(self.camera_update_time)
            self.signal.emit(1)

    def stop(self):
        self.thread_Active = False
        self.quit()
