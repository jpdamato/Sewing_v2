import time
from typing import Any
from frame_renderer.window import Window
from frame_renderer.monitor import MonitorData

import math
import glfw
import numpy as np
import cv2


class WindowManager:
    def __init__(self):
        self.windows: dict[str, Window] = {}
        self._resize_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        if not glfw.init():
            print("Error initializing GLFW")
            return

    def __del__(self):
        glfw.terminate()

    def get_monitors(self):
        glfw_monitors: list[Any] = glfw.get_monitors()
        print("Monitores detectados:")
        print(f"glfw_monitors: {glfw_monitors}")
        monitors = list(map(lambda monitor: MonitorData(monitor), glfw_monitors))
        return monitors

    def create_window(
        self,
        height: int,
        width: int,
        title: str,
        position=(0, 0),
        maximized=True,
        decorators=False,
    ):
        # Needed to move the window to the correct position on multi-monitor setups
        #   in other way the window will be detected as fullscreen on the primary monitor
        height = math.floor(height / 1.001)
        width = math.floor(width / 1.001)

        window = Window(
            height,
            width,
            title,
            position=position,
            maximized=maximized,
            decorators=decorators,
        )
        if window.glfw_window is None:
            print("Error creating GLFW window")
            return None
        self.windows[title] = window

    def render(self, window_title: str, frame: np.ndarray):
        window = self.windows.get(window_title)
        if window is None:
            print(f"No window found with title: {window_title}")
            return
        frame_height, frame_width = frame.shape[:2]
        if frame_height != window.height or frame_width != window.width:
            frame = self._resize_frame(frame, window.width, window.height)

        window.render(frame)

        glfw.poll_events()

    def set_window_position(self, window_title: str, position):
        window = self.windows.get(window_title)
        if window is None:
            print(f"No window found with title: {window_title}")
            return
        window.set_window_position(position)

    def maximize_window(self, window_title: str):
        window = self.windows.get(window_title)
        if window is None:
            print(f"No window found with title: {window_title}")
            return
        window.maximize()

    def close_window(self, window_title: str):
        window = self.windows.get(window_title)
        if window is None:
            print(f"No window found with title: {window_title}")
            return
        window.close()
        del self.windows[window_title]

    def close_all_windows(self):
        for window in self.windows.values():
            window.close()
        self.windows.clear()

    def _resize_frame(
        self, frame: np.ndarray, new_width: int, new_height: int
    ) -> np.ndarray:
        return cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
