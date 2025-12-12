from abc import ABC, abstractmethod
from PyQt6.QtWidgets import QWidget
from typing import Optional
from src.data.structures import FrameData

class BasePluginWidget(QWidget):
    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.plugin_title = title
        
    def on_frame_update(self, data: FrameData) -> None:
        """
        Called by the Main Window when the timeline changes.
        Must be overridden by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement 'on_frame_update'")

    def reset(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} must implement 'reset'")