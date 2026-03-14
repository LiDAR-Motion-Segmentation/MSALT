from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional
import logging
from src.core.objects import BoundingBox3D
from src.core.annotation_manager import AnnotationManager

logger = logging.getLogger(__name__)


class Command(ABC):
    """Abstract Base Class for all editor commands."""

    @abstractmethod
    def execute(self) -> bool:
        """Applies the logic. Returns True if successful."""
        pass

    @abstractmethod
    def undo(self) -> None:
        """Reverts the logic."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Returns a human-readable name for the UI (e.g. 'Undo Delete')."""
        pass


class CommandHistory:
    """
    Manages the stack of commands for Undo/Redo functionality.
    Follows a Singleton-like usage within the MainWindow.
    """

    def __init__(self, max_history: int = 50) -> None:
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        self._max_history = max_history

    def push(self, command: Command):
        if command.execute():
            self._undo_stack.append(command)
            self._redo_stack.append(command)

            # to enforce stack limit
            if len(self._undo_stack) > self._max_history:
                self._undo_stack.pop(0)

            logger.debug(f"Command Executed: {command.name()}")

    def undo(self) -> Optional[str]:
        if not self._undo_stack:
            return None

        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        return cmd.name()

    def redo(self) -> Optional[str]:
        if not self._redo_stack:
            return None

        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)
        return cmd.name()


class AddBoxCommand(Command):
    def __init__(
        self, manager: AnnotationManager, frame_idx: int, box: BoundingBox3D
    ) -> None:
        self.manager = manager
        self.frame_idx = frame_idx
        self.box = deepcopy(box)

    def execute(self) -> bool:
        self.manager.add_box(self.frame_idx, self.box)
        return True

    def undo(self) -> None:
        self.manager.delete_box(self.frame_idx, self.box)

    def name(self) -> str:
        return f"Add Box {self.box.track_id}"


class ModifyBoxCommand(Command):
    """
    Handles updates to geometry or labels.
    Crucial: Requires 'before' and 'after' states.
    """

    def __init__(
        self,
        manager: AnnotationManager,
        frame_idx: int,
        old_box: BoundingBox3D,
        new_box: BoundingBox3D,
    ) -> None:
        self.manager = manager
        self.frame_idx = frame_idx
        self.old_state = deepcopy(old_box)
        self.new_state = deepcopy(new_box)

    def execute(self) -> bool:
        """
        Apply the modification: replace old_state with new_state.
        """
        self.manager.delete_box(self.frame_idx, self.old_state)
        self.manager.add_box(self.frame_idx, self.new_state)
        return True

    def undo(self) -> None:
        """
        Revert the modification: restore old_state.
        """
        self.manager.delete_box(self.frame_idx, self.new_state)
        self.manager.add_box(self.frame_idx, self.old_state)

    def name(self) -> str:
        return f"Modify Box {self.new_state.track_id}"


class BulkDeleteCommand(Command):
    """Composite command for deleting multiple boxes at once."""

    def __init__(
        self, manager: AnnotationManager, frame_idx: int, boxes: List[BoundingBox3D]
    ) -> None:
        self.manager = manager
        self.frame_idx = frame_idx
        self.boxes = [deepcopy(b) for b in boxes]

    def execute(self) -> bool:
        for box in self.boxes:
            self.manager.delete_box(self.frame_idx, box)
        return True

    def undo(self) -> None:
        for box in self.boxes:
            self.manager.add_box(self.frame_idx, box)

    def name(self) -> str:
        return f"Delete {len(self.boxes)} Boxes"


class DeleteBoxCommand(Command):
    def __init__(
        self, manager: AnnotationManager, frame_idx: int, box: BoundingBox3D
    ) -> None:
        self.manager = manager
        self.frame_idx = frame_idx
        self.box = deepcopy(box)  # trying to save the state before deleting

    def execute(self) -> bool:
        self.manager.delete_box(self.frame_idx, self.box)
        return True

    def undo(self) -> None:
        self.manager.add_box(self.frame_idx, self.box)

    def name(self) -> str:
        return f"Delete Box {self.box.track_id}"
