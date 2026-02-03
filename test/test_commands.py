from dataclasses import dataclass
from typing import Dict, List

from src.core.objects import BoundingBox3D
from src.core.commands import (
    AddBoxCommand,
    BulkDeleteCommand,
    CommandHistory,
    DeleteBoxCommand,
    ModifyBoxCommand,
)


@dataclass
class _FrameState:
    boxes: List[BoundingBox3D]


class FakeAnnotationManager:
    """
    Minimal stand-in for AnnotationManager for testing command behavior
    without touching disk or depending on full manager logic.
    """

    def __init__(self) -> None:
        self.frames: Dict[int, _FrameState] = {}

    def _ensure_frame(self, frame_idx: int) -> _FrameState:
        if frame_idx not in self.frames:
            self.frames[frame_idx] = _FrameState(boxes=[])
        return self.frames[frame_idx]

    def add_box(self, frame_idx: int, box: BoundingBox3D) -> None:
        state = self._ensure_frame(frame_idx)
        state.boxes.append(box)

    def delete_box(self, frame_idx: int, box: BoundingBox3D) -> None:
        state = self._ensure_frame(frame_idx)
        state.boxes = [b for b in state.boxes if b != box]

    def get_boxes(self, frame_idx: int) -> List[BoundingBox3D]:
        return list(self._ensure_frame(frame_idx).boxes)


def _make_box(track_id: int, x: float = 0.0) -> BoundingBox3D:
    return BoundingBox3D(
        x=x,
        y=0.0,
        z=0.0,
        dx=2.0,
        dy=2.0,
        dz=2.0,
        heading=0.0,
        label="car",
        track_id=track_id,
    )


def test_add_box_command_execute_and_undo():
    manager = FakeAnnotationManager()
    history = CommandHistory()
    frame = 0
    box = _make_box(track_id=1)

    cmd = AddBoxCommand(manager, frame, box)
    history.push(cmd)

    boxes = manager.get_boxes(frame)
    assert len(boxes) == 1
    assert boxes[0] == cmd.box  # deep-copied inside command

    history.undo()
    assert manager.get_boxes(frame) == []


def test_delete_box_command_execute_and_undo():
    manager = FakeAnnotationManager()
    history = CommandHistory()
    frame = 0
    box = _make_box(track_id=1)

    manager.add_box(frame, box)
    assert len(manager.get_boxes(frame)) == 1

    cmd = DeleteBoxCommand(manager, frame, box)
    history.push(cmd)
    assert manager.get_boxes(frame) == []

    history.undo()
    boxes = manager.get_boxes(frame)
    assert len(boxes) == 1
    assert boxes[0] == box


def test_bulk_delete_command_execute_and_undo():
    manager = FakeAnnotationManager()
    history = CommandHistory()
    frame = 0
    boxes = [_make_box(track_id=i) for i in range(1, 4)]
    for b in boxes:
        manager.add_box(frame, b)

    cmd = BulkDeleteCommand(manager, frame, boxes)
    history.push(cmd)
    assert manager.get_boxes(frame) == []

    history.undo()
    restored = manager.get_boxes(frame)
    assert len(restored) == 3
    # Order is not important, but all track_ids should be present
    assert sorted(b.track_id for b in restored) == [1, 2, 3]


def test_modify_box_command_execute_undo_and_redo():
    manager = FakeAnnotationManager()
    history = CommandHistory()
    frame = 0
    old_box = _make_box(track_id=1, x=0.0)
    new_box = _make_box(track_id=1, x=5.0)

    manager.add_box(frame, old_box)
    assert len(manager.get_boxes(frame)) == 1

    cmd = ModifyBoxCommand(manager, frame, old_box, new_box)
    history.push(cmd)

    # After execute: old box replaced by new_box
    boxes_after = manager.get_boxes(frame)
    assert len(boxes_after) == 1
    assert boxes_after[0] == cmd.new_state

    # Undo should restore old state
    history.undo()
    boxes_undo = manager.get_boxes(frame)
    assert len(boxes_undo) == 1
    assert boxes_undo[0] == cmd.old_state

    # Redo should re-apply the modification
    history.redo()
    boxes_redo = manager.get_boxes(frame)
    assert len(boxes_redo) == 1
    assert boxes_redo[0] == cmd.new_state

