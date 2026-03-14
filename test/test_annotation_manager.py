from src.core.annotation_manager import AnnotationManager
from src.core.objects import BoundingBox3D


def _make_box(track_id: int = -1, label: str = "car") -> BoundingBox3D:
    return BoundingBox3D(
        x=0.0,
        y=0.0,
        z=0.0,
        dx=2.0,
        dy=2.0,
        dz=2.0,
        heading=0.0,
        label=label,
        track_id=track_id,
    )


def test_add_box_assigns_track_id_when_unset():
    manager = AnnotationManager()
    box = _make_box(track_id=-1)

    manager.add_box(0, box)
    boxes = manager.get_boxes(0)

    assert len(boxes) == 1
    # First box in a frame should get track_id = 1
    assert boxes[0].track_id == 1


def test_add_box_preserves_existing_track_id():
    manager = AnnotationManager()
    box = _make_box(track_id=42)

    manager.add_box(0, box)
    boxes = manager.get_boxes(0)

    assert len(boxes) == 1
    assert boxes[0].track_id == 42


def test_delete_box_removes_box_from_frame():
    manager = AnnotationManager()
    box = _make_box(track_id=1)

    manager.add_box(0, box)
    assert len(manager.get_boxes(0)) == 1

    manager.delete_box(0, box)
    assert manager.get_boxes(0) == []


def test_remove_box_by_track_id():
    manager = AnnotationManager()
    box1 = _make_box(track_id=1)
    box2 = _make_box(track_id=2)

    manager.add_box(0, box1)
    manager.add_box(0, box2)
    assert len(manager.get_boxes(0)) == 2

    manager.remove_box(0, track_id=1)
    boxes = manager.get_boxes(0)

    assert len(boxes) == 1
    assert boxes[0].track_id == 2


def test_deselect_all_clears_selected_flag_across_frames():
    manager = AnnotationManager()
    box1 = _make_box(track_id=1)
    box2 = _make_box(track_id=2)

    box1.selected = True
    box2.selected = True

    manager.add_box(0, box1)
    manager.add_box(1, box2)

    manager.deselect_all()

    for frame_idx in [0, 1]:
        for box in manager.get_boxes(frame_idx):
            assert not box.selected
