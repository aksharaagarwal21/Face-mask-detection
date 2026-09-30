from conftest import probs_for
from tracker import CentroidTracker


def box(cx, cy, s=40):
    return (cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2)


def test_ids_persist_while_faces_move():
    t = CentroidTracker(max_distance=50)
    t.update([box(50, 50), box(200, 50)])
    first = list(t.detection_ids)
    t.update([box(205, 55), box(55, 52)])           # same faces, listed in the other order
    assert t.detection_ids == [first[1], first[0]]


def test_optimal_matching_keeps_both_ids_when_a_detection_is_contested():
    t = CentroidTracker(max_distance=100)
    t.update([box(100, 100), box(140, 100)])
    a, b = t.detection_ids
    # The detection at x=120 is 20px from both faces. Greedy nearest-centroid
    # matching hands it to face a, leaves face b unmatched and registers the
    # detection at x=60 as a new face. The optimal assignment keeps both IDs.
    t.update([box(120, 100), box(60, 100)])
    assert t.detection_ids == [b, a]
    assert t.next_object_id == 2


def test_new_face_far_away_gets_a_new_id():
    t = CentroidTracker(max_distance=50)
    t.update([box(50, 50)])
    t.update([box(50, 50), box(300, 200)])
    assert len(set(t.detection_ids)) == 2


def test_face_is_dropped_after_max_disappeared():
    t = CentroidTracker(max_disappeared=2)
    t.update([box(50, 50)])
    for _ in range(3):
        t.update([])
    assert len(t.objects) == 0


def test_smoothing_absorbs_a_single_bad_frame():
    t = CentroidTracker(smoothing=0.5)
    for _ in range(5):
        t.update([box(50, 50)], probs=[probs_for("with_mask", 0.95)])
    t.update([box(50, 50)], probs=[probs_for("without_mask", 0.9)])   # one blurry frame
    label, _ = t.smoothed(t.detection_ids[0])
    assert label == "with_mask"


def test_no_smoothing_follows_the_latest_frame():
    t = CentroidTracker(smoothing=1.0)
    t.update([box(50, 50)], probs=[probs_for("with_mask", 0.95)])
    t.update([box(50, 50)], probs=[probs_for("without_mask", 0.9)])
    assert t.smoothed(t.detection_ids[0])[0] == "without_mask"


def test_reset_clears_everything():
    t = CentroidTracker()
    t.update([box(50, 50)], probs=[probs_for("with_mask")])
    t.reset()
    assert not t.objects and not t.probs and t.next_object_id == 0
