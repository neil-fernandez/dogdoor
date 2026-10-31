from src.vision.metrics import compute_report


def test_compute_report():
    y_true = [0, 0, 1, 2, 3]
    y_pred = [0, 1, 1, 2, 3]
    r = compute_report(y_true, y_pred, ["a", "b", "c", "d"])

    assert r["accuracy"] == 0.8
    assert r["class_names"] == ["a", "b", "c", "d"]
    assert len(r["confusion_matrix"]) == 4
    assert all(len(row) == 4 for row in r["confusion_matrix"])
    assert len(r["per_class"]) == 4
    # class "a": one true positive, no false positives, one false negative
    assert r["per_class"][0]["precision"] == 1.0
    assert r["per_class"][0]["recall"] == 0.5
    assert r["per_class"][0]["support"] == 2


def test_zero_division_is_handled():
    # class "d" never appears in predictions or truth: P/R/F1 should be 0, not error
    r = compute_report([0, 0, 1], [0, 0, 1], ["a", "b", "c", "d"])
    assert r["per_class"][3]["precision"] == 0.0
    assert r["per_class"][3]["recall"] == 0.0
    assert r["per_class"][3]["f1"] == 0.0
