from src.vision.config import Roster, list_images, load_roster


def test_default_roster_classes():
    r = load_roster()
    assert r.class_ids == ["beau", "red", "misty", "other"]
    assert r.class_names == ["Beau", "Red", "Misty", "Other"]
    assert r.num_classes == 4
    assert r.id_to_index["beau"] == 0
    assert r.id_to_index["other"] == 3


def test_roster_is_generalizable():
    r = Roster(
        {
            "animals": [{"id": "rex", "name": "Rex"}, {"id": "luna", "name": "Luna"}],
            "negative_class": {"id": "unknown", "name": "Unknown"},
        }
    )
    assert r.class_ids == ["rex", "luna", "unknown"]
    assert r.num_classes == 3
    assert r.index_to_name[2] == "Unknown"


def test_list_images_filters_extensions(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.webp").write_bytes(b"x")
    names = sorted(p.name for p in list_images(tmp_path))
    assert names == ["a.jpg", "b.png", "d.webp"]


def test_list_images_missing_dir():
    assert list_images(__import__("pathlib").Path("does-not-exist-xyz")) == []
