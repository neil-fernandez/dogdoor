from PIL import Image

from src.vision.config import Roster
from src.vision.dataset import RosterDataset


def _make(tmp_path):
    roster = Roster(
        {"animals": [{"id": "beau"}, {"id": "red"}], "negative_class": {"id": "other"}}
    )
    for cid, count in [("beau", 3), ("red", 2), ("other", 1)]:
        d = tmp_path / cid
        d.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (16, 16), (i * 40, i * 40, i * 40)).save(d / f"{i}.jpg")
    return roster


def test_samples_and_labels(tmp_path):
    roster = _make(tmp_path)
    ds = RosterDataset(tmp_path, roster, transform=None)
    assert len(ds) == 6
    labels = sorted(ds.samples[i][1] for i in range(len(ds)))
    assert labels == [0, 0, 0, 1, 1, 2]


def test_class_counts(tmp_path):
    roster = _make(tmp_path)
    ds = RosterDataset(tmp_path, roster, transform=None)
    assert ds.class_counts() == {"beau": 3, "red": 2, "other": 1}


def test_getitem_shape(tmp_path):
    roster = _make(tmp_path)
    from torchvision import transforms as T

    ds = RosterDataset(tmp_path, roster, transform=T.ToTensor())
    img, label = ds[0]
    assert tuple(img.shape) == (3, 16, 16)
    assert label == 0
