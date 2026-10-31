import pytest
import torch

from src.vision.model import BACKBONES, build_model


@pytest.mark.parametrize("backbone", sorted(BACKBONES))
def test_backbone_output_shape(backbone):
    """Regression: the classifier head must be replaced with the correct in_features."""
    model = build_model(backbone, 4, freeze=False, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert tuple(out.shape) == (1, 4), backbone


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        build_model("not_a_model", 4)
