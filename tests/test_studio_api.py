from fastapi.testclient import TestClient

from src.studio.app import app

client = TestClient(app)


def test_all_get_routes_render():
    for path in ["/", "/datasets", "/train", "/predict", "/experiments", "/models", "/datasets/beau"]:
        assert client.get(path).status_code == 200, path


def test_unknown_class_404():
    assert client.get("/datasets/not-a-class").status_code == 404
