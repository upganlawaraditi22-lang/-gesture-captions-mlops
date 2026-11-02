import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from gesture_captions.config import Settings
from gesture_captions.predictor import (
    OnnxPredictor,
    PlaceholderPredictor,
    load_predictor,
    read_labels,
)


def make_channel_mean_model(path, channels_first=True, softmax=True):
    """Build a small classifier whose three classes are the mean R, G, and B values."""
    shape = [1, 3, 8, 8] if channels_first else [1, 8, 8, 3]
    axes = [2, 3] if channels_first else [1, 2]
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["axes"],
            value=helper.make_tensor("axes_value", TensorProto.INT64, [2], axes),
        ),
        helper.make_node("ReduceMean", ["x", "axes"], ["mean"], keepdims=0),
    ]
    if softmax:
        nodes.append(helper.make_node("Softmax", ["mean"], ["y"], axis=1))
    else:
        nodes.append(helper.make_node("Identity", ["mean"], ["y"]))
    graph = helper.make_graph(
        nodes,
        "test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, str(path))


def solid_bgr(blue, green, red):
    frame = np.zeros((32, 32, 3), np.uint8)
    frame[:] = (blue, green, red)
    return frame


def test_placeholder_is_labelled_demo():
    prediction = PlaceholderPredictor().predict(np.zeros((8, 8, 3), np.uint8))
    assert prediction.label == "DEMO_DARK"


@pytest.mark.parametrize("channels_first", [True, False])
def test_onnx_converts_bgr_to_rgb(tmp_path, channels_first):
    model_path = tmp_path / "model.onnx"
    make_channel_mean_model(model_path, channels_first)
    predictor = OnnxPredictor(model_path, ("red", "green", "blue"))
    assert predictor.predict(solid_bgr(0, 0, 255)).label == "red"
    assert predictor.predict(solid_bgr(255, 0, 0)).label == "blue"


def test_onnx_applies_softmax_to_raw_scores(tmp_path):
    model_path = tmp_path / "model.onnx"
    make_channel_mean_model(model_path, softmax=False)
    prediction = OnnxPredictor(model_path, ("red", "green", "blue")).predict(solid_bgr(0, 128, 64))
    assert prediction.label == "green"
    assert 0.33 < prediction.confidence < 0.6


def test_label_count_mismatch_is_rejected(tmp_path):
    model_path = tmp_path / "model.onnx"
    make_channel_mean_model(model_path)
    with pytest.raises(ValueError, match="labels"):
        OnnxPredictor(model_path, ("only", "two"))


def test_load_uses_placeholder_without_model(tmp_path):
    settings = Settings(model_path=str(tmp_path / "missing.onnx"))
    assert load_predictor(settings).model_type == "placeholder"


def test_load_uses_model_when_present(tmp_path):
    model_path = tmp_path / "model.onnx"
    labels_path = tmp_path / "labels.txt"
    make_channel_mean_model(model_path)
    labels_path.write_text("red\ngreen\n\nblue\n", encoding="utf-8")
    settings = Settings(model_path=str(model_path), labels_path=str(labels_path))
    assert load_predictor(settings).model_type == "onnx"


def test_invalid_model_file_fails_loudly(tmp_path):
    model_path = tmp_path / "model.onnx"
    labels_path = tmp_path / "labels.txt"
    model_path.write_bytes(b"not a model")
    labels_path.write_text("a\n", encoding="utf-8")
    settings = Settings(model_path=str(model_path), labels_path=str(labels_path))
    with pytest.raises(Exception):  # noqa: B017 - onnxruntime raises its own error types
        load_predictor(settings)


def test_empty_labels_file_is_rejected(tmp_path):
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No labels"):
        read_labels(labels_path)
