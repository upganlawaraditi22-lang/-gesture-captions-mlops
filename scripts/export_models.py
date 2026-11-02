"""Export trained PyTorch checkpoints to ONNX so the deployed app doesn't need PyTorch.

Run with PyTorch, onnx, and onnxscript installed, from the MLOps repo:
    python scripts/export_models.py --source ~/Desktop/gesture-captions

Each model produces models/<name>.onnx and models/<name>.npz (labels and normalisation values).
Only run this on checkpoints you trust, because they are loaded with pickle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

MODELS = ("static_mlp", "spell_mlp", "seq_lstm")


def export(name: str, source: Path, output: Path) -> None:
    from nn_models import StaticMLP, build_sequence_model

    checkpoint = torch.load(source / "models" / f"{name}.pt", weights_only=False)
    labels = [str(label) for label in checkpoint["labels"]]
    is_sequence = "arch" in checkpoint
    if is_sequence:
        frames = int(checkpoint["frames"])
        model = build_sequence_model(checkpoint["arch"], checkpoint["in_dim"], len(labels), frames)
        shape = (1, frames, checkpoint["in_dim"])
    else:
        frames = 0
        model = StaticMLP(checkpoint["in_dim"], len(labels))
        shape = (1, checkpoint["in_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    onnx_path = output / f"{name}.onnx"
    torch.onnx.export(
        model,
        (torch.zeros(shape),),
        onnx_path,
        input_names=["x"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )
    np.savez(
        output / f"{name}.npz",
        labels=np.array(labels),
        mean=np.asarray(checkpoint["mean"], dtype=np.float32),
        std=np.asarray(checkpoint["std"], dtype=np.float32),
        frames=frames,
    )

    # The exported model must give the same answers as the PyTorch one.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.tensor(sample)).numpy()
    actual = session.run(None, {"x": sample})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-3)
    print(f"{name}: {len(labels)} classes, ONNX output matches PyTorch: {labels}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.expanduser()))
    for name in MODELS:
        export(name, args.source.expanduser(), args.output)


if __name__ == "__main__":
    main()
