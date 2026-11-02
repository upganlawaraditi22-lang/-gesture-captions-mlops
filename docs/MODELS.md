# Models

```text
Video frame -> mirror -> MediaPipe Hands -> landmark features -> ONNX classifier
            -> steady sign -> sentence in English, Hindi, or Marathi
```

| Sign type | Model | Input | Confidence needed | Classes |
|---|---|---|---|---|
| Static signs | `static_mlp` | 126 numbers | 75% | 18 |
| Finger spelling | `spell_mlp` | 126 numbers | 75% | 36 |
| Motion signs | `seq_lstm` | 30 frames of 130 numbers | 80% | 5 |

The `.onnx` files were exported from the project's PyTorch checkpoints, so the deployed app doesn't
need PyTorch. Each `.npz` file holds that model's labels and normalisation values.

## After retraining

Install PyTorch, `onnx`, and `onnxscript`, then run this from this repo:

```bash
python scripts/export_models.py --source ~/Desktop/gesture-captions
```

The script stops if the ONNX output doesn't match PyTorch.

`gesture_captions/features.py` must stay the same as the feature code used in training.

## Scope

The models recognise only the signs they were trained on, using data recorded by the project's
signers. Accuracy on other people or in other lighting may be lower.
