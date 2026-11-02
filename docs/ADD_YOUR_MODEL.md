# Add your trained model

The app uses a placeholder until these two files exist:

```text
models/gesture_model.onnx
models/labels.txt
```

`labels.txt` has one gesture per line, in the same order as the model's output classes.
Use `no_gesture`, `none`, `background`, or `nothing` for a class that shouldn't add a caption word.

## Export to ONNX

**PyTorch**

```python
dummy = torch.randn(1, 3, 224, 224)  # use your training input size
torch.onnx.export(model.eval(), dummy, "models/gesture_model.onnx", opset_version=18)
```

**Keras / TensorFlow**

```bash
pip install tf2onnx
python -m tf2onnx.convert --saved-model saved_model_dir --output models/gesture_model.onnx
```

## Match your preprocessing

`OnnxPredictor._preprocess` in `gesture_captions/predictor.py` converts frames to RGB and scales
them to 0-1. If training used something else, such as ImageNet normalisation, change this function
to match. A mismatch usually won't cause an error, but predictions will be wrong.

## If the model isn't a plain image classifier

For a landmark or sequence model, such as MediaPipe landmarks with an LSTM, add a class with:

```python
model_type = "landmarks"
labels = ...


def predict(self, frame_bgr) -> Prediction: ...
```

Return it from `load_predictor`, and add its packages to `requirements.txt`. Sequence models can
keep recent frames inside the class, but the app shares one predictor across sessions, so each
session would need its own buffer.

## Commit it with Git LFS

```bash
git lfs install
git add .gitattributes models/gesture_model.onnx models/labels.txt
git commit -m "Add trained gesture model"
```
