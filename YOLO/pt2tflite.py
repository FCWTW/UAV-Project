'''
Source:
https://docs.ultralytics.com/integrations/tflite/
https://github.com/tensorflow/probability/releases/tag/v0.21.0

conda create -n yolo-tf python=3.9 -y
pip install tensorflow==2.13.0 tensorflow-addons==0.22.0
pip install tensorflow-probability==0.21.0
pip install ultralytics
pip install onnx onnx-tf
'''

from ultralytics import YOLO
from onnx_tf.backend import prepare
import onnx
import tensorflow as tf

# .pt to .onnx
model = YOLO("yolo11n.pt")
model.export(format="onnx", opset=16)

# .onnx to SavedModel
onnx_model = onnx.load("yolo11n.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf")

# SavedModel to .tflite
converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
tflite_model = converter.convert()
with open("yolo11n.tflite", "wb") as f:
    f.write(tflite_model)

# Test
tflite_model = YOLO("yolo11n.tflite")
results = tflite_model("https://ultralytics.com/images/bus.jpg")