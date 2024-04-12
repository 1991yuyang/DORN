import torch as t
import torch.onnx
from network import Dorn


model_weights_pth = r"/home/yuyang/python_project/DORN/model/epoch.pth"
resnet_type = "resnet18"
K = 120
us_use_interpolate = True
input_image_size = [512, 512]  # [w, h]
onnx_output_pth = "model/depth.onnx"


model = Dorn(resnet_type, K, us_use_interpolate)
model.load_state_dict(t.load(model_weights_pth, map_location="cpu"))
model.eval()
input_shape = [1, 3] + input_image_size[::-1]
dummy_input = t.randn(input_shape).type(t.FloatTensor)
t.onnx.export(
                model,         # model being run
                dummy_input,       # model input (or a tuple for multiple inputs)
                onnx_output_pth,       # where to save the model
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=11,
                input_names=['input'],   # the model's input names
                output_names=['output']
              )
