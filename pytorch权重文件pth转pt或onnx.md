- 转pt

```python
import torch
from old.tiny import Tiny

net=Tiny()
ckpt = torch.load('.pth', map_location="cpu")
net.eval()
net.load_state_dict(ckpt)
x = torch.randn((1, 3, 416, 416))
traced_script_module = torch.jit.trace(net, x)
traced_script_module.save('yolov3-tiny.pt')
```

- 转onnx

```python
import torch
from old.tiny import Tiny

net=Tiny()
ckpt = torch.load('.pth', map_location="cpu")
net.eval()
net.load_state_dict(ckpt)
x = torch.randn((1, 3, 416, 416))
# input_names和output_names会分配给相应变量
torch.onnx.export(net, x, './yolov3-tiny.onnx', opset_version=12, input_names=['input'], output_names=['output1', 'output2'])
```

- 验证onnx

```python
import onnxruntime   
import onnx

session = onnxruntime.InferenceSession('yolov3-tiny.onnx', None)
读取和处理image
outputs = session.run([], {'input': image})
后处理
```

