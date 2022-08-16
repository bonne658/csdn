- 以yolov3-tiny为例，加载函数是`ting.py`的`load_darknet`
- base.py

```python
import torch
import torch.nn as nn

class BaseConv(nn.Module):
	def __init__(self, in_channel, out_channel, k_size, stride, pad, bn, act):
		super().__init__()
		self.conv = nn.Conv2d(in_channel, out_channel, k_size, stride, pad, bias=not bn)
		if bn == True:
			self.bn = nn.BatchNorm2d(out_channel)
		else:
			self.bn = nn.Identity()
		if act == 'leaky': self.act = nn.LeakyReLU(0.1)
		elif act == 'linear': self.act = nn.Identity()
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return self.act(x)
		
if __name__ == '__main__':
	data=torch.zeros(2,3, 320,320)
	bc=BaseConv(3, 16, 3, 1, 1, False, 'leaky')
	x = bc(data)
	print(x.shape)
```

- tiny.py（网络结构）

```python
import torch
import torch.nn as nn
from base import BaseConv
import numpy as np

class Backbone(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv0 = BaseConv(3, 16, 3, 1, 1, True, 'leaky')
		self.maxpool1 = nn.MaxPool2d(2)
		self.conv2 = BaseConv(16, 32, 3, 1, 1, True, 'leaky')
		self.maxpool3 = nn.MaxPool2d(2)
		self.conv4 = BaseConv(32, 64, 3, 1, 1, True, 'leaky') 
		self.maxpool5 = nn.MaxPool2d(2)
		self.conv6 = BaseConv(64, 128, 3, 1, 1, True, 'leaky')
		self.maxpool7 = nn.MaxPool2d(2)
		self.conv8 = BaseConv(128, 256, 3, 1, 1, True, 'leaky')
		self.maxpool9 = nn.MaxPool2d(2)
		self.conv10 = BaseConv(256, 512, 3, 1, 1, True, 'leaky')
		self.pad = nn.ZeroPad2d((0,1,0,1))
		self.maxpool11 = nn.MaxPool2d(2, 1)
		self.conv12 = BaseConv(512, 1024, 3, 1, 1, True, 'leaky')
		self.conv13 = BaseConv(1024, 256, 1, 1, 0, True, 'leaky')
		self.conv14 = BaseConv(256, 512, 3, 1, 1, True, 'leaky')
	def forward(self, x):
		x0=self.conv0(x)
		x1=self.maxpool1(x0)  # 1    16*208*208
		x2=self.conv2(x1)
		x3=self.maxpool3(x2)  # 3    32*104*104
		x4=self.conv4(x3)
		x5=self.maxpool5(x4)  # 5    64*52*52
		x6=self.conv6(x5)
		x7=self.maxpool7(x6)  # 7    128*26*26
		r2=self.conv8(x7)
		x9=self.maxpool9(r2)  # 9    256*13*13
		x10=self.conv10(x9)   # 10   512*13*13
		x11=self.pad(x10)
		x11=self.maxpool11(x11)       # 11     512*13*13
		x12=self.conv12(x11)          # 12     1024*13*13
		r1=self.conv13(x12)           # 13     256*13*13
		x14=self.conv14(r1)           # 14     512*13*13
		return r1, r2, x14		

class Tiny(nn.Module):
	def __init__(self):
		super().__init__()
		self.backbone = Backbone()
		self.convs=[]
		for m in self.backbone.modules():
			if isinstance(m, BaseConv):
				self.convs.append(m)
		self.conv15 = BaseConv(512, 255, 1, 1, 0, False, 'linear')
		self.conv18 = BaseConv(256, 128, 1, 1, 0, True, 'leaky')
		self.upsample19 = nn.Upsample(scale_factor=2)
		self.conv21 = BaseConv(384, 256, 3, 1, 1, True, 'leaky')
		self.conv22 = BaseConv(256, 255, 1, 1, 0, False, 'linear')
		self.convs.append(self.conv15)
		self.convs.append(self.conv18)
		self.convs.append(self.conv21)
		self.convs.append(self.conv22)
		
	def forward(self, x):
		r1, r2, x14 = self.backbone(x)
		out1=self.conv15(x14)         # 15     255*13*13
		x18=self.conv18(r1)           # 18     128*13*13
		x19=self.upsample19(x18)      # 19     128*26*26
		x20=torch.cat([x19, r2], 1)   # 20     384*26*26
		x21=self.conv21(x20)          # 21     256*26*26
		out2=self.conv22(x21)         # 22     255*26*26
		return out1, out2
		
	def load_darknet(self, weights_path):
		with open(weights_path, "rb") as f:
			header = np.fromfile(f, dtype=np.int32, count=5)
			weights = np.fromfile(f, dtype=np.float32)
		ptr = 0
		for i,conv in enumerate(self.convs):
			conv_layer = conv.conv
			if i!=9 and i!=12:
				bn_layer = conv.bn
				num_b = bn_layer.bias.numel()
				# bn bias
				bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
				bn_layer.bias.data.copy_(bn_b)
				ptr += num_b
				# bn weight
				bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
				bn_layer.weight.data.copy_(bn_w)
				ptr += num_b
				# Running Mean
				bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
				bn_layer.running_mean.data.copy_(bn_rm)
				ptr += num_b
				# Running Var
				bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
				bn_layer.running_var.data.copy_(bn_rv)
				ptr += num_b
			else:
				# conv bias
				num_b = conv_layer.bias.numel()
				conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
				conv_layer.bias.data.copy_(conv_b)
				ptr += num_b
			# conv
			num_w = conv_layer.weight.numel()
			conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
			conv_layer.weight.data.copy_(conv_w)
			ptr += num_w
			rest = weights.size - ptr
			if rest <= 0 : 
				print(i)
				break
		print('rest: ', rest)
		return rest
		
	def init(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight.data, 0.0, 0.02)
			if isinstance(m, nn.BatchNorm2d):
				nn.init.normal_(m.weight.data, 1.0, 0.02)
				nn.init.constant_(m.bias.data, 0.0)

if __name__ == '__main__':
	data=torch.zeros(2,3,416,416)
	t=Tiny()
	x1, x2=t(data)
	print(x1.shape, x2.shape)
	print(t.load_darknet('/home/lwd/code/darknet/yolov3-tiny.weights'))
```

- predict.py

```python
from PIL import Image, ImageDraw
import numpy as np
import torch, copy, time, sys, cv2
from tiny import Tiny

def get_boxes(output, anchors):
	h=output.size(2)
	w=output.size(3)
	output=output.view(3,85,h,w).permute(0,2,3,1).contiguous()
	# conf
	conf = torch.sigmoid(output[..., 4])
	cl = torch.sigmoid(output[..., 5:])
	clv, cli = torch.max(cl, -1)
	conf = conf * clv
	mask = conf > 0.15
	conf = conf[mask].unsqueeze(-1)
	cli = cli[mask].unsqueeze(-1)
	# grid
	FloatTensor = torch.cuda.FloatTensor if conf.is_cuda else torch.FloatTensor
	grid_h, grid_w = torch.meshgrid(torch.arange(h), torch.arange(w))
	grid_h = grid_h.repeat(3,1,1).type(FloatTensor)
	grid_w = grid_w.repeat(3,1,1).type(FloatTensor)
	tx = (torch.sigmoid(output[..., 0]) + grid_w) / w
	ty = (torch.sigmoid(output[..., 1]) + grid_h) / h
	tx = tx[mask].unsqueeze(-1)
	ty = ty[mask].unsqueeze(-1)
	# anchor
	aw = torch.Tensor(anchors[0::2]).view(3,1).repeat(1,h*w).view(3,h,w).type(FloatTensor)
	ah = torch.Tensor(anchors[1::2]).view(3,1).repeat(1,h*w).view(3,h,w).type(FloatTensor)
	tw = torch.exp(output[..., 2]) * aw
	th = torch.exp(output[..., 3]) * ah
	tw = tw[mask].unsqueeze(-1)
	th = th[mask].unsqueeze(-1)
	return torch.cat([tx, ty, tw, th, cli, conf], -1)
	
def iou(a,b):
	A=len(a)
	B=len(b)
	area1=a[:,2]*a[:,3]
	area1=area1.unsqueeze(1).expand(A,B)
	area2=b[:,2]*b[:,3]
	area2=area2.unsqueeze(0).expand(A,B)
	ba=torch.zeros(a.shape).cuda()
	bb=torch.zeros(b.shape).cuda()
	ba[:,0:2]=a[:,0:2]-a[:,2:]/2.0
	ba[:,2:]=ba[:,0:2]+a[:,2:]
	bb[:,0:2]=b[:,0:2]-b[:,2:]/2.0
	bb[:,2:]=bb[:,0:2]+b[:,2:]
	ba=ba.unsqueeze(1).expand(A,B,4)
	bb=bb.unsqueeze(0).expand(A,B,4)
	lt=torch.max(ba[:,:,0:2], bb[:,:,0:2])
	rb=torch.min(ba[:,:,2:], bb[:,:,2:])
	inter=torch.clamp((rb-lt),min=0)
	inter=inter[:,:,0]*inter[:,:,1]
	return inter/(area1+area2-inter)

def nms(box):
	box = box[torch.argsort(box[:,-1])]
	result=[]
	while len(box) > 0:
		result.append(box[0])
		if len(box) == 1: break
		ious=iou(box[0:1, 0:4], box[1:, 0:4])
		box=box[1:][ious.squeeze(0) < 0.5]
	return torch.stack(result)

def deal(boxes):
	labels = boxes[:, -2].unique()
	result=[]
	for l in labels:
		box = boxes[boxes[:, -2]==l]
		box = nms(box)
		for b in box: 
			result.append(b)
	return torch.stack(result)

classes=[]
anchors=[[44, 43,  87, 39,  64,102], [20, 18,  43, 21,  28, 34]]
for line in open('/home/lwd/code/darknet/data/coco.names'):
	classes.append(line[:-1])
net=Tiny()
net.load_darknet('/home/lwd/code/darknet/backup/yolov3-tiny_best.weights')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.cuda()
net.eval()
with open('log.txt', 'w') as f:
	for line in open('/home/lwd/data/20220523.txt'):
		print(line[:-1])
		raw = Image.open(line[:-1])
		ih, iw = np.shape(raw)[0:2]
		# inference
		raw = raw.convert('RGB')
		image = raw.resize((416, 416))
		image = np.array(image, dtype='float32') / 255.0
		image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
		with torch.no_grad():
			images = torch.from_numpy(image)
			#images = images[:,[2,1,0],:,:]
			images = images.cuda()
			outputs = net(images)
		
		draw = ImageDraw.Draw(raw)
		thld_boxes=[]
		for i,output in enumerate(outputs):
			# decode output
			boxes = get_boxes(output, anchors[i])
			if len(boxes) == 0: continue
			boxes[:,0] = boxes[:,0] * iw
			boxes[:,1] = boxes[:,1] * ih
			boxes[:,2] = boxes[:,2] / 416.0 * iw
			boxes[:,3] = boxes[:,3] / 416.0 * ih
			for b in boxes:
				thld_boxes.append(b)
		if len(thld_boxes) != 0: 
			# nms
			boxes=deal(torch.stack(thld_boxes))
			for b in boxes:
				cx = b[0]
				cy = b[1]
				w = b[2]
				h = b[3]
				draw.rectangle([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
				draw.text((cx-w/2, cy-h/2+11), classes[int(b[4])], fill="#FF0000")
				f.write(classes[int(b[4])]+' '+str(b[5].item())+'\n')
		del draw
		raw.save('result/image/'+line[:-1].split('/')[-1])
```

