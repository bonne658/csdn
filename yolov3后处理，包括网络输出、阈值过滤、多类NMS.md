- pytorch版本：1.7.1+cu101

```python
from net.yolov3 import Yolo
from PIL import Image, ImageDraw
import numpy as np
import torch, copy, time

def sigmoid(x):
	temporary = 1 + torch.exp(-x)
	return 1.0 / temporary

def get_boxes(output, anchors):
	h = output.size(2)
	w = output.size(3)
	output = output.view(3, 85, h, w).permute(0,2,3,1)
	tc = torch.sigmoid(output[..., 4])    # 3*h*w
	cl = torch.sigmoid(output[..., 5:])   # 3*h*w*80
	clv, cli = torch.max(cl,-1)
	mask = tc * clv > 0.9
	# print(torch.where(mask))
	cli = cli[mask].unsqueeze(-1)
	# 3*h*w
	tx = torch.sigmoid(output[..., 0][mask])
	ty = torch.sigmoid(output[..., 1][mask])
	tw = torch.exp(output[..., 2][mask])
	th = torch.exp(output[..., 3][mask])
	# grid
	FloatTensor = torch.cuda.FloatTensor if tx.is_cuda else torch.FloatTensor
	grid_x, grid_y = torch.meshgrid(torch.linspace(0,w-1,w), torch.linspace(0,h-1,h))
	grid_x = grid_x.repeat(3,1,1)[mask].type(FloatTensor)
	grid_y = grid_y.repeat(3,1,1)[mask].type(FloatTensor)
	tx = ((tx+grid_y) / w).unsqueeze(-1)
	ty = ((ty+grid_x) / h).unsqueeze(-1)
	# anchor
	aw = torch.Tensor(anchors[0::2]).view(3,1).repeat(1,h*w).view(3,h,w)[mask].type(FloatTensor)
	ah = torch.Tensor(anchors[1::2]).view(3,1).repeat(1,h*w).view(3,h,w)[mask].type(FloatTensor)
	tw = (tw * aw).unsqueeze(-1)
	th = (th * ah).unsqueeze(-1)
	return torch.cat([tx, ty, tw, th, cli], -1)

def get_boxes(output, anchors):
	h=output.size(2)
	w=output.size(3)
	output=output.view(3,85,h,w).permute(0,2,3,1)
	# conf
	conf = torch.sigmoid(output[..., 4])
	cl = torch.sigmoid(output[..., 5:])
	clv, cli = torch.max(cl, -1)
	conf = conf * clv
	mask = conf > 0.5
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

# def get_boxes(output, anchors):
# 	c = output.size(1)
# 	h = output.size(2)
# 	w = output.size(3)
# 	boxes = []
# 	output = output.view(3, 85, h, w).permute(0,2,3,1)
# 	for i in range(3):
# 		for j in range(h):
# 			for k in range(w):
# 				temporary = output[i][j][k]
# 				for t in range(5,85):
# 					if(sigmoid(temporary[4]) * sigmoid(temporary[t]) < 0.9): 
# 						continue
# 					x = (sigmoid(temporary[0]) + k) / w
# 					y = (sigmoid(temporary[1]) + j) / h
# 					tw = torch.exp(temporary[2]) * anchors[i*2]
# 					th = torch.exp(temporary[3]) * anchors[i*2+1]
# 					boxes.append([x,y,tw,th,t-5])
# 					print(i,j,k)
# 	return boxes

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
		#print(ious)
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
for line in open('/home/lwd/code/darknet/data/coco.names'):
	classes.append(line[:-1])
model_path='/home/lwd/code/dl/myTorch/yolo/model/99.pth'
net=Yolo(80)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paras=torch.load(model_path, map_location=device)

net.load_state_dict(paras)
net = net.eval()
net = net.cuda()

with open('result/log.txt', 'w') as f:
	for line in open('/home/lwd/data/20220220.txt'):
		f.write(line[:-1].split('/')[-1]+'\n')
		print(line[:-1])
		raw = Image.open(line[:-1])
		ih, iw = np.shape(raw)[0:2]
		image = raw.resize((416, 416), Image.BICUBIC)
		image = np.array(image, dtype='float32') / 255.0
		image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
		with torch.no_grad():
			images = torch.from_numpy(image)
			images = images.cuda()
			outputs = net(images)
		anchors=[[116,90,  156,198,  373,326],[30,61,  62,45,  59,119],[10,13,  16,30,  33,23]]
		draw = ImageDraw.Draw(raw)
		thld_boxes=[]
		for i,output in enumerate(outputs):
			ti = time.time()
			boxes = get_boxes(output, anchors[i])
			if len(boxes) == 0: continue
			boxes[:,0] = boxes[:,0] * iw
			boxes[:,1] = boxes[:,1] * ih
			boxes[:,2] = boxes[:,2] / 416.0 * iw
			boxes[:,3] = boxes[:,3] / 416.0 * ih
			for b in boxes:
				thld_boxes.append(b)
			print(output.size(2), 'time: ', time.time()-ti)
		if len(thld_boxes) == 0: continue
		boxes=deal(torch.stack(thld_boxes))
		print(len(boxes))
		for b in boxes:
				cx = b[0]
				cy = b[1]
				w = b[2]
				h = b[3]
				draw.rectangle([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
				draw.text((cx-w/2, cy-h/2), classes[int(b[4])], fill="#FF0000")
				f.write(classes[int(b[4])]+'\n')
			# if(len(boxes) > 0):break
		del draw
		# raw.show()
		raw.save('result/image/'+line[:-1].split('/')[-1])

```
