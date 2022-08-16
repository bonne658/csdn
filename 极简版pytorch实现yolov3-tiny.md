- 参考`https://github.com/bubbliiiing/yolo3-pytorch`
- `train.py`流程：加载数据`dataloader.py`正向传播`tiny.py`反向传播`loss.py`
- dataloader.py

```python
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.length             = len(self.annotation_lines)
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        # 归一化（除以255），whc转chw
        image       = np.transpose(np.array(image, dtype=np.float32)/255.0, (2, 0, 1))
        # 左上右下形式
        box         = np.array(box, dtype=np.float32)
        
        if len(box) != 0:
            # 转化成比例形式
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            # 转化成中心+宽高形式
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.02, sat=1.5, val=1.5, random=True):
        line    = annotation_line.split()
        # 标签：中心+宽高形式
        label_line = line[0][:-4]+'.txt'
        boxes = []
        for lin in open(label_line):
            t = lin.split()
            boxes.append([t[1],t[2],t[3],t[4],t[0]])
        box = np.array(boxes, dtype=np.float32)
        # 图像
        image   = Image.open(line[0])
        iw, ih  = image.size
        h, w    = input_shape
        if len(box) > 0:
        	# 转化成数字形式
        	box[:, [0,2]] = box[:, [0,2]] * iw
        	box[:, [1,3]] = box[:, [1,3]] * ih
        	# 转化成左上右下形式
        	box[:, 0:2] = box[:, 0:2] - box[:, 2:4] / 2
        	box[:, 2:4] = box[:, 0:2] + box[:, 2:4]
        # 验证
        if not random:
            # 计算图片等比例缩放到输入大小的宽高，可能有一个小于输入尺寸
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            # 嵌入坐标
            dx = (w-nw)//2
            dy = (h-nh)//2
            # 放缩
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            # 嵌入在中间
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)
            if len(box)>0:
                np.random.shuffle(box)
                # 将标签转换到新图片
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                # 左上不小于0
                box[:, 0:2][box[:, 0:2]<0] = 0
                # 右下不大于宽高
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                # 宽高要大于一个像素
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] 
            # 标签形式：数字，左上右下
            return image_data, box
                
        # 训练
        # 宽高的新比率
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        # 相对输入尺寸的放缩比例
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        # 放缩
        image = image.resize((nw,nh), Image.BICUBIC)
        # 随机一个嵌入坐标
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        # 嵌入
        new_image.paste(image, (dx, dy))
        image = new_image
        # 翻转图像
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # 色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        if len(box)>0:
            np.random.shuffle(box)
            # 将标签转换到新图片
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            # 左上不小于0
            box[:, 0:2][box[:, 0:2]<0] = 0
             # 右下不大于宽高
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            # 宽高要大于一个像素
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        # 标签形式：数字，左上右下
        return image_data, box
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes


```

- tiny.py在[这里](https://blog.csdn.net/random_repick/article/details/126261694?spm=1001.2014.3001.5502)
- loss.py

```python
import torch, math, sys
import numpy as np
import torch.nn as nn

def iou(a,b):
	A=len(a)
	B=len(b)
	area1=a[:,2]*a[:,3]
	area1=area1.unsqueeze(1).expand(A,B)
	area2=b[:,2]*b[:,3]
	area2=area2.unsqueeze(0).expand(A,B)
	aa=torch.zeros_like(a)
	aa[:,0:2]=a[:,0:2]-a[:,2:4]/2
	aa[:,2:4]=aa[:,0:2]+a[:,2:4]
	aa=aa.unsqueeze(1).expand(A,B,4)
	bb=torch.zeros_like(b)
	bb[:,0:2]=b[:,0:2]-b[:,2:4]/2
	bb[:,2:4]=bb[:,0:2]+b[:,2:4]
	bb=bb.unsqueeze(0).expand(A,B,4)
	lt=torch.max(aa[:,:,0:2], bb[:,:,0:2])
	rb=torch.min(aa[:,:,2:4], bb[:,:,2:4])
	inter=torch.clamp((rb-lt), min=0)
	inter=inter[:,:,0]*inter[:,:,1]
	return inter/(area1+area2-inter)

def clip(a):
	mi=1e-7
	ma=1-mi
	b=(a>=mi).float()*a+(a<mi).float()*mi
	b=(b<=ma).float()*b+(b>ma).float()*ma
	return b
	
def BCELoss(pred, target):
	p=clip(pred)
	return -target*torch.log(p)-(1-target)*torch.log(1-p)
	
def MSELoss(pred, target):
	return torch.pow((pred-target), 2)

class Loss(nn.Module):
	def __init__(self, input_size, anchors, classes, anchors_mask=[[0,1,2], [3,4,5]]):
		super().__init__()
		self.input_size = input_size
		self.anchors = anchors
		self.bbox_attrs = 5 + classes
		self.anchors_mask = anchors_mask
		self.ignore_threshold = 0.5
		
	'''
	l:            第l组anchors_mask
	out：         b*255*h*w， 网络输出之一
	targets:      b*N*5，比例形式的gt
	'''
	def forward(self, l, out, target):
		b = out.size(0)
		in_h = out.size(2)
		in_w = out.size(3)
		s = self.input_size[0] // in_w
		scaled_anchors = [(aw/s, ah/s) for aw,ah in self.anchors]
		# 正样本
		y_true, no_obj, scale = self.get_target(l, target, scaled_anchors, in_h, in_w)
		scale=2-scale
		out = out.view(b, 3, self.bbox_attrs, in_h, in_w).permute(0,1,3,4,2)
		x = torch.sigmoid(out[...,0])
		y = torch.sigmoid(out[...,1])
		w = out[...,2]
		h = out[...,3]
		# 记得sigmoid
		c = torch.sigmoid(out[...,4])
		cl=torch.sigmoid(out[...,5:])
		# 负样本
		no_obj = self.get_ignore(l,x,y,h,w,target, scaled_anchors, in_h, in_w, no_obj)
		if x.is_cuda:
			y_true = y_true.cuda()
			no_obj = no_obj.cuda()
			scale = scale.cuda()
		# loss
		xloss=torch.sum(BCELoss(x, y_true[...,0])*y_true[...,4]*scale)
		yloss=torch.sum(BCELoss(y, y_true[...,1])*y_true[...,4]*scale)
		wloss=torch.sum(MSELoss(w, y_true[...,2])*y_true[...,4]*scale*0.5)
		hloss=torch.sum(MSELoss(h, y_true[...,3])*y_true[...,4]*scale*0.5)
		closs=torch.sum(BCELoss(c, y_true[...,4])*y_true[...,4] + BCELoss(c, y_true[...,4])*no_obj)
		clsloss=torch.sum(BCELoss(cl[y_true[...,4]==1], y_true[...,5:][y_true[...,4]==1]))
		loss = xloss + yloss + wloss + hloss + closs + clsloss
		num=torch.sum(y_true[...,4])
		num=torch.max(num, torch.ones_like(num))
		# print(torch.sum(y_true[0,...,4]).item())
		# print(torch.sum(y_true[1,...,4]).item())
		#sys.exit()
		return loss, num
		
	'''
	l:            第l组anchors_mask
	targets:      b*N*5，比例形式的gt
	anchors:      9*2，已经放缩过的
	in_h：        特征图高度
	in_w：        特征图宽度
	每个batch：
		N*4的gt和9*4的anchor求iou
		每个gt的最大IOU对应的anchor：
			如果不在当前mask： continue
			否则：gt中心点坐标和anchor序号确定位置，赋值
	'''
	def get_target(self, l, targets, anchors, in_h, in_w):
		b = len(targets)
		c = len(self.anchors_mask[l])
		y_true = torch.zeros(b,c,in_h, in_w,self.bbox_attrs,requires_grad = False)
		no_obj = torch.ones(b,c,in_h, in_w,requires_grad = False)
		scale = torch.zeros(b,c,in_h, in_w,requires_grad = False)
		# 
		for bi in range(b):
			if(len(targets[bi]) == 0): continue
			# gt和anchors以(0,0)为中心计算iou
			batch_target = torch.zeros(len(targets[bi]), 4)
			batch_target[:,2] = targets[bi][:,2] * in_w
			batch_target[:,3] = targets[bi][:,3] * in_h
			anchor4 = torch.zeros(len(anchors), 4)
			anchor4[:,2:] = torch.FloatTensor(anchors)
			ious = iou(batch_target, anchor4)  # N * 9
			bests = torch.argmax(ious, dim=1)  # 每个值在0~8之间
			#print(bests)
			# 1.忘记赋值
			batch_target[:,0] = targets[bi][:,0] * in_w
			batch_target[:,1] = targets[bi][:,1] * in_h
			for it, best in enumerate(bests):
				if best not in self.anchors_mask[l]:
					continue
				c = self.anchors_mask[l].index(best)  # 0~2之间
				# gt中心点所在网格
				i = torch.floor(batch_target[it,0]).long()
				j = torch.floor(batch_target[it,1]).long()
				#print(bi,c,j,i)
				# 赋值
				no_obj[bi,c,j,i] = 0
				y_true[bi,c,j,i,0] = batch_target[it,0] - i.float()
				y_true[bi,c,j,i,1] = batch_target[it,1] - j.float()
				# 2.用错anchors(没放缩的self.anchors)
				y_true[bi,c,j,i,2] = math.log(batch_target[it,2]/anchors[best][0])
				y_true[bi,c,j,i,3] = math.log(batch_target[it,3]/anchors[best][1])
				y_true[bi,c,j,i,4] = 1
				clss=targets[bi][it][4].long()
				y_true[bi,c,j,i,5+clss] = 1
				scale[bi,c,j,i] = batch_target[it,2]*batch_target[it,3]/in_h/in_w
		return y_true, no_obj, scale
		
	'''
	l:            第l组anchors_mask
	x, y, h, w:   b*3*h*w，网络输出，其中x,y已经过sigmoid
	targets:      b*N*5，比例形式的gt
	anchors:      9*2，已经放缩过的
	in_h：        特征图高度
	in_w：        特征图宽度
	no_obj：      b*3*h*w，标记负样本
	将anchors_mask对应的anchors分布到特征图每个网格上，形状是b*3*h*w*2
	将x, y, h, w结合上面的anchors转化并concat成b*3*h*w*4的预测值
	每个batch：
		计算与gt的iou
		取每个预测框的最大iou值
		最大IOU超过阈值的是忽略样本，即no_obj对应的值设为0
	'''
	def get_ignore(self, l, x, y, h, w, targets, anchors, in_h, in_w, no_obj):
		ft = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
		b = len(targets)
		# 转换h，w
		anchor_l = np.array(anchors)[self.anchors_mask[l]]
		anchor_w = ft(anchor_l[:,0:1])
		anchor_h = ft(anchor_l[:,1:])
		anchor_w = anchor_w.repeat(1,in_h*in_w).repeat(b,1).view(b,3,in_h,in_w)
		anchor_h = anchor_h.repeat(1,in_h*in_w).repeat(b,1).view(b,3,in_h,in_w)
		tw = (torch.exp(w.data)*anchor_w).unsqueeze(-1)
		th = (torch.exp(h.data)*anchor_h).unsqueeze(-1)
		# 转换x，y
		grid_y, grid_x = torch.meshgrid(torch.arange(in_w), torch.arange(in_h))
		# tensor可以这样转设备
		grid_x = grid_x.repeat(b,3,1,1).type(ft)
		grid_y = grid_y.repeat(b,3,1,1).type(ft)
		tx = (x.data + grid_x).unsqueeze(-1)
		ty = (y.data + grid_y).unsqueeze(-1)
		# concat
		pred = torch.cat([tx, ty, tw, th], -1)
		for bi in range(b):
			if(len(targets[bi]) == 0): continue
			# 计算iou
			pre = pred[bi].view(-1,4)
			# 形状，设备信息也一样
			gt = torch.zeros_like(targets[bi])
			gt[:,[0,2]] = targets[bi][:,[0,2]] * in_w
			gt[:,[1,3]] = targets[bi][:,[1,3]] * in_h
			gt = gt[:,:4]
			ious=iou(gt, pre)
			# 判断，赋值
			maxx, _ = torch.max(ious, dim=0)
			maxx = maxx.view(3,in_h,in_w)
			no_obj[bi][maxx > self.ignore_threshold] = 0
		return no_obj
```

- train.py

```python
from tiny import Tiny
from loss import Loss
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import numpy as np
import torch, sys, cv2
import torch.optim as optim
from dataloader import YoloDataset, yolo_dataset_collate

def show_batch(image, label):
	for i in range(len(image)):
		im = np.transpose(image[i]*255.0,(1,2,0)).astype('uint8')[:,:,[2,1,0]]
		ih, iw = np.shape(im)[0:2]
		cv2.imshow("im", im)
		cv2.waitKey(0)
		# for lab in label[i]:
		# 	print(lab)

# data
batch_size = 2
data_txt='/home/lwd/data/all.txt'
with open(data_txt) as f:
	train_lines = f.readlines()
train_dataset=YoloDataset(train_lines, (416, 416), True)
train_data = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
test_txt='/home/lwd/data/test.txt'
with open(test_txt) as f:
	test_lines = f.readlines()
test_dataset=YoloDataset(test_lines, (416, 416), False)
test_data = DataLoader(test_dataset, shuffle = False, batch_size = batch_size, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
train_step = len(train_lines) // batch_size
val_step = len(test_lines) // batch_size
# net
model_path=''
net=Tiny()
net.init()
net.load_darknet('/home/lwd/code/darknet/yolov3-tiny.conv.15')
net = net.cuda()

if len(model_path) > 1:
	paras=torch.load(model_path, map_location='cuda')
	net.load_state_dict(paras)
# hyperparameter
anchors = [[44, 43],  [87, 39],  [64,102], [20, 18],  [43, 21],  [28, 34]]
los = Loss((416, 416), anchors, 80)
lr = 1e-4
optimizer = optim.Adam(net.parameters(), lr, weight_decay = 5e-4)
#lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)
# iterator
i = 1
lr_cnt = 0
vl_last = 9
for param in net.backbone.parameters():
	param.requires_grad = False
while True:
	net.train()
	# if i % 111 == 0 and lr > 1e-4:
	# 	lr *= 0.1
	# 	for param_group in optimizer.param_groups:
	# 		param_group["lr"] = lr
	if i == 400:
	# 	optimizer = optim.Adam(net.parameters(), 1e-4, weight_decay = 5e-4)
	# 	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
		for param in net.backbone.parameters():
			param.requires_grad = True
	train_loss = 0
	for bi, (batch_image, batch_label) in enumerate(train_data):
		loss = 0
		number = 0
		#show_batch(batch_image, batch_label)
		batch_image  = torch.from_numpy(batch_image).type(torch.FloatTensor).cuda()
		batch_label = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in batch_label]
		optimizer.zero_grad()
		outputs = net(batch_image)
		for oi, output in enumerate(outputs):
			loss_item, num_pos = los(oi, output, batch_label)
			loss += loss_item
			number += num_pos
		loss_value = loss / number
		loss_value.backward()
		optimizer.step()
		train_loss += loss_value.item()
	net.eval()
	val_loss = 0
	for bi, (batch_image, batch_label) in enumerate(test_data):
		loss = 0
		number = 0
		# show_batch(batch_image, batch_label)
		batch_image  = torch.from_numpy(batch_image).type(torch.FloatTensor).cuda()
		batch_label = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in batch_label]
		optimizer.zero_grad()
		outputs = net(batch_image)
		for oi, output in enumerate(outputs):
			loss_item, num_pos = los(oi, output, batch_label)
			loss += loss_item
			number += num_pos
		loss_value = loss / number
		val_loss += loss_value.item()
	vl=val_loss / val_step
	print('epoch: ', i, ' ------ train_loss:', train_loss / train_step, '   val_loss:', val_loss / val_step)
	print(optimizer.param_groups[0]['lr'])
		
	if vl < vl_last: 
		torch.save(net.state_dict(), 'result/model/'+str(i)+':'+str(vl)[:5]+'.pth')
		vl_last = vl
		#break
	# lr_scheduler.step()
	if i > 999: 
		break
	i += 1
```

- 备注
  - 使用darknet的预训练模型训练
  - 学习率固定1e-4
  - 前400次训练不更新预训练权重
  - 在验证loss=5.5左右得到可用的模型
  - 试了从头训练，loss在9附近降不下去，可能是训练集太小
