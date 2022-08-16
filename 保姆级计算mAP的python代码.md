- 处理数据集中的每一副图片，判断图片中的预测框是TP还是FP
  - 预测框按confidence降序排序
  - 对预测框中的每个类别：计算预测框和真实框的iou，取iou最大值满足阈值的，去重，得到TP

```python
def iou(a, b):
	A=len(a)
	B=len(b)
	# 计算a的面积并扩展成A×B
	area1=a[:,2]*a[:,3]
	area1=np.repeat(area1.reshape(A,1), B, 1)
	# 计算b的面积并扩展成A×B
	area2=b[:,2]*b[:,3]
	area2=np.repeat(area2.reshape(1,B), A, 0)
	# a转化成左上右下格式
	aa=np.zeros((A,4))
	aa[:,0:2]=a[:,0:2]-a[:,2:4]/2.0
	aa[:,2:4]=aa[:,0:2]+a[:,2:4]
	aa=np.repeat(aa.reshape(A,1,4), B, 1)
	# b转化成左上右下格式
	bb=np.zeros((B,4))
	bb[:,0:2]=b[:,0:2]-b[:,2:4]/2.0
	bb[:,2:4]=bb[:,0:2]+b[:,2:4]
	bb=np.repeat(bb.reshape(1,B,4), A, 0)
	# 求交集的左上
	lt=np.maximum(aa[:,:,0:2], bb[:,:,0:2])
	# 求交集的右下
	rb=np.minimum(aa[:,:,2:4], bb[:,:,2:4])
	inter=rb-lt
	# 右下小于左上，设0
	inter[inter<0]=0
	# 计算交集的面积，inter大小为A×B
	inter=inter[:,:,0]*inter[:,:,1]
	# 返回值的形状是A×B
	return inter/(area1+area2-inter)

'''
判断一副图片是TP或FP
preds : 数组，经过后处理的预测框，N×6，(x,y,w,h,class_id,confidence)
gts : 数组，真实框，M×5，(x,y,w,h,class_id)
sums : 长度为类别数的列表，sums[i]是某个类别的列表，sums[i][j]是(confidence, a), a=1 if TP else 0
thld : iou阈值
return :更新后的sums
'''
def deal_image(preds, gts, sums, thld):
	### 按confidence降序排序 ###
	preds = preds[preds[:,-1].argsort()[::-1]]
	# 预测框的class_id转化成整型
	class_id = preds[:,-2].astype(int)
	# 当前预测框中的类别
	cls = np.unique(class_id)
	# 所有类别数，用于判断预测框的class_id是否超出范围
	cls_all = len(sums)
	# 遍历当前预测框中的类别
	for c in cls:
		if c >= cls_all: continue
		# 取类别c的pred和gt
		pred = preds[class_id == c]
		# 图中没有任何目标，直接赋值0
		if len(gts) < 1:
			for j in range(len(pred)):
				sums[c].append([pred[j][-1], 0.0])
			continue
		gt = gts[gts[:,-1] == c]
		# 图中没有当前目标，直接赋值0
		if(len(gt) == 0): 
			for j in range(len(pred)):
				sums[c].append([pred[j][-1], 0.0])
			continue
		# 计算iou
		ious=iou(pred, gt)
		### 保证一个gt对应一个pred ###
		# 每个预测框与真实框最大的iou
		ma = ious.max(1)
		# 最大iou的索引，也是对应真实框的序号
		magt = ious.argmax(1)
		# iou不满足阈值的设-1
		magt[ma < thld] = -1
		# 去除重复的真实框序号
		for i in range(len(magt)):
			if i == 0 or magt[i] == -1: continue
			# 当前序号在前面出现过，设-1
			if magt[i] in magt[0:i]: 
				magt[i] = -1
		# 真实序号(0,1,2...)设1
		magt[magt > -1] = 1
		# -1设0
		magt[magt < 0] = 0
		# 赋值
		items = np.zeros((len(pred), 2))
		# confidence
		items[:,0]=pred[:,-1]
		# magt是1的items设为1，就是满足阈值的预测框设为TP
		items[:,1][magt.astype(bool)] = 1
		# 放入对应位置
		for conf, tfp in items:
			sums[c].append([conf, tfp])
	return sums
```

- 根据整个数据集的处理结果，计算每一类的precision，recall，然后计算AP，再用类别平均得到mAP
  - 对每个类别：按confidence降序排序、计算precision和recall、处理precision使每个值等于右侧（包括自己）的最大值、计算AP

```python
'''
sums : 长度为类别数的列表，sums[i]是某个类别的列表，sums[i][j]是(confidence, a), a=1 if TP else 0
gt_nums : 长度为类别数的列表，gt_nums[i]是是某个类别的gt总数
return : mAP
'''	
def get_map(sums, gt_nums):
	exist_class = 0
	mAP = 0
	# 遍历每个类别
	for i in range(len(sums)):
		if gt_nums[i] == 0:
			# 没有gt却有预测框，仅增加分母
			if len(sums[i]) != 0:
				exist_class += 1
			continue
		# 有gt却没有预测框，仅增加分母
		if len(sums[i]) == 0:
			exist_class += 1
			continue
		# 该类的所有预测框的(confidence, a), a=1 if TP else 0
		items = np.array(sums[i])
		# 按confidence降序排序
		items = items[items[:,0].argsort()[::-1]]
		# 取排序后的TP或FP
		tps = items[:,1]
		# 计算precision，recall
		precision=np.zeros(len(tps))
		recall=np.zeros(len(tps))
		tp = 0
		for j in range(len(tps)):
			tp += tps[j]
			precision[j] = tp / (j + 1)
			recall[j] = tp / gt_nums[i]
		# precision设为右侧元素的最大值
		for j in range(len(tps)-2, -1, -1):
			precision[j] = np.max([precision[j], precision[j+1]])
		AP = 0
		# recall前面加0，方便计算
		recall = np.append(0, recall)
		print('class: '+str(i)+'  num: '+str(gt_nums[i]))
		np.set_printoptions(precision=3)
		print(precision)
		print(recall)
		# 计算AP
		for j in range(len(tps)):
			AP += precision[j] * (recall[j+1] - recall[j])
		# 分子更新
		mAP += AP
		# 分母更新
		exist_class += 1
	return mAP / exist_class
```

- 计算recall时需要整个数据集的真实框数量，单独计算

```python
def get_gt_num(gts, gt_nums):
	if len(gts) == 0:
		return gt_nums
	# 真实框的class_id转化成整型
	class_id = gts[:,-1].astype(int)
	# 当前真实框中的类别
	cls = np.unique(class_id)
	for c in cls:
		# 取类别c的gt
		gt = gts[class_id == c]
		# 类别c的gt总数更新
		gt_nums[c] += len(gt)
	return gt_nums
```

- 全部代码

```python
# -*- coding: utf-8 -*-
import numpy as np

def get_gt_num(gts, gt_nums):
	if len(gts) == 0:
		return gt_nums
	# 真实框的class_id转化成整型
	class_id = gts[:,-1].astype(int)
	# 当前真实框中的类别
	cls = np.unique(class_id)
	for c in cls:
		# 取类别c的gt
		gt = gts[class_id == c]
		# 类别c的gt总数更新
		gt_nums[c] += len(gt)
	return gt_nums

def iou(a, b):
	A=len(a)
	B=len(b)
	# 计算a的面积并扩展成A×B
	area1=a[:,2]*a[:,3]
	area1=np.repeat(area1.reshape(A,1), B, 1)
	# 计算b的面积并扩展成A×B
	area2=b[:,2]*b[:,3]
	area2=np.repeat(area2.reshape(1,B), A, 0)
	# a转化成左上右下格式
	aa=np.zeros((A,4))
	aa[:,0:2]=a[:,0:2]-a[:,2:4]/2.0
	aa[:,2:4]=aa[:,0:2]+a[:,2:4]
	aa=np.repeat(aa.reshape(A,1,4), B, 1)
	# b转化成左上右下格式
	bb=np.zeros((B,4))
	bb[:,0:2]=b[:,0:2]-b[:,2:4]/2.0
	bb[:,2:4]=bb[:,0:2]+b[:,2:4]
	bb=np.repeat(bb.reshape(1,B,4), A, 0)
	# 求交集的左上
	lt=np.maximum(aa[:,:,0:2], bb[:,:,0:2])
	# 求交集的右下
	rb=np.minimum(aa[:,:,2:4], bb[:,:,2:4])
	inter=rb-lt
	# 右下小于左上，设0
	inter[inter<0]=0
	# 计算交集的面积，inter大小为A×B
	inter=inter[:,:,0]*inter[:,:,1]
	# 返回值的形状是A×B
	return inter/(area1+area2-inter)

'''
判断一副图片是TP或FP
preds : 数组，经过后处理的预测框，N×6，(x,y,w,h,class_id,confidence)
gts : 数组，真实框，M×5，(x,y,w,h,class_id)
sums : 长度为类别数的列表，sums[i]是某个类别的列表，sums[i][j]是(confidence, a), a=1 if TP else 0
thld : iou阈值
return :更新后的sums
'''
def deal_image(preds, gts, sums, thld):
	### 按confidence降序排序 ###
	preds = preds[preds[:,-1].argsort()[::-1]]
	# 预测框的class_id转化成整型
	class_id = preds[:,-2].astype(int)
	# 当前预测框中的类别
	cls = np.unique(class_id)
	# 所有类别数，用于判断预测框的class_id是否超出范围
	cls_all = len(sums)
	# 遍历当前预测框中的类别
	for c in cls:
		if c >= cls_all: continue
		# 取类别c的pred和gt
		pred = preds[class_id == c]
		# 图中没有任何目标，直接赋值0
		if len(gts) < 1:
			for j in range(len(pred)):
				sums[c].append([pred[j][-1], 0.0])
			continue
		gt = gts[gts[:,-1] == c]
		# 图中没有当前目标，直接赋值0
		if(len(gt) == 0): 
			for j in range(len(pred)):
				sums[c].append([pred[j][-1], 0.0])
			continue
		# 计算iou
		ious=iou(pred, gt)
		### 保证一个gt对应一个pred ###
		# 每个预测框与真实框最大的iou
		ma = ious.max(1)
		# 最大iou的索引，也是对应真实框的序号
		magt = ious.argmax(1)
		# iou不满足阈值的设-1
		magt[ma < thld] = -1
		# 去除重复的真实框序号
		for i in range(len(magt)):
			if i == 0 or magt[i] == -1: continue
			# 当前序号在前面出现过，设-1
			if magt[i] in magt[0:i]: 
				magt[i] = -1
		# 真实序号(0,1,2...)设1
		magt[magt > -1] = 1
		# -1设0
		magt[magt < 0] = 0
		# 赋值
		items = np.zeros((len(pred), 2))
		# confidence
		items[:,0]=pred[:,-1]
		# magt是1的items设为1，就是满足阈值的预测框设为TP
		items[:,1][magt.astype(bool)] = 1
		# 放入对应位置
		for conf, tfp in items:
			sums[c].append([conf, tfp])
	return sums

'''
sums : 长度为类别数的列表，sums[i]是某个类别的列表，sums[i][j]是(confidence, a), a=1 if TP else 0
gt_nums : 长度为类别数的列表，gt_nums[i]是是某个类别的gt总数
return : mAP
'''	
def get_map(sums, gt_nums):
	exist_class = 0
	mAP = 0
	# 遍历每个类别
	for i in range(len(sums)):
		if gt_nums[i] == 0:
			# 没有gt却有预测框，仅增加分母
			if len(sums[i]) != 0:
				exist_class += 1
			continue
		# 有gt却没有预测框，仅增加分母
		if len(sums[i]) == 0:
			exist_class += 1
			continue
		# 该类的所有预测框的(confidence, a), a=1 if TP else 0
		items = np.array(sums[i])
		# 按confidence降序排序
		items = items[items[:,0].argsort()[::-1]]
		# 取排序后的TP或FP
		tps = items[:,1]
		# 计算precision，recall
		precision=np.zeros(len(tps))
		recall=np.zeros(len(tps))
		tp = 0
		for j in range(len(tps)):
			tp += tps[j]
			precision[j] = tp / (j + 1)
			recall[j] = tp / gt_nums[i]
		# precision设为右侧元素的最大值
		for j in range(len(tps)-2, -1, -1):
			precision[j] = np.max([precision[j], precision[j+1]])
		AP = 0
		# recall前面加0，方便计算
		recall = np.append(0, recall)
		print('class: '+str(i)+'  num: '+str(gt_nums[i]))
		np.set_printoptions(precision=3)
		print(precision)
		print(recall)
		# 计算AP
		for j in range(len(tps)):
			AP += precision[j] * (recall[j+1] - recall[j])
		# 分子更新
		mAP += AP
		# 分母更新
		exist_class += 1
	return mAP / exist_class
	
if __name__ == '__main__':
	a=np.array([1,1,2,2, 2,2,4,4, 2,3,2,4, 0,0,2,2]).reshape(4,4)
	print(iou(a,a))
	'''
	[[1.         0.25       0.09090909 0.14285714]
 	[0.25       1.         0.33333333 0.05263158]
 	[0.09090909 0.33333333 1.         0.        ]
 	[0.14285714 0.05263158 0.         1.        ]]
 	'''
```

- 使用方式

```python
sums=[]
gt_nums=[]
for pi in range(80): 
	sums.append([])
	gt_nums.append(0)
for 每幅图片:
	ih, iw = np.shape(图片)[0:2]
	gts=[]
	# darknet数据格式
	label_file=图片路径[:-4]+'txt'
	for lf in open(label_file):
		lfs=lf.split()
		gts.append([float(lfs[1])*iw, float(lfs[2])*ih, float(lfs[3])*iw, float(lfs[4])*ih, float(lfs[0])])
	gt_nums = get_map.get_gt_num(np.array(gts), gt_nums)
	前向传播获得输出，再后处理得到最后的预测框boxes（假设在GPU上）
	sums = get_map.deal_image(boxes.cpu().numpy(), np.array(gts), sums, 0.5)
print(get_map.get_map(sums, gt_nums))
```

