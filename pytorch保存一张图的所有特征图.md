- 需要先了解所用网络的结构
```python
import torch.nn as nn
import torch

class Simple(nn.Module):
	def __init__(self):
		super().__init__()
		self.feature = nn.Sequential(
			nn.Conv2d(3, 16, 3, 1, 1, bias=True),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, 3, 1, 1, bias=True),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 3, 1, 1, bias=True),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, 3, 1, 1, bias=True),
			nn.MaxPool2d(2),
			nn.Conv2d(128, 256, 3, 1, 1, bias=True),
			nn.MaxPool2d(2),
			nn.Conv2d(256, 512, 3, 1, 1, bias=True)
		)
		self.fc = nn.Sequential(
			nn.Linear(512*16, 1000),
			nn.ReLU(),
			nn.Linear(1000, 100),
			nn.ReLU(),
			nn.Linear(100, 6)
		)
		
	def forward(self, x):
		f = self.feature(x)
		f = f.view(-1, 512*16)
		return self.fc(f)
```
- 加载模型，预处理图片，记得要保证存储目录存在哦～

```python
net=Simple()
paras=torch.load('', map_location='cuda')
net.load_state_dict(paras)
net.cuda().eval()
raw = cv2.imread('')
im = cv2.resize(raw, (150, 150))
im = im / 255.0
im = np.transpose(im, (2,0,1))
im = im.reshape(1,3,150,150)
im = torch.from_numpy(im).float().cuda()
for ch in net.feature.children():
		im = ch(im)
		b, c, h, w = im.shape
		for ci in range(c):
			fm = im[0][ci]
			ma = torch.max(fm)
			mi = torch.min(fm)
			fm = 255 * (fm - mi) / (ma - mi)
			fm = fm.cpu().detach().numpy().astype('uint8')
			fm = fm.reshape(h, w, 1)
			cv2.imwrite('log/feature/{}-{}.jpg'.format(h, ci), fm)
```
