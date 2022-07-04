- 就是遍历每一个参数，把大小加起来
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paras=torch.load('weights.pth', map_location=device)
count = 0
for k2 in paras:
	l = list(paras[k2].size())
	temporary = 1
	for i in l:
		temporary *= i
	count += temporary
print(count)
```
