- 因为声明的网络模型比保存的模型bn层多一个参数`num_batches_tracked`，所以跳过它
```python
import torch, copy

net=yourNet()
state_dict = copy.deepcopy(net.state_dict())
keys=[]
for key in state_dict:
	if(key.split('.')[-1] == 'num_batches_tracked'):
		continue
	keys.append(key)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paras=torch.load('weights.pth', map_location=device)
for k1,k2 in zip(keys, paras):
	state_dict[k1] = paras[k2]
```
