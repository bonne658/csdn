- 有发现gather就可以实现，哭笑.jpg
```python
import torch

a = torch.arange(24).view(4,6)
b = torch.tensor([4,0,5,2])
c = a.gather(dim=-1,index=b.unsqueeze(1)).squeeze(1)
```
- 以下是原文
```python
import torch

a = torch.arange(24).view(4,6)
print(a)
b = torch.tensor([4,0,5,2])
mask = torch.zeros_like(a)
bb = b.unsqueeze(1)
print(bb)
mask = mask.scatter_(1, bb, torch.ones_like(bb))
res = a[mask.bool()]
print(res)
```
- 输出

```python
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]])
tensor([[4],
        [0],
        [5],
        [2]])
tensor([ 4,  6, 17, 20])
```

- 另外，还有对pytorch维度的理解：从`unsqueeze`可以看出，对n维向量，从外到里是从0到n-1。[这篇](https://blog.csdn.net/random_repick/article/details/125906013?spm=1001.2014.3001.5502)也有维度的事情，可以看看，互相参照，加强理解和记忆。
