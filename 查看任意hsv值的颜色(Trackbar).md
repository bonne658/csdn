```python
import cv2
import numpy as np

def nothing(*arg):
	pass

cv2.namedWindow("hsv", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("h", "hsv", 0, 180, nothing)
cv2.createTrackbar("s", "hsv", 127, 255, nothing)
cv2.createTrackbar("v", "hsv", 127, 255, nothing)
im=np.zeros((300,300,3), 'uint8')
while 1:
	h = cv2.getTrackbarPos('h', 'hsv')
	s = cv2.getTrackbarPos('s', 'hsv')
	v = cv2.getTrackbarPos('v', 'hsv')
	im[...,0]=h
	im[...,1]=s
	im[...,2]=v
	im=cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
	cv2.imshow('hsv', im)
	ch=cv2.waitKey(5)
	if ch == 27:  # esc
		break
```
- 轻微改动就能变成查看任意rgb值的颜色哦
