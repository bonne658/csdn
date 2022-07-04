## 一、安装
- pytorch==1.7.1
- torchvision-0.8.2
- cuda10.1
```shell
git clone https://github.com/Megvii-BaseDetection/YOLOX
conda create -n yolox3.7 python=3.7
source activate yolox3.7
cd YOLOX
git clone https://github.com/NVIDIA/apex
cd apex
pip install https://download.pytorch.org/whl/cu101/torch-1.7.1%2Bcu101-cp37-cp37m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu101/torchvision-0.8.2%2Bcu101-cp37-cp37m-linux_x86_64.whl
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
pip3 install -r requirements.txt
python3 setup.py install
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi#subdirectory=PythonAPI'
pip install loguru thop tabulate
pip install opencv-python
```
- 问题`cannot import name 'PILLOW_VERSION' from 'PIL'`，方法`conda install pillow==6.1`

## 二、DEMO
- [下载](https://download.csdn.net/download/random_repick/85423676)预训练文件，目前github上的有问题
```
python tools/demo.py image -f exps/default/yolox_tiny.py -c yolox_tiny.pth --path image.png  --conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu
```
- 问题`没有yolox模块`，方法`在demo.py里最前面加上import sys和sys.path.append("/home/lwd/code/dl/YOLOX")`
- 问题`torch.nn‘ has no attribute ‘SiLU‘`，方法`对应行改成module = SiLU()`
- 问题`yolox_tiny.pth is a zip archive`，是因为pytorch版本低于1.6，遂决定使用cuda10.1

## 三、训练自己的数据集
- 参考[这篇](https://blog.csdn.net/weixin_38353277/article/details/121380027)，使用VOC格式训练
- 先使用labelme给数据打好标签，然后在图片和json文件的目录下运行以下代码
- 需要先新建`saved_path`对应的文件夹
- 和参考博文不一样的是，坐标使用int，因为使用浮点数在训练时会出现错误：`invalid literal for int() with base 10`，错误的原因是直接将浮点型的字符串转成int，比如`int('111.0')`，应该是`int(float('111.0'))`。

```python
import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split

labelme_path = "./"              
saved_path = "./MyVOC/VOC0520"                


if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")
    

files = glob(labelme_path + "*.json")
files = [i.split("/")[-1].split(".json")[0] for i in files]

#4. xml
count = 0
for json_file_ in files:
    json_filename = labelme_path + json_file_ + ".json"
    json_file = json.load(open(json_filename,"r",encoding="utf-8"))
    height, width, channels = cv2.imread(labelme_path + json_file_ +".jpg").shape
    with codecs.open(saved_path + "Annotations/"+json_file_ + ".xml","w","utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
        xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>The UAV autolanding</database>\n')
        xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>NULL</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>'+ str(width) + '</width>\n')
        xml.write('\t\t<height>'+ str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["shapes"]:
            points = np.array(multi["points"]).astype(int)
            xmin = min(points[:,0])
            xmax = max(points[:,0])
            ymin = min(points[:,1])
            ymax = max(points[:,1])
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>'+ str(label)+'</name>\n') 
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(json_filename,xmin,ymin,xmax,ymax,label)
                count+=1
        xml.write('</annotation>')
print(str(count)+' boxes')       
#5.
image_files = glob(labelme_path + "*.jpg")
print("copy image files to VOC/JPEGImages/")
for image in image_files:
    shutil.copy(image,saved_path +"JPEGImages/")
    
#6.split files for txt
txtsavepath = saved_path + "ImageSets/Main/"
ftrainval = open(txtsavepath+'/trainval.txt', 'w')
#ftest = open(txtsavepath+'/test.txt', 'w')
ftrain = open(txtsavepath+'/train.txt', 'w')
fval = open(txtsavepath+'/val.txt', 'w')
total_files = glob("./MyVOC/Annotations/*.xml")
total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
#test_filepath = ""
for file in total_files:
    ftrainval.write(file + "\n")
#test
#for file in os.listdir(test_filepath):
#    ftest.write(file.split(".jpg")[0] + "\n")
#split
train_files,val_files = train_test_split(total_files,test_size=0.15,random_state=42)
#train
for file in train_files:
    ftrain.write(file + "\n")
#val
for file in val_files:
    fval.write(file + "\n")

ftrainval.close()
ftrain.close()
fval.close()
#ftest.close()
```
- 然后`ln -s MyVOC /home/lwd/code/dl/YOLOX/datasets/MyVOC`(换成自己的路径哦)，目录结构如下图
![在这里插入图片描述](https://img-blog.csdnimg.cn/e551fe17492e4dd69c836b4f723191f9.png)
- 接下来修改代码
  - cp exps/default/yolox_tiny.py exps/example/yolox_voc/yolox_voc_tiny.py
  - gedit exps/example/yolox_voc/yolox_voc_tiny.py
  - 在init函数里加`self.num_classes = 3`，换成你的类别数
  - 把`exps/example/yolox_voc/yolox_voc_s.py`里init函数之外的东西复制过来。
  - 修改训练集路径为`data_dir=os.path.join("/home/lwd/code/dl/YOLOX/datasets", "MyVOC"),`和`image_sets=[('0520', 'trainval')],`。
  - 修改测试集路径为`data_dir=os.path.join("/home/lwd/code/dl/YOLOX/datasets", "MyVOC"),`和`image_sets=[('0520', 'val')],`
  - 再修改类别名称
  - gedit yolox/data/datasets/voc_classes.py
  - 把`VOC_CLASSES`改成自己的
- 执行`python setup.py install`编译yolox
- 开始训练`python tools/train.py -f exps/example/yolox_voc/yolox_voc_tiny.py -d 1 -b 2 --fp16 -c yolox_tiny.pth`
- 训练结果在`YOLOX_outputs/yolox_voc_tiny`中

## 四、验证训练结果
- gedit yolox/data/datasets/\_\_init\_\_.py
- 加上`from .voc_classes import VOC_CLASSES`
- gedit tools/demo.py
- 把里面的`COCO_CLASSES`全改成`VOC_CLASSES`
- python setup.py install
- python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_tiny.py -c /home/lwd/code/dl/YOLOX/YOLOX_outputs/yolox_voc_tiny/best_ckpt.pth --path 0012.jpg  --conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu

## 五、批量验证图片
- 修改`tools/demo.py`中`if args.demo == "image":`部分：

```python
if args.demo == "image":
        lwd=os.listdir(args.path)
        for li in lwd:
        	image_demo(predictor, vis_folder, args.path+'/'+li, current_time, args.save_result)
```
- python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_tiny.py -c /home/lwd/code/dl/YOLOX/YOLOX_outputs/yolox_voc_tiny/best_ckpt.pth --path /home/lwd/data/20220523   --conf 0.15 --nms 0.45 --tsize 640 --save_result --device gpu
- 上面的命令改变了`path`参数的值，填你要测试的图片所在的文件夹
