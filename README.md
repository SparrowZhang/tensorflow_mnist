# tensorflow_mnist
实现了一下tensorflow官网的教程[c2s02_minist_pros.tex](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/tex_pdf/tutorials/c2s02_minist_pros.tex)中mnist库的训练识别。
<br><br>可以保存训练的网络参数，以及读取现存的参数继续训练，并实现单张图片识别获取识别的标签。
<br><br>还修改了input_data.py中的read_data_sets2函数，直接读取本地图片文件制作训练集、验证集以及测试集。
<br><br>代码中的数据集可以从代码中的数据集可以从代码中的数据集可以从代码中的数据集可以通过百度网盘从[这里](http://pan.baidu.com/s/1ctbTTG)下载。下载之后解压到当前路径即可。
<br><br>[saveModel.py](https://github.com/SparrowZhang/tensorflow_mnist/blob/master/saveModel.py)中第一阶段的训练是代码中初始化参数，从0开始训练；第二阶段的训练可以直接读取第一阶段中保存的参数继续训练。
<br><br>[readModel.py](https://github.com/SparrowZhang/tensorflow_mnist/blob/master/readModel.py)是测试保存的参数，可以读单个的图片来进行分类，也可以读一系列图片进行分类。
