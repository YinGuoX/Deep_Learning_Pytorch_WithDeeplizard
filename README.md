# Deep_Learning_Pytorch_WithDeeplizard
* 主要来自Deeplizard的[Neural Network Programming - Deep Learning with PyTorch](https://deeplizard.com/learn/video/v5cngxo4mIg)
* 是Deeplizard的中文+Colab版

##### 1_PyTorch Prerequisites - Neural Network Programming Series
* 简单的介绍这一系列文章所需要的前置知识和将获取的知识
##### 2_PyTorch Explained - Python Deep Learning Neural Network API
* 简单的介绍了pytorch的历史、特性、优势
##### 3_PyTorch Install - Quick And Easy
* pytorch的简单安装(建议百度即可)
* 查看GPU是否可用
  * torch.cuda.is_available
* 查看pytorch的版本   
  * torch.version.cuda
##### 4_CUDA Explained - Why Deep Learning Uses GPUs
* 什么是GPU？
* 什么是CUDA？
  * CUDA是与GPU硬件配对的软件平台，能够更容易使用GPU的并行处理能力来加速计算
  * cuDNN是CUDA专门处理深度神经网络的库
  * pytorch使用CUDA十分容易
    *　t=t.cuda()
    *　t.to(device)也可
##### 5_Tensors Explained - Data Structures Of Deep Learning
* 什么是Tensor?
  * 从数学上理解：标量、向量、矩阵、n维向量
  * 从计算机上理解：数值、数组、二维数组、n维数组
* Tensor是个泛化的概念
###### 6_Rank, Axes, And Shape Explained - Tensors For Deep Learning
* Tensor的三个属性：rank、axes、shape
  * rank
    * 表示tensor中存在的维数
    * 告诉我们需要多少个索引才可以引用张量内特定的元素 
  * axes
    * 表示张量的一个特定维度
    * rank的值就告诉了我们有多少个axes
    * 每个axes的长度告诉了我们沿此axes有多少个索引可以使用
  * shape(t.size()==t.shape)
    * 表示每个axes的长度
    * 可以表示rank、axes、axes长度的所有需要用的到信息  

##### 7_CNN Tensor Shape Explained - Convolutional Neural Networks And Feature Maps
* 在卷积神经网络中输入形状的解释
  *  一般一批图像的形状都是以下几种格式
    * [Batch, Channels, Height, Width]
    * NCHW
    * NHWC
    * CHWN

##### 8_PyTorch Tensors Explained - Neural Network Programming
* tensor的基本使用
  * tensor的属性
    * t.dtype
    * t.device
    * t.layout
  * 创建tensor
    * torch.Tensor(data)
    * torch.tensor(data)
    * torch.as_tensor(data)
    * torch.from_numpy(data)
    * torch.eye()
    * torch.zeros()
    * torch.ones()
    * torch.rand()
##### 9_Creating PyTorch Tensors For Deep Learning - Best Options
* 创建tensor的最佳选择
  * torch.Tensor(data)
    * 是torch.Tensor的构造函数
    * 创建时使用默认的dtype(torch.get_default_dtype())=>无法自己指定dtype
    * 是copy数据，而不是share数据(share意味着data改变,tensor也会改变)
  * torch.tensor(data) 推荐使用
    * 是返回torch.Tensor对象的函数
    * 可以自己指定dtype和自己配置各种信息
    * 是copy数据，而不是share数据(share意味着data改变,tensor也会改变)
  * torch.as_tensor(data)=>节省内存,推荐使用
    * 可以接受各种类似数组的对象和张量
    * share数据 
  * torch.from_numpy()
    * 只接受numpy.ndarrays类型的数据
    * share数据  
* 注意： 
  * 因为numpy.ndarray对象是在CPU上分配的，当使用GPU时，as_tensor（）函数必须将数据从CPU复制到GPU。
  * as_tensor（）的内存共享不适用于列表之类的内置Python数据结构。
  * as_tensor（）调用要求开发人员了解共享功能。这是必要的，这样我们就不会无意中对基础数据进行不必要的更改，从而影响多个对象。
  * 如果两个进程之间对numpy.ndarray和tensor对象有大量的来回操作，那么as_tensor（）的性能改进会更大。但是，如果只有一个加载操作，从性能的角度看应该不会有太大的影响。 
