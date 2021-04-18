# Deep_Learning_Pytorch_WithDeeplizard
* 主要来自Deeplizard的[Neural Network Programming - Deep Learning with PyTorch](https://deeplizard.com/learn/video/v5cngxo4mIg)
* 是Deeplizard的中文+Colab版
## 大纲
* Pytorch的基础知识:tensor(1-13)
* 神经网络的基本流程
  * 准备数据：ETL(15-16)
  * 建立模型：(17-26)
    * 创建一个扩展nn.Module基类的神经网络类
    * 在类构造函数中，将网络的图层定义为类属性
    * 使用网络的图层属性以及nn.functional API操作来定义网络的前向传播
    * 理解批处理如何传递到网络
    * 了解前向传播的具体转换过程
    * 计算损失和梯度，并更新权重 
  * 训练模型(27)
    * 建立训练循环 
  * 分析模型结果(28-30)
    * 建立、绘制、解释混淆矩阵=>预测和真实的比较
    * 使用stack、concate重叠数据形状
    * 使用TensorBoard=>模型训练过程的详细数据迭代过程
  * 超参数调试(31-34)
    * for循环调参
    * 优化for=>RunBuilder类 
    * 优化TensorBoard=>RunManager类
  * 加速优化模型(35-39)
    * num_worker加速模型
    * GPU加速模型  
    * 数据归一化优化模型
    * 批量归一化优化模型
    * 设置随机数种子对比模型
---
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
    * t=t.cuda()
    * t.to(device)也可
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

##### 10_Flatten, Reshape, And Squeeze Explained - Tensors For Deep Learning With PyTorch
* Tensor的reshaping操作
  * 获取张量的形状：
    * t.size()
    * t.shape
  * 获取张量的rank：
    * len(t.shape)
  * 获取张量的元素个数：
    * torch.tensor(t.shape).prod()
    * t.numel()
  * 改变张量的形状
    * t.reshape(xx,xx,xx)
    * t.squeeze()：把张量长度为1的轴删除
    * t.unsqueeze(dim=x)：在维度x上添加一个长度为1的轴       


##### 11_CNN Flatten Operation Visualized - Tensor Batch Processing For Deep Learning
* reshaping操作使其能够成为CNN的一个输入数据
  * torch.stack((t1,t2,t3))后面会详细讨论
  * torch.cat((t1,t2,t3))后面会详细讨论
* flatten一批Tensor
  * t.reshape(1,-1)
  * t.reshape(-1)
  * t.view(t.numel())
  * t.flatten()
  * t.flatten(start_dim=x):指定从哪个轴开始flatten 

##### 12_Tensors For Deep Learning - Broadcasting And Element-Wise Operations With PyTorch
* Tensor的元素操作
  * 传统的加减乘除：+-*/
  * 内置的加减乘除：t.add(t1)、t.sub(x)、t1.mul(x)、t1.div(x)
  * 传统的比较操作：>、<、==、...
  * 内置的比较操作：t.eq(x)、t.ge(x)、t.gt(x)、t.lt(x)、t.le(x)
  * 一些元素操作的函数：t.abs()、t.sqrt()、t.neg()、
* 广播的概念：不同形状的张量在元素操作中的处理方式
  *  np.broadcast_to(2,t1.shape)：将标量值2广播成t1的形状张量
  *  t1+2 == t1+torch.tensor(np.broadcast_to(2,t1.shape))

##### 13_Code For Deep Learning - ArgMax And Reduction Tensor Ops
* Tensor的Reduction操作：对Tensor的部分张量、规约张量的操作
  * t.sum()
  * t.numel()
  * t.prod()
  * t.mean()
  * t.std()
  * t.sum(dim=0)与t.sum(dim=1)的区别
  * t.max()
  * t.argmax()
  * t.mean().item()：获得一个数值
  * t.mean().tolist()
  * t.mean().numpy()
  * 分清当指定dim时是对啥进行操作

##### 14_Dataset For Deep Learning - Fashion MNIST
* 简单介绍了一些数据集的注意事项和MNIST数据集与Fashion-MINIST数据集的来源和组成

##### 15_CNN Image Preparation Code Project - Learn To Extract, Transform, Load (ETL)
* 神经网络流程
  * **准备数据**
  * 建立模型
  * 训练模型
  * 分析模型结果 
* ETL：数据源**抽取**数据、**转换**数据格式、**加载**数据结构
  * 一般可以使用torchvision可以快速对样列数据进行ETL
  * 如：实现抽取转换：
    *  train_set = torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))
  * 实现加载：
    * train_loader =torch.utils.data.DataLoader(train_set,batch_size=1000,shuffle=True)
##### 16_PyTorch Datasets And DataLoaders - Training Set Exploration For Deep Learning And AI
* datasets、DataLoader的一般使用和查看属性
* 如何访问Dataset的数据
  * sample=next(iter(train_set))
  * image,label = sample
  * plt.imshow(image.squeeze(),cmap='gray')
* 如何访问DataLoader的数据
  * batch = next(iter(display_loader))  
  * images,labels = batch
  * 再根据images的shape对数据进行抽取展示即可(有代码)
##### 17_Build PyTorch CNN - Object Oriented Neural Networks
* 神经网络流程
  * 准备数据
  * **建立模型**
  * 训练模型
  * 分析模型结果 
* 面向对象的方式构建模型 
  * nn.Module作为基类
  * 网络的层作为类的属性 

##### 18_CNN Layers - PyTorch Deep Neural Network Architecture
* 理解卷积层的各个参数
  * 超参数：人工选择的参数
    * kernel_size,out_channels,out_features
  * 数据依赖的超参数：取决于数据的参数  
    * in_channels,in_features,out_features(输出层)

##### 19_CNN Weights - Learnable Parameters In PyTorch Neural Networks
* 理解卷积层的权重参数(可学习参数)
  * 访问神经网络的每一层：
    * 点表示法访问对象的属性和方法
    * network.conv1  
  * 访问神经网络的每一层的权重=>conv1也是一个对象，weight是conv层的内部权重张量对象
    * network.conv1.weight =>是一个Parameter类，是一个拓展的tensor类
  * 查看权重的形状
    * network.conv1.weight.shape 
  * 理解矩阵乘法的运算=>前向传播的过程
    * weight_matrix.matmul(in_features)  
  *  访问神经网络的每一层参数
    * for param in network.parameters():
    * for name,param in network.named_parameters():   
* 方法覆盖：
  * 重写__repr__(self)方法可以重新设置对象的字符串表示
  * 如：print(network)

##### 20_Callable Neural Networks - Linear Layers In Depth
* 理解线性层如何工作
  * 调用对象实例进行矩阵乘法
* 理解神经网络如何前向传播
  * nn.Module类重写了__call__()方法，使调用该对象实例时可以直接调用特定的方法(forward())
  * 看源码可理解=>torch/nn/modules/module.py (version 1.0.1)

##### 21_How To Debug PyTorch Source Code - Deep Learning In Python
* 使用VsCode怎么Debug代码(建议百度谷歌实在)

##### 22_CNN Forward Method - PyTorch Deep Learning Implementation
* 神经网络流程
  * 准备数据
  * 建立模型
    * 创建一个扩展nn.Module基类的神经网络类
    * 在类构造函数中，将网络的图层定义为类属性
    * **使用网络的图层属性以及nn.functional API操作来定义网络的前向传递**
  * 训练模型
  * 分析模型的结果
* 调用nn.Module实例的forward()方法时，我们将调用实际的实例，而不是直接调用forward()方法=>因为重写了__call__()方法
* 使用nn.functional API 是为了将权重和操作分开
  * 每一层都有一个权重(数据)
  * nn.functional.relu()等只是单纯的操作，不会保存权重等数据  

##### 23_CNN Image Prediction With PyTorch - Forward Propagation Explained
* 了解网络输入参数的形状要求
* reshaping单张图片使其可以传递到网络并进行前向传播

##### 24_Neural Network Batch Processing - Pass Image Batch To PyTorch CNN
* 神经网络流程
  * 准备数据
  * 建立模型
    * 理解批处理如何传递到网络
  * 训练模型
  * 分析模型的结果
* 了解网络输入参数的形状要求=>确保输入的一批数据的形状符合要求
* 查看前向传播的结果
  * 为什么是dim=1
    * 此时输出的预测张量的形状为(batch size, number of prediction classes)
    * dim=1是最后一个维度=>始终包含数值，而不是张量
  * 获得预测结果的正确预测的数量 
    * preds.argmax(dim=1).eq(labels).sum().item()

##### 25_CNN Output Size Formula - Bonus Neural Network Debugging Session
* 神经网络流程
  * 准备数据
  * 建立模型
    * 了解前向传播的具体转换过程
  * 训练模型
  * 分析模型的结果
* 分析数据流经每一层的形状变化
* 分析数据流经每一个操作的数值变化
* 卷积层输出大小公式
 * $$O_h = \frac{n_h-f_h+2p}{s}+1$$

##### 26_CNN Training With Code Example - Neural Network Programming Course
* 神经网络流程
  * 准备数据
  * 建立模型
  * 训练模型
    * 计算损失和梯度，并更新权重 
  * 分析模型的结果
* 一个epoch的概念
  * 完成了一个完整数据集(所有批次)的前向传播和反向传播并且更新了参数的过程 
* 设置允许进行梯度跟踪
  * torch.set_grad_enabled(True)  
* 理解如何计算损失
  * 使用nn.functional API的cross_entropy()函数  
* 理解如何计算梯度
  * Pytorch会随着数据流过网络，将所有计算添加到计算图中
  * 通过计算图来计算权重的梯度
  * 计算神经网络权重的梯度
    * loss.backward()
  * 查看权重的梯度 
    * network.conv1.weight.grad.shape
* 理解如何更新权重
  * torch.optim中优化器采用不同的算法来使用梯度对权重进行更新
  * 采用Adam优化算法：optimizer =optim.Adam(network.parameters(),lr = 0.01)
  * 进行权重更新：optimizer.step()     

##### 27_CNN Training Loop Explained - Neural Network Code Project
* 神经网络流程
  * 准备数据
  * 建立模型
  * 训练模型
    * **建立训练循环**
  * 分析模型的结果 
* 在进行反向传播之前，需要将梯度归0
  * optimizer.zero_grad()
  * 因为在loss.backward()之后，将会计算梯度并且添加到每一层的权重的grad中
  * optimizer.step()更新梯度后，就需要将梯度重新置0

##### 28_CNN Confusion Matrix With PyTorch - Neural Network Programming
* 神经网络流程
  * 准备数据
  * 建立模型
  * 训练模型
  * 分析模型的结果 
    * **建立、绘制、解释混淆矩阵**
* 获取所有样本的预测值
  * 此时模型应该均训练完成，因此可以无需梯度计算功能，所以可将其关闭
* 预测值与原标签进行叠加获取有序对=>计算混淆矩阵
* 绘制混淆矩阵  

##### 29_Stack Vs Concat In PyTorch, TensorFlow & NumPy - Deep Learning Tensor Ops
* 理解concatenating和stacking的区别并且在Pytorch,TensorFlow,Numpy分别有什么异同
  * concatenating：沿着存在的轴进行连接
  * stacking：创建新的轴进行连接 
  * concatenating和stacking的联系
* Pytorch
  * torch.stack((x,x,x),dim=0)<==>torch.cat((x.unsqueeze(0),x.unsqueeze(0),x.unsqueeze(0)),dim=0)
* TensorFlow：axis<==pytorch中的dim
  * tf.stack((x,x,x),axis=0)<==>tf.concat((tf.expand_dims(x,0),tf.expand_dims(x,0),tf.expand_dims(x,0)),axis=0)
* Numpy
  * np.stack((x,x,x),axis=0)<==>np.concatenate((np.expand_dims(x,0),np.expand_dims(x,0),np.expand_dims(x,0),axis=0)
* stack、concat的使用
  * 将单张图像合并成一个批处理
  * 将多个批次图像合成一个批处理
  * 单张图像和多个批次图像合成一个批处理

##### 30_TensorBoard With PyTorch - Visualize Deep Learning Metrics
* 神经网络流程
  * 准备数据
  * 建立模型
  * 训练模型
  * 分析模型的结果 
    * **使用TensorBoard**
* 理解使用TensorBoard进行可视化各种指标
  * 如何使用？(看代码) 

##### 31_Hyperparameter Tuning And Experimenting - Training Deep Neural Networks
* 神经网络流程
  * 准备数据
  * 建立模型
  * 训练模型
  * 分析模型的结果 
  * 超参数调试
* 使用for循环进行超参数遍历调试 
* 使用TensorBoard分别进行可视化

##### 32_Training Loop Run Builder - Neural Network Experimentation Code
* 使用面向对象实现构建RunBuilder类=>优化for循环调超参
* RunBuilder类：
  * 使用namedtuple和product来包装超参数 

##### 33_CNN Training Loop Refactoring - Simultaneous Hyperparameter Testing
* 使用面向对象思想抽取出RunManager类=>优化TensorBoard的记录
* RunManager类
  * 使训练循环更加简洁，使用专门的类来保存记录训练信息 

##### 34_PyTorch DataLoader Num_workers - Deep Learning Speed Limit Increase
* 使用多线程的功能进行加速训练过程
  * 使用DataLoader类的num_workers属性 
  * 使用num_workers和batch_size来进行调参=>选择最佳结果

##### 35_PyTorch On The GPU - Training Neural Networks With CUDA
* 使用GPU进行加速训练过程
  * 数据转GPU：
    * x.to('cuda')=>推荐使用
    * x.cuda() 
  * 数据转CPU：
    * x.to('cpu')=>推荐使用
    * x.cpu() 
  * 查看数据在什么设备中计算
    * x.device
* 将网络实例移动到GPU
  * network.to('cuda')=>无需新的实例来接收，是就地重新分配的
* 重写RunManager()类实现与设备无关的Pytorch代码
  * torch.cuda.is_available()     

#####　36_PyTorch Dataset Normalization - Torchvision.Transforms.Normalize
*　如何归一化数据集，及带来何种影响？

##### 37_PyTorch DataLoader Source Code - Debugging Session
* 理解在DataLoader在归一化数据时如何发挥作用
* 查看DataLoader源码=>建议自行百度谷歌理解较好

##### 38_PyTorch Sequential Models - Neural Networks Made Easy
* 使用Pytorch 的nn.Sequential构建神经网络的便捷性
* 设置Pytorch的随机数种子，使网络的初始化参数相同
  * torch.manual_seed(50)

##### 39_Batch Norm In PyTorch - Add Normalization To Conv Net Layers
* 理解批量归一化及如何实现=>利用均值、标准差来调整神经网络的中间输出，使其保存稳定
* nn.BatchNorm2d(x)
