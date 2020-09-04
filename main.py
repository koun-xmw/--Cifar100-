#  首先当然肯定要导入torch和torchvision，至于第三个是用于进行数据预处理的模块
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable  # 这一步还没有显式用到variable，但是现在写在这里也没问题，后面会用到
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # 导入torch.potim模块

#加载数据集

#  **由于torchvision的datasets的输出是[0,1]的PILImage，所以我们先先归一化为[-1,1]的Tensor**
#  首先定义了一个变换transform，利用的是上面提到的transforms模块中的Compose( )
#  把多个变换组合在一起，可以看到这里面组合了ToTensor和Normalize这两个变换
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 定义了我们的训练集，名字就叫trainset，至于后面这一堆，其实就是一个类：
# torchvision.datasets.CIFAR10( )也是封装好了的，就在我前面提到的torchvision.datasets
# 模块中,不必深究，如果想深究就看我这段代码后面贴的图1，其实就是在下载数据
# （不翻墙可能会慢一点吧）然后进行变换，可以看到transform就是我们上面定义的transform
#本地路径 "D:\Python\cifar-10-batches-py\data_batch_1"
trainset = torchvision.datasets.CIFAR100(root='D:\Python\\', train=True,
                                        download=False, transform=transform)
# trainloader其实是一个比较重要的东西，我们后面就是通过trainloader把数据传入网
# 络，当然这里的trainloader其实是个变量名，可以随便取，重点是他是由后面的
# torch.utils.data.DataLoader()定义的，这个东西来源于torch.utils.data模块，
#  网页链接http://pytorch.org/docs/0.3.0/data.html，这个类可见我后面图2
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=4)
# 对于测试集的操作和训练集一样，我就不赘述了
testset = torchvision.datasets.CIFAR100(root='D:\Python\\', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=4)
# 类别信息也是需要我们给定的
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("Dataset Ready !")

#定义网络

# 首先是调用Variable、 torch.nn、torch.nn.functional

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 100)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout10(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 和python中一样，类定义完之后实例化就很简单了，我们这里就实例化了一个net
net = Net()
net.load_state_dict(torch.load(r'D:\Python\python_files\机器学习\cnn_cifar100_6.pkl'))
# print(net)
# print("Finish Net ! ")


#定义优化函数

criterion = nn.CrossEntropyLoss()  # 同样是用到了神经网络工具箱 nn 中的交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)


# 开始训练
#要在main函数下执行，因为要用到多线程 num_workers=4
if __name__ == '__main__':

    n_epoch = 2 #训练次数
    for epoch in range(n_epoch):  # loop over the dataset multiple times 指定训练一共要循环几个epoch

        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for i, data in enumerate(trainloader, 0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            # enumerate是python的内置函数，既获得索引也获得数据，详见下文
            # get the inputs
            inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)  # 将数据转换成Variable，第二步里面我们已经引入这个模块
            # 所以这段程序里面就直接使用了，下文会分析
            # zero the parameter gradients
            optimizer.zero_grad()  # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = net(inputs)  # 把数据输进网络net，这个net()在第二步的代码最后一行我们已经定义了
            loss = criterion(outputs, labels)  # 计算损失值,criterion我们在第三步里面定义了
            loss.backward()  # loss进行反向传播，下文详解
            optimizer.step()  # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮

            # print statistics                   # 这几行代码不是必须的，为了打印出loss方便我们看而已，不影响训练过程
            running_loss += loss.item()  # 从下面一行代码可以看出它是每循环0-1999共两千次才打印一次
            if (i+1) % 50 == 0:  # print every 2000 mini-batches   所以每个2000次之类先用running_loss进行累加
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))  # 然后再除以2000，就得到这两千次的平均损失值
                running_loss = 0.0  # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用
    # torch.save(net.state_dict(),"net_params.pkl")  #保存
    print('Finished Training')
    torch.save(net.state_dict(), "cnn_cifar100_6.pkl")  # 再次保存
    #预测
    correct = 0  # 定义预测正确的图片数，初始化为0
    total = 0  # 总共参与测试的图片数，也初始化为0
    for data in testloader:  # 循环每一个batch
        images, labels = data
        outputs = net(Variable(images))  # 输入网络进行测试
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)  # 更新测试图片的数量
        correct += (predicted == labels).sum()  # 更新正确分类的图片的数量,为tensor类型
    print('Accuracy of the network : ',correct.numpy()/total) # 最后打印结果,要把correct转为数值类型
