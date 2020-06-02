#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import sys

IN_IPYTHON = True
try:
    __IPYTHON__
except NameError:
    IN_IPYTHON = False

if IN_IPYTHON:
    workspace_dir, output_fpath = 'food-11', 'predict.csv'
else:
    try:
        workspace_dir = sys.argv[1]
    except:
        workspace_dir = 'food-11'

    try:
        output_fpath = sys.argv[2]
    except:
        output_fpath = "predict.csv"


# In[3]:


import torch.nn as nn
import torch.nn.functional as F
import torch

class StudentNet(nn.Module):
    '''
      在這個Net裡面，我們會使用Depthwise & Pointwise Convolution Layer來疊model。
      你會發現，將原本的Convolution Layer換成Dw & Pw後，Accuracy通常不會降很多。

      另外，取名為StudentNet是因為這個Model等會要做Knowledge Distillation。
    '''

    def __init__(self, base=16, width_mult=1):
        '''
          Args:
            base: 這個model一開始的ch數量，每過一層都會*2，直到base*16為止。
            width_mult: 為了之後的Network Pruning使用，在base*8 chs的Layer上會 * width_mult代表剪枝後的ch數量。        
        '''
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        # bandwidth: 每一層Layer所使用的ch數量
        bandwidth = [ base * m for m in multiplier]

        # 我們只Pruning第三層以後的Layer
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # 第一層我們通常不會拆解Convolution Layer。
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            # 接下來每一個Sequential Block都一樣，所以我們只講一個Block
            nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 是限制Neuron最小只會到0，最大只會到6。 MobileNet系列都是使用ReLU6。
                # 使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # 過完Pointwise Convolution不需要再做ReLU，經驗上Pointwise + ReLU效果都會變差。
                nn.MaxPool2d(2, 2, 0),
                # 每過完一個Block就Down Sampling
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            # 到這邊為止因為圖片已經被Down Sample很多次了，所以就不做MaxPool
            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[5], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[6], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),

            # 這邊我們採用Global Average Pooling。
            # 如果輸入圖片大小不一樣的話，就會因為Global Average Pooling壓成一樣的形狀，這樣子接下來做FC就不會對不起來。
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            # 這邊我們直接Project到11維輸出答案。
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


# In[4]:


import os
import cv2
import numpy as np

IMAGE_SIZE = 192
def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        # resize to IMAGE_SIZE x ? or ? x IMAGE_SIZE
        height = img.shape[0]
        width = img.shape[1]
        rate = IMAGE_SIZE / max(height, width)
        height = int(height * rate)
        width = int(width * rate)
        img = cv2.resize(img, (width, height))
        # pad black
        # from https://blog.csdn.net/qq_20622615/article/details/80929746
        W, H = IMAGE_SIZE, IMAGE_SIZE
        top = (H - height) // 2
        bottom = (H - height) // 2
        if top + bottom + height < H:
            bottom += 1
        left = (W - width) // 2
        right = (W - width) // 2
        if left + right + width < W:
            right += 1
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # to np array
        x[i, :, :] = img
        if label:
            y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x


# In[5]:


import torchvision.transforms as transforms

transform_mean = np.array([ 69.58238342,  92.66689336, 115.24940137]) / 255
transform_std = np.array([71.8342021 , 76.83536755, 83.40123168]) / 255

train_transform1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomChoice([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective()
    ]),
    transforms.RandomChoice([
        transforms.RandomAffine(10), # 隨機線性轉換
        transforms.RandomRotation(40)
    ]),
    transforms.ColorJitter(), # 隨機色溫等
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    transforms.Normalize(
        transform_mean,
        transform_std
    )
])
train_transform2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomOrder([
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective()
        ]),
        transforms.RandomAffine(30), # 隨機線性轉換
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.5, 1.0)), # 隨機子圖
    ]),
    transforms.RandomChoice([
        transforms.ColorJitter(), # 隨機色溫等
        transforms.RandomGrayscale(),
    ]),
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    transforms.RandomErasing(0.2),
    transforms.Normalize(
        transform_mean,
        transform_std
    )
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
    transforms.Normalize(
        transform_mean,
        transform_std
    )
])


# In[6]:


from torch.utils.data import DataLoader, Dataset, ConcatDataset

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


# In[7]:


if False:
    print("Reading data")
    train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
    print("Size of validation data = {}".format(len(val_x)))

    batch_size = 32
    train_set = ConcatDataset([
        ImgDataset(train_x, train_y, train_transform1),
        ImgDataset(train_x, train_y, train_transform2),
        ImgDataset(train_x, train_y, test_transform),
    #     ImgDataset(val_x, val_y, train_transform1),
    #     ImgDataset(val_x, val_y, train_transform2),
    #     ImgDataset(val_x, val_y, test_transform)
    ])
    val_set = ImgDataset(val_x, val_y, test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=(16 if os.name=='posix' else 0))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=(16 if os.name=='posix' else 0))


# In[8]:


CALCULATE_STD_MEAN = False
if CALCULATE_STD_MEAN:
    tmp = ConcatDataset([train_set, val_set])
    tot, tot2 = np.zeros(3), np.zeros(3)
    tot_n = len(tmp) * IMAGE_SIZE ** 2
    for x, y in tmp:
        x = np.array(x, dtype=np.float64)
        tot += x.sum(axis=(0,1))
        tot2 += (x*x).sum(axis=(0,1))
    tot /= tot_n
    tot2 /= tot_n
    tot, np.sqrt(tot2 - tot*tot)


# In[9]:


class TeacherNet_oToToT(nn.Module):
    def __init__(self):
        super(TeacherNet_oToToT, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, IMAGE_SIZE, IMAGE_SIZE]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Dropout(0.4),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Dropout(0.4),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Dropout(0.4),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc = nn.Sequential(
            nn.Linear(12*12*512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Dropout(0.4),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
                        
            nn.Linear(1024, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


# In[ ]:


teacher_net = TeacherNet_oToToT().cuda()
# teacher_net.load_state_dict(torch.load('teacher_model.bin'))
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizers = [
    (torch.optim.Adam, 0.002),
    (torch.optim.SGD, 0.001)
]
num_epochs = [
    80,
    250
]


# In[ ]:


import time

TRAIN_TEACHER_NET = False

if TRAIN_TEACHER_NET:
    best_acc = 0

    for (optimizer, lr), num_epoch in zip(optimizers, num_epochs):
        optimizer = optimizer(teacher_net.parameters(), lr)
        for epoch in range(num_epoch):
            epoch_start_time = time.time()
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0

            teacher_net.train() # 確保 model 是在 train model (開啟 Dropout 等...)
            for i, data in enumerate(train_loader):
                optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
                train_pred = teacher_net(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
                batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
                batch_loss.backward() 
                optimizer.step() # 以 optimizer 用 gradient 更新參數值

                train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                train_loss += batch_loss.item()

#             print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % 
#                 (epoch + 1, num_epoch, time.time()-epoch_start_time, train_acc/len(train_set), train_loss/len(train_set)))
                
            teacher_net.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    val_pred = teacher_net(data[0].cuda())
                    batch_loss = loss(val_pred, data[1].cuda())
                    val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                    val_loss += batch_loss.item()

                if val_acc > best_acc:
                    torch.save(teacher_net.state_dict(), 'teacher_model.bin')

                #將結果 print 出來
                print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % 
                      (epoch + 1, num_epoch, time.time()-epoch_start_time, train_acc/len(train_set),
                       train_loss/len(train_set), val_acc/len(val_set), val_loss/len(val_set)))
#     torch.save(teacher_net.state_dict(), 'teacher_model.bin')


# In[ ]:


teacher_net = TeacherNet_oToToT().cuda()
# teacher_net.load_state_dict(torch.load('teacher_model.bin'))


# In[ ]:


CHECK_TEACHER_NET = False
if CHECK_TEACHER_NET:
    test_x = readfile(os.path.join(workspace_dir, "testing"), False)
    print("Size of Testing data = {}".format(len(test_x)))
    test_set = ImgDataset(test_x, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=(16 if os.name=='posix' else 0))

    teacher_net.eval()

    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = teacher_net(data.cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)

    with open(output_fpath, 'w') as f:
        f.write('Id,Category\n')
        for i, y in enumerate(prediction):
            f.write('{},{}\n'.format(i, y))


# In[ ]:


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
 
    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class StudentNet(nn.Module):
    '''
      在這個Net裡面，我們會使用Depthwise & Pointwise Convolution Layer來疊model。
      你會發現，將原本的Convolution Layer換成Dw & Pw後，Accuracy通常不會降很多。

      另外，取名為StudentNet是因為這個Model等會要做Knowledge Distillation。
    '''

    def __init__(self, base=16, width_mult=1):
        '''
          Args:
            base: 這個model一開始的ch數量，每過一層都會*2，直到base*16為止。
            width_mult: 為了之後的Network Pruning使用，在base*8 chs的Layer上會 * width_mult代表剪枝後的ch數量。        
        '''
        super(StudentNet, self).__init__()
        multiplier = [2, 4, 8, 8, 8, 16, 16, 16, 16]

        # bandwidth: 每一層Layer所使用的ch數量
        bandwidth = [int(base * m) for m in multiplier]

        # 我們只Pruning第三層以後的Layer
        for i in range(4, 8):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # 第一層我們通常不會拆解Convolution Layer。
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            # 接下來每一個Sequential Block都一樣，所以我們只講一個Block
            nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 是限制Neuron最小只會到0，最大只會到6。 MobileNet系列都是使用ReLU6。
                # 使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # 過完Pointwise Convolution不需要再做ReLU，經驗上Pointwise + ReLU效果都會變差。
                nn.MaxPool2d(2, 2, 0),
                # 每過完一個Block就Down Sampling
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),
            
            # 到這邊為止因為圖片已經被Down Sample很多次了，所以就不做MaxPool

            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                swish(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                swish(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                swish(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),
            
            nn.Sequential(
                nn.Conv2d(bandwidth[7], bandwidth[7], 3, 1, 1, groups=bandwidth[7]),
                nn.BatchNorm2d(bandwidth[7]),
                swish(),
                nn.Conv2d(bandwidth[7], bandwidth[8], 1),
            ),

            # 這邊我們採用Global Average Pooling。
            # 如果輸入圖片大小不一樣的話，就會因為Global Average Pooling壓成一樣的形狀，這樣子接下來做FC就不會對不起來。
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            # 這邊我們直接Project到11維輸出答案。
            nn.Linear(bandwidth[8], 128),
            nn.BatchNorm1d(128),
            nn.ReLU6(),
            
            nn.Dropout(0.4),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            swish(),
                        
            nn.Linear(128, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


# In[ ]:


def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss


# In[ ]:


#from torchvision.models import mobilenet_v2
# student_net = mobilenet_v2(
#     num_classes=11,
#     width_mult=0.6,
#     round_nearest=4,
#     inverted_residual_setting = [
#         # t, c, n, s
#         [1, 16, 1, 1],
#         [6, 24, 2, 2],
# #         [6, 32, 3, 2],
#         [6, 64, 4, 2],
#         [6, 96, 3, 1],
# #         [6, 160, 3, 2],
#         [6, 320, 1, 1],
#     ]
# ).cuda()

student_net_base = 9.5
student_net = StudentNet(student_net_base).cuda()
# student_net.load_state_dict(torch.load('student_model.bin'))

optimizer = torch.optim.Adam(student_net.parameters(), lr=1e-3)


# In[ ]:


def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 處理 input
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
        # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()    
        else:
            # 只是算validation acc的話，就開no_grad節省空間。
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num


# In[ ]:


num_epoch = 0

# TeacherNet永遠都是Eval mode.
teacher_net.eval()
now_best_acc = 0
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    
    student_net.train()
    train_loss, train_acc = run_epoch(train_loader, update=True)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(val_loader, update=False)

    # 存下最好的model。
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), 'student_model.bin')
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % 
            (epoch + 1, num_epoch, time.time()-epoch_start_time, train_acc,
            train_loss, valid_acc, valid_loss))


# In[ ]:


num_epoch = 0

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    
    student_net.train()
    train_loss, train_acc = run_epoch(train_loader, update=True, alpha=0)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(val_loader, update=False, alpha=0)

    # 存下最好的model。
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), 'student_model.bin')
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % 
            (epoch + 1, num_epoch, time.time()-epoch_start_time, train_acc,
            train_loss, valid_acc, valid_loss))


# In[ ]:


def network_slimming(old_model, new_model):
    params = old_model.state_dict()
    new_params = new_model.state_dict()
    
    # selected_idx: 每一層所選擇的neuron index
    selected_idx = []
    # 我們總共有7層CNN，因此逐一抓取選擇的neuron index們。
    for i in range(8):
        # 根據上表，我們要抓的gamma係數在cnn.{i}.1.weight內。
        importance = params[f'cnn.{i}.1.weight']
        # 抓取總共要篩選幾個neuron。
        old_dim = len(importance)
        new_dim = len(new_params[f'cnn.{i}.1.weight'])
        # 以Ranking做Index排序，較大的會在前面(descending=True)。
        ranking = torch.argsort(importance, descending=True)
        # 把篩選結果放入selected_idx中。
        selected_idx.append(ranking[:new_dim])

    now_processed = 1
    for (name, p1), (name2, p2) in zip(params.items(), new_params.items()):
        # 如果是cnn層，則移植參數。
        # 如果是FC層，或是該參數只有一個數字(例如batchnorm的tracenum等等資訊)，那麼就直接複製。
        if name.startswith('cnn') and p1.size() != torch.Size([]) and now_processed != len(selected_idx):
            # 當處理到Pointwise的weight時，讓now_processed+1，表示該層的移植已經完成。
            if name.startswith(f'cnn.{now_processed}.3'):
                now_processed += 1

            # 如果是pointwise，weight會被上一層的pruning和下一層的pruning所影響，因此需要特判。
            if name.endswith('3.weight'):
                # 如果是最後一層cnn，則輸出的neuron不需要prune掉。
                if len(selected_idx) == now_processed:
                    new_params[name] = p1[:,selected_idx[now_processed-1]]
                # 反之，就依照上層和下層所選擇的index進行移植。
                # 這裡需要注意的是Conv2d(x,y,1)的weight shape是(y,x,1,1)，順序是反的。
                else:
                    new_params[name] = p1[selected_idx[now_processed]][:,selected_idx[now_processed-1]]
            else:
                new_params[name] = p1[selected_idx[now_processed]]
        else:
            new_params[name] = p1

    # 讓新model load進被我們篩選過的parameters，並回傳new_model。        
    new_model.load_state_dict(new_params)
    return new_model


# In[ ]:


student_net = StudentNet(student_net_base).cuda()
# student_net.load_state_dict(torch.load('student_model.bin'))

now_width_mult = 1
for i in range(0):
    now_width_mult *= 0.95
    print(now_width_mult)
    new_net = StudentNet(student_net_base, width_mult=now_width_mult).cuda()
    student_net = network_slimming(student_net, new_net)
    now_best_acc = 0
    for epoch in range(200):
        epoch_start_time = time.time()

        student_net.train()
        train_loss, train_acc = run_epoch(train_loader, update=True)
        student_net.eval()
        valid_loss, valid_acc = run_epoch(val_loader, update=False)

        # 存下最好的model。
        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(student_net.state_dict(), f'student_model-pruned_{i}.bin')
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % 
                (epoch + 1, 100, time.time()-epoch_start_time, train_acc,
                train_loss, valid_acc, valid_loss))
    for epoch in range(0):
        epoch_start_time = time.time()

        student_net.train()
        train_loss, train_acc = run_epoch(train_loader, update=True, alpha=0)
        student_net.eval()
        valid_loss, valid_acc = run_epoch(val_loader, update=False, alpha=0)

        # 存下最好的model。
        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(student_net.state_dict(), f'student_model-pruned_{i}.bin')
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % 
                (epoch + 1, 0, time.time()-epoch_start_time, train_acc,
                train_loss, valid_acc, valid_loss))


# In[ ]:


import pickle

def encode16(params, fname):
    '''將params壓縮成16-bit後輸出到fname。

    Args:
      params: model的state_dict。
      fname: 壓縮後輸出的檔名。
    '''

    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        # 有些東西不屬於ndarray，只是一個數字，這個時候我們就不用壓縮。
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode16(fname):
    '''從fname讀取各個params，將其從16-bit還原回torch.tensor後存進state_dict內。

    Args:
      fname: 壓縮後的檔名。
    '''

    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param

    return custom_dict


# In[ ]:

if False:
    params = torch.load('student_model.bin')
    encode16(params, 'student_model_16bit.bin')
    print(f"16-bit cost: {os.stat('student_model_16bit.bin').st_size} bytes.")


# In[ ]:


student_net = StudentNet(student_net_base).cuda()
student_net.load_state_dict(decode16('model.bin'))
num_epoch = 0

# TeacherNet永遠都是Eval mode.
teacher_net.eval()
now_best_acc = 0
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    
    student_net.train()
    train_loss, train_acc = run_epoch(train_loader, update=True)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(val_loader, update=False)

    # 存下最好的model。
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), 'student_model_16bit.bin')
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % 
            (epoch + 1, num_epoch, time.time()-epoch_start_time, train_acc,
            train_loss, valid_acc, valid_loss))


# In[ ]:


test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=(16 if os.name=='posix' else 0))

student_net.eval()

prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = student_net(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

with open(output_fpath, 'w') as f:
    f.write('Id,label\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

