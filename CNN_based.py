from torchvision.models import resnext50_32x4d as resnext
#from torchvision.models import vgg16
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
import tqdm
from matplotlib import cm
import matplotlib.pyplot as plt
from torchvision import transforms
#from dataset import HPAdataset
#from model import HPANet
from sklearn.model_selection import train_test_split
import torch
import numpy as np # linear algebra
import os
import tqdm
import pandas as pd

"""
print(os.listdir("../input/cassava-leaf-disease-classification"))
train_df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
#print(train_df)
img_names = train_df['image_id']
label = train_df['label']
print(label)
label = np.array(label)
print(label)
print(img_names[:2])
dicts= {}
for l in label:
    if l in dicts:
        dicts[l] += 1
    else:
        dicts[l] =1
print(dicts)
"""

#import timm
# for GPU only
print('gpu,', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.out = nn.Linear(32 * 7 * 7, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

    def inference(self, inputs):

        '''
        inputs: B x [img_size x img_size x num_channel]
        labels: B x num_cls
        '''

        img = inputs
        x = self.conv1(img)
        x = self.conv2(x)
        # x = self.dropout(self.relu(self.linear2(x)))
        x = x.view(x.size(0), -1)
        output = self.out(x)
        x = torch.argmax(output, dim=1)
        return x


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2
import albumentations as A



class Matadataset(Dataset):
    def __init__(self, path , target_path, batch_size, mode_train=False,  num_cls=2):
        self.mode_train = mode_train
        '''
        if self.mode_train:
            image_base = os.path.join(path,'train_images/')
            csv_path = os.path.join(path,'train.csv')
        else:
            image_base = os.path.join(path,'test_images/')
            csv_path = os.path.join(path,'test.csv')
        '''
        image_base = path
        batch = batch_size
        target_list = os.listdir(target_path)
        target = []
        for i in target_list:
            target.append(target_path+i)
        Rs = ()
        for i in range(len(target)):
            R1 = np.load(target[i])
            Rs = Rs + (R1,)
        R = np.concatenate(Rs)
        #R = np.load("datasets/compressed/720-750trans7sam2500CV10/freq-720-750trans7sam2500CV10.npy")
        info = pd.DataFrame(R)
        info.columns = ["A", "B", "C"]


        dataset_names = info.A
        feature_names = info.B

        labels = []
        # TODO ======= leaving for 18/Mar/2021 =======
        for i in range(len(info.C)):
            # print(info['label'][i])
            img_name = os.path.join(image_base, str(dataset_names[i]) + '-' + str(feature_names[i]) +'-dpi6' +'.png')
            if not os.path.exists(img_name):
                continue
            labels.append(info.C[i])

        num_total = len(labels)
        imgs = list()

        # TODO ======= leaving for 18/Mar/2021 =======
        for i in range(len(dataset_names)):
            img_name = os.path.join(image_base, str(dataset_names[i]) + '-' + str(feature_names[i]) + '-dpi6'+'.png')
            if not os.path.exists(img_name):
                continue
            imgs.append(img_name)

        if int(0.9 * len(imgs)) % batch != 0:
            a = int(0.9 * len(imgs)) - (int(0.9 * len(imgs)) % batch)

        if self.mode_train:
            self.imgs = imgs[:a]  # [:int(0.98*len(imgs))]
            self.labels = labels[:a]  # [:int(0.*len(labels))]
        else:
            self.imgs = imgs[a:]
            self.labels = labels[a:]

        self.preprocess = A.Compose([
            #CenterCrop(1920, 1920, p=1.),
            CenterCrop(28, 28, p=1.),
            #Resize(1920, 1920),
            Resize(28, 28),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
        print(len(self.labels), len(self.imgs))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # print(idx,len(self.labels),self.labels)
        img_name = self.imgs[idx]
        img = Image.open(img_name)
        img = img.convert("RGB")
        img = np.array(img)
        if self.mode_train:
            img = self.preprocess(image=img)['image'].float()
        else:
            img = self.preprocess(image=img)['image'].float()
        label = torch.tensor(self.labels[idx])
        return img, label


class Matadataset_val(Dataset):
    def __init__(self, path, num_cls=2):
        import glob
        image_base = path
        self.imgs = glob.glob(image_base + '*.png')
        self.preprocess = A.Compose([
            CenterCrop(256, 256, p=1.),
            Resize(256, 256),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
        print(len(self.imgs))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # print(idx,len(self.labels),self.labels)
        img_name = self.imgs[idx]
        img = Image.open(img_name)
        img = img.convert("RGB")
        img = np.array(img)

        img = self.preprocess(image=img)['image'].float()
        return img_name.split('/')[-1], img


def validate_test(model, dataset_test):
    predictions = []
    ids = []

    for i, (img_name, img) in enumerate(dataset_test):
        img = img.unsqueeze(0).to(device)
        pred = model.inference(img).cpu().detach().item()
        predictions.append(pred)
        ids.append(img_name)
    sub = pd.DataFrame({'image_id': ids, 'label': predictions})
    sub.to_csv('./submission.csv', index=False)


def validate(model, dataset_test):
    model.eval()

    num_corr = 0
    num_total = len(dataset_test)
    for i, (img, label) in enumerate(dataset_test):
        img = img.unsqueeze(0).to(device)
        pred = model.inference(img).cpu().detach().item()

        label = label.cpu().detach().item()

        if pred == label:
            num_corr += 1

    print("accuracy is", num_corr * 1.0 / num_total, num_corr, '/', num_total)


def inference(model, test_loader, device):
    model.to(device)
    # tk0 = tqdm(enumerate(test_loader), total=len(test_loader))

    predictions = []
    ids = []
    for i, (img_name, img) in enumerate(test_loader):
        img = img.to(device)

        # for state in states:
        # model.load_state_dict(states)
        # model.eval()
        with torch.no_grad():
            pred = model.inference(img).cpu().detach().item()
            predictions.append(pred)
            ids.append(img_name)

    sub = pd.DataFrame({'image_id': ids, 'label': predictions})
    sub.to_csv('./submission.csv', index=False)
    sub.head()


def train(seed, lr_set, batch_size_set, num_epochs_set, path_set, target_path_set, save_name_best_set, save_name_end_set):
    sed = seed
    torch.manual_seed(sed)
    #seed, lr, batch_size, num_epochs, path

    lr = lr_set
    batch_size = batch_size_set
    num_epochs = num_epochs_set
    path = path_set
    save_name_best = save_name_best_set
    #save_name_end = 'model_state/ckpt_routine_{}.pt'.format(epoch)
    target = target_path_set
    dataset_train = Matadataset(path, target, mode_train=True, batch_size=batch_size)
    dataset_test = Matadataset(path, target, mode_train=False, batch_size=batch_size)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    model = CNN().float().cuda()

    # if opt.resume:
    # model, optimizer = load_checkpoint(os.path.join(checkpoint_dir,'model_best'),model,optimizer)

    num_total_instance = len(dataset_train)
    #num_batch = np.ceil(num_total_instance / batch_size)
    num_batch = np.floor(num_total_instance / batch_size)
    #num_batch = num_batch - (num_batch % batch_size)

    optimizer = optim.Adam([
        {
            "params": model.parameters(), "lr": lr,
        },
        # {
        # "params": model.model_base.parameters(), "lr":lr*0.1,
        # }
    ]
    )
    model.to(device)
    min_loss = float('inf')

    for epoch in range(num_epochs):
        training_loss = 0.0
        validate(model, dataset_test)
        model.train()

        for index, (imgs, labels) in enumerate(train_loader):
            labels = labels.squeeze()
            imgs = imgs.to(device)
            labels = labels.to(device)
            #inputs = imgs, labels
            output = model(imgs)
            loss = model.loss(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            if index % 10 == 0:
                print("Epoch:[%d|%d], Batch:[%d|%d] loss: %f" % (
                epoch, num_epochs, index, num_batch, loss.item() / batch_size))

        # validate(model,dataset_test)
        training_loss_epoch = training_loss / (len(train_loader) * batch_size)

    if training_loss_epoch < min_loss:
        min_loss = training_loss_epoch
        print('New best performance! saving...')
        torch.save(model.state_dict(), save_name_best)

    save_name_end = save_name_end_set
    torch.save(model.state_dict(), save_name_end)
'''
seed = 67
lr_set = 5e-5
batch_size_set = 16
num_epochs_set = 500
path_set = "figure1/"
save_name_best_set = 'model_state/'+'batch_size'+str(batch_size_set) +'ckpt_best.pt'
save_name_end_set = 'model_state/ckpt_routine_'+str(num_epochs_set)+'.pt'

train(seed, lr_set, batch_size_set, num_epochs_set, path_set, save_name_best_set, save_name_end_set)
'''
"""
#for Resnext
model = CNN().float()
#model = enet_v2(enet_type[i], out_dim=5)
model.load_state_dict(torch.load('../input/first-2/ckpt_best.pt'), strict=True)
#state_dict = torch.load('../input/first-2/ckpt_best.pt')
#states = state_dict
#test_dataset = Leafdataset_val("../input/cassava-leaf-disease-classification/")
test_loader = DataLoader(dataset_val, batch_size=50, shuffle=False)
inference(model, test_loader, device)
"""