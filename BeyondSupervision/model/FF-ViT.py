'''
This is pytorch impplementation of improved FF-ViT from
Beyond supervision: Harnessing self-supervised learning in unseen plant disease recognition
https://www.sciencedirect.com/science/article/pii/S0925231224013791

The original implementation of FF-ViT is from 
Pairwise Feature Learning for Unseen Plant Disease Recognition
https://ieeexplore.ieee.org/abstract/document/10222401/
'''


import pandas as pd
import numpy as np
import os
import torch
import timm
from tqdm import tqdm
import cv2
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from albumentations import (
    HorizontalFlip, HueSaturationValue, VerticalFlip, RandomResizedCrop, RandomBrightnessContrast, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, Transpose)
from albumentations.pytorch import ToTensorV2
from timm.models.vision_transformer import Block
from functools import partial

    
class CustomDatasetForFFVIT(Dataset):
    def __init__(self, csv_plant ,csv_disease, transforms):

        self.get_plant = 0
        self.get_disease = 0
        self.get_P_image = np.zeros((num_classes,1),dtype=int)
        self.get_D_image = np.zeros((num_disease,1),dtype=int)
        
        # Read the csv file for plant path list
        self.plant_path_list = pd.read_csv(csv_plant, header=None)
        self.disease_path_list = pd.read_csv(csv_disease, header=None)
        
        # assign transformation
        self.transforms = transforms
        
        self.total = 0
        for x in self.plant_path_list.index:
            self.data_plant = pd.read_csv(self.plant_path_list[0][x], header=None)
            self.total = self.total + len(self.data_plant)
            
        self.data_path = data_path        
    def __getitem__(self, index):
        # Read the csv for the class
        self.data_info = pd.read_csv(self.plant_path_list[0][self.get_plant], header=None)        
        # First column contains the image name
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        
        # Get image name for plant
        single_image_name1 = self.image_arr[self.get_P_image[self.get_plant][0]]
        # Obtain image path for plant
        img_path1 = os.path.join(self.data_path, single_image_name1)
        # Obtain image label for plant
        single_image_label1 = self.label_arr[self.get_P_image[self.get_plant][0]]

        # Open image
        image = cv2.imread(img_path1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Transform image to tensor
        transformed = self.transforms(image=image)
        img_as_tensor1 = transformed["image"]
        
        # set next image and check end of the list
        self.get_P_image[self.get_plant][0] += 1
        if self.get_P_image[self.get_plant][0] > (len(self.data_info.index) - 1):
            self.get_P_image[self.get_plant][0] = 0

        # Read the csv for the plant
        self.data_info = pd.read_csv(self.disease_path_list[0][self.get_disease], header=None)        
        # First column contains the image name
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])

        # Get image name for disease
        single_image_name2 = self.image_arr[self.get_D_image[self.get_disease][0]]
        # Obtain image path for disease
        img_path2 = os.path.join(self.data_path, single_image_name2)
        # Obtain image label for disease
        single_image_label2 = self.label_arr[self.get_D_image[self.get_disease][0]]

        # Open image
        image = cv2.imread(img_path2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Transform image to tensor
        transformed = self.transforms(image=image)
        img_as_tensor2 = transformed["image"]
        
        # set next image and check end of the list
        self.get_D_image[self.get_disease][0] += 1
        if self.get_D_image[self.get_disease][0] > (len(self.data_info.index) - 1):
            self.get_D_image[self.get_disease][0] = 0

        self.get_disease += 1        
        if self.get_disease > (len(self.disease_path_list.index) - 1):
            self.get_disease = 0
            self.get_plant += 1        
            if self.get_plant > (len(self.plant_path_list.index) - 1):
                self.get_plant = 0
        # return (img_path1, single_image_label1, img_path2, single_image_label2)
        return (img_as_tensor1, single_image_label1, img_as_tensor2, single_image_label2)
    
    def __len__(self):
        return self.total
    
class CustomDatasetFromImagesForalbumentation(Dataset):
    def __init__(self, csv_path, transforms):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image name
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels for class
        self.label_arr_cls = np.asarray(self.data_info.iloc[:, 1])
        # Third column is the labels for disease
        self.label_arr_dis = np.asarray(self.data_info.iloc[:, 2])

        self.transforms = transforms
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.data_path = data_path

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Obtain image path
        img_path = os.path.join(self.data_path, single_image_name)
        # Open image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform image to tensor
        transformed = self.transforms(image=image)
        img_as_tensor = transformed["image"]

        # Get label(class and disease) of the image based on the cropped pandas column
        class_image_label = self.label_arr_cls[index]
        disease_image_label = self.label_arr_dis[index]
        
        return (img_as_tensor, class_image_label, disease_image_label)

    def __len__(self):
        return self.data_len

class modelclassifierwithbase(nn.Module):
    def __init__(self,model_cls,model_dis,num_classes,num_disease):
        super(modelclassifierwithbase,self).__init__()
        self.model_cls = model_cls      
        self.model_dis = model_dis
        
        self.dim = 768
        self.num_heads = 12
        self.mlp_ratio = 4.
        self.qkv_bias = True
        self.drop = 0.
        self.attn_drop = 0.
        self.drop_path = 0.
        self.act_layer = nn.GELU
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        # self-attention block for synthetic features
        self.block1 = Block(
                dim= self.dim,
                num_heads= self.num_heads,
                mlp_ratio= self.mlp_ratio,
                qkv_bias= self.qkv_bias,
                attn_drop= self.attn_drop,
                drop_path= self.drop_path,
                norm_layer= self.norm_layer,
                act_layer= self.act_layer
            )
        
        self.block3 = Block(
                dim= self.dim,
                num_heads= self.num_heads,
                mlp_ratio= self.mlp_ratio,
                qkv_bias= self.qkv_bias,
                attn_drop= self.attn_drop,
                drop_path= self.drop_path,
                norm_layer= self.norm_layer,
                act_layer= self.act_layer
            )
        
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.layerNorm1 = nn.LayerNorm(normalized_shape=self.dim,eps=1e-6, elementwise_affine=True, device=device)
        self.layerNorm2 = nn.LayerNorm(normalized_shape=self.dim,eps=1e-6, elementwise_affine=True, device=device)
        
        # classifer for original features
        self.linearc = nn.Linear(self.dim,num_classes,device=device)
        self.lineard = nn.Linear(self.dim,num_disease,device=device)
        
        # classifier for synthetic features
        self.linear_c = nn.Linear(768, num_classes)
        self.linear_d = nn.Linear(768, num_disease)

    def forward(self, x, y):
        
        # Obtain original features from crop and disease model
        self.x1 = self.model_cls.forward_features(x)
        self.x2 = self.model_dis.forward_features(y)

        # original features classifier
        self.c = self.linear_c(self.x1[:, 0])
        self.d = self.linear_d(self.x2[:, 0])
        
        # Feature fusion (summation, multiplication and concatenation)
        self.output_combine = self.x1 + self.x2
        # self.output_combine = torch.mul(self.x1, self.x2)
        # self.output_combine = torch.cat((self.output_combine,self.x2),-1)
        
        # Attention block for synthetic crop features
        self.xc = self.block1(self.output_combine)
        # self.xc = self.block2(self.xc)
        self.xc = self.layerNorm1(self.xc)
        
        # Synthetic crop classifier
        self.xc = self.linearc(self.gelu1(self.xc[:, 0] + self.x2[:, 0]))
        self.xc = self.xc

        # Attention block for synthetic disease features
        self.xd = self.block3(self.output_combine)
        # self.xd = self.block4(self.xd)
        self.xd = self.layerNorm2(self.xd)
        
         # Synthetic disease classifier       
        self.xd = self.lineard(self.gelu1(self.xd[:, 0] + self.x1[:, 0]))
        self.xd = self.xd
        return (self.c, self.d, self.xc, self.xd)

# Data path for datasets and metadatas
data_path = 'C:/Users/User/Desktop/Vision Transformer/plantvillage (mix)'
Save_model_path = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/save model"

train_plant_list = 'C:/Users/User/Desktop/Vision Transformer/csv file/PV 37c 2L/PV train 2 label plant path list.csv'
train_disease_list = 'C:/Users/User/Desktop/Vision Transformer/csv file/PV 37c 2L/PV train 2 label disease path list.csv'
test_csv_path1 = 'C:/Users/User/Desktop/Vision Transformer/csv file/PV 37c 2L/Test/PV test (disease separated) 2L.csv' #6
test_csv_path2 = 'C:/Users/User/Desktop/Vision Transformer/csv file/PV 37c 2L/Test/PV pepper bacteria spot test 2L.csv' #6


# Hyperparameters and variables
img_size = 224
batch_size = 1
num_epochs = 30
num_classes = 14
num_disease = 21
pretrained = True
learning_rate_classifier = 0.001
momentum_classifier = 0.9
weight_decay_classifier = 0.00001

# Default data augmentation
train_transforms = Compose([
            RandomResizedCrop(img_size, img_size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

valid_transforms = Compose([
            Resize(img_size, img_size),
            CenterCrop(img_size, img_size, p=1.),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

# Data path for datasets and metadatas
data_path = 'C:/Users/User/Desktop/Vision Transformer/plantvillage (mix)'
Save_model_path = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/save model"

train_plant_list = 'C:/Users/User/Desktop/Vision Transformer/csv file/PV 37c 2L/PV train 2 label plant path list.csv'
train_disease_list = 'C:/Users/User/Desktop/Vision Transformer/csv file/PV 37c 2L/PV train 2 label disease path list.csv'
test_csv_path1 = 'C:/Users/User/Desktop/Vision Transformer/csv file/PV 37c 2L/Test/PV test (disease separated) 2L.csv' 
test_csv_path2 = 'C:/Users/User/Desktop/Vision Transformer/csv file/PV 37c 2L/Test/PV pepper bacteria spot test 2L.csv' 

# Dataset declarations
train_dataset = CustomDatasetForFFVIT(train_plant_list, train_disease_list, transforms = train_transforms)
test_dataset1 = CustomDatasetFromImagesForalbumentation(test_csv_path1, transforms = valid_transforms)
test_dataset2 = CustomDatasetFromImagesForalbumentation(test_csv_path2, transforms = valid_transforms)

# Dataset loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)
test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)

# ViT model from Timm finetuned weight from own dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
model_name = "vit_base_patch16_224"
model_cls = timm.create_model(model_name, pretrained=False,num_classes=0)

# load finetune weight for crop model
if pretrained:
    print("Using Pre-Trained Model for crop")
    MODEL_PATH = "path to your pretrained weight in pth"
    model_cls.load_state_dict(torch.load(MODEL_PATH),strict=False)
model_dis = timm.create_model(model_name, pretrained=False,num_classes=0)

# load finetune weight for disease model
if pretrained:
    print("Using Pre-Trained Model for disease")
    MODEL_PATH = "path to your pretrained weight in pth"
    model_dis.load_state_dict(torch.load(MODEL_PATH),strict=False)

model_cls.to(device)
model_dis.to(device)

model_classifier = modelclassifierwithbase(model_cls,model_dis,num_classes,num_disease)
model_classifier.to(device)

# Use this if for checkpoint or fine-tune for model
# if pretrained:
#     print("Using Pre-Trained Model for FF-ViT")
#     MODEL_PATH = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/load model/TIP/vit_base_patch16_224_30e_0.0001L_99.2512_14.8148_FFViT_2B1C_sum+att+skip_294b_37c_4Out-classifier_S1-1.pth"
#     model_classifier.load_state_dict(torch.load(MODEL_PATH),strict=True)

# Declare optimizer and losses
error1 = nn.CrossEntropyLoss()
error2 = nn.CrossEntropyLoss()
error3 = nn.CrossEntropyLoss()
error4 = nn.CrossEntropyLoss()
optimizer_classifier = torch.optim.SGD(model_classifier.parameters(), lr=learning_rate_classifier, weight_decay = weight_decay_classifier, momentum = momentum_classifier)
for g in optimizer_classifier.param_groups:
    g['lr'] = learning_rate_classifier
    
# Use this if for checkpoint or fine-tune for optimizer
# if pretrained:
#     print("Using Pre-Trained optimizer")
#     OPTIMIZER_PATH = "C:/Users/User/Desktop/Vision Transformer/Pre-trained model/load model/TIP/vit_base_patch16_224_30e_0.0001L_99.2512_14.8148_FFViT_2B1C_sum+att+skip_294b_37c_4Out-opt_S1-1.pth"
#     optimizer_classifier.load_state_dict(torch.load(OPTIMIZER_PATH))


# Training counter
label_combine_train_list_full = []
predictions_test_cls_list_full = []
predictions_test_dis_list_full = []
predictions_test_combine_list_full = []

for epoch in range(num_epochs):

    print(f"\nStart of Epoch {epoch+1} of {num_epochs}")
    print('Current Learning rate: {0}'.format(optimizer_classifier.param_groups[0]['lr']))
    
    # Training counter for calculation or debug
    total_train = 0
    correct_train = 0
    correct_train_cls = 0
    correct_train_dis = 0
    correct_train_cls_2 = 0
    correct_train_dis_2 = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0
    total_loss4 = 0    
    total_loss_combine = 0
    model_cls.train()
    model_dis.train()
    model_classifier.train()

    # Batch size (using gradient accumulation method)    
    # iter_batch = 294
    iter_batch = 32
    
    for batch_idx, (images1, labels_cls, images2, labels_dis) in enumerate(tqdm(train_loader)):
        images1, labels_cls, images2, labels_dis = images1.to(device), labels_cls.to(device) , images2.to(device), labels_dis.to(device) 

        # To obtain the combined labels from individual concepts
        label_combine_train = (labels_cls+((num_disease-1)*labels_cls))+labels_dis
        label_combine_train_list = label_combine_train.cpu().numpy()
        label_combine_train_list_full = np.append(label_combine_train_list_full,label_combine_train_list)
        
        # Obtain the output for original crop, original disease, synthetic crop and synthetic disease
        output = model_classifier(images1,images2)
        
        # Calculate loss for original crop and disease
        loss1 = error1(output[0], labels_cls)
        total_loss1 += loss1.item()
        loss2 = error2(output[1], labels_dis)
        total_loss2 += loss2.item()
        
        # Calculate loss for synthetic crop and disease
        loss3 = error3(output[2], labels_cls)
        total_loss3 += loss3.item()
        loss4 = error4(output[3], labels_dis)
        total_loss4 += loss4.item()
        
        # Moving weighted sum, a (Previous FF-ViT implementations)
        # a = (epoch+1)/num_epochs
        # loss_combine = torch.mul((1-a), (loss1 + loss2)) + torch.mul(a,loss3 + loss4)

        # Obtain the combined losses
        loss_combine = loss1 + loss2 + loss3 + loss4
        total_loss_combine += loss_combine.item()
    

        # Gradient calculation and model update        
        loss_combine = loss_combine / iter_batch
        loss_combine.backward()
        if ((batch_idx + 1) % iter_batch == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer_classifier.step()
            optimizer_classifier.zero_grad()        
            # print(f"update weight at mini-batch {batch_idx+1}")
        
        # Obtain the predicted label    
        predictions_cls = torch.max(output[0], 1)[1].to(device)
        predictions_list_cls = predictions_cls.cpu().numpy()
        predictions_dis = torch.max(output[1], 1)[1].to(device)
        predictions_list_dis = predictions_dis.cpu().numpy()
        predictions_cls_2 = torch.max(output[2], 1)[1].to(device)
        predictions_list_cls_2 = predictions_cls_2.cpu().numpy()
        predictions_dis_2 = torch.max(output[3], 1)[1].to(device)
        predictions_list_dis_2 = predictions_dis_2.cpu().numpy()
        
        # Calculate accuracy
        correct_train_cls += (predictions_cls == labels_cls).sum()
        correct_train_dis += (predictions_dis == labels_dis).sum()  
        correct_train_cls_2 += (predictions_cls_2 == labels_cls).sum()
        correct_train_dis_2 += (predictions_dis_2 == labels_dis).sum() 
        
        # Post-calculation for plant disease identification prediction
        for x in range(len(labels_cls)):
            if (predictions_cls_2[x] == labels_cls[x]):
                if (predictions_dis_2[x] == labels_dis[x]):
                    correct_train += 1

        total_train += len(labels_cls)
    
    # Total losses calcuations 
    c = total_loss1 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
    d = total_loss2 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
    cd = total_loss3 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
    cd2 = total_loss4 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
    tcd = total_loss_combine / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))

    # Total accuracy calculations
    accuracy_train = correct_train * 100 / total_train
    accuracy_train_cls = correct_train_cls * 100 / total_train
    accuracy_train_dis = correct_train_dis * 100 / total_train
    accuracy_train_cls_2 = correct_train_cls_2 * 100 / total_train
    accuracy_train_dis_2 = correct_train_dis_2 * 100 / total_train
    print('Original crop acc: {:.4f}'.format(accuracy_train_cls))
    print('Original Disease acc: {:.4f}'.format(accuracy_train_dis))
    print('Synthetic crop acc: {:.4f}'.format(accuracy_train_cls_2))
    print('Synthetic Disease acc: {:.4f}'.format(accuracy_train_dis_2))    
    print('Combined plant disease acc: {:.4f}'.format(accuracy_train))
    print('Class base loss: {0}'.format(c))
    print('Disease base loss: {0}'.format(d))
    print('classifier class loss: {0}'.format(cd))
    print('classifier disease loss: {0}'.format(cd2))
    print('Total loss: {0}'.format(c+d+cd+cd2))
    print('backward loss: {0}'.format(tcd))
    # print('Current ratio: {0}'.format(a))
    print(f"\nEpoch {epoch+1} of {num_epochs} Done!")


    # Seen test dataset evaluation   
    print(f"\nSeen")
    total_test = 0
    correct_test_cls = 0
    correct_test_dis = 0
    correct_test_cls_2 = 0
    correct_test_dis_2 = 0
    correct_test_combine = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0
    total_loss4 = 0   
    total_loss_combine = 0
    model_cls.eval()
    model_dis.eval()
    model_classifier.eval()
  
    for images, labels_cls, labels_dis in tqdm(test_loader1):
        images, labels_cls, labels_dis = images.to(device), labels_cls.to(device) , labels_dis.to(device)
        
        # To obtain the combined labels from individual concepts       
        label_combine_test = (labels_cls+((num_disease-1)*labels_cls))+labels_dis
        label_combine_test_list = label_combine_test.cpu().numpy()

        # Obtain the output for original crop, original disease, synthetic crop and synthetic disease        
        output = model_classifier(images,images)
        
        # Calculate loss
        loss1 = error1(output[0], labels_cls)
        total_loss1 += loss1.item()
        loss2 = error2(output[1], labels_dis)
        total_loss2 += loss2.item()        
        loss3 = error3(output[2], labels_cls)
        total_loss3 += loss3.item()
        loss4 = error4(output[3], labels_dis)
        total_loss4 += loss4.item()
        
        # Obtain the combined losses
        loss_combine = loss1 + loss2 + loss3 + loss4
        total_loss_combine += loss_combine.item()

        # Obtain the predicted label
        predictions_test_cls = torch.max(output[0], 1)[1].to(device)
        predictions_test_cls_list = predictions_test_cls.cpu().numpy()
        predictions_test_cls_list_full = np.append(predictions_test_cls_list_full,predictions_test_cls_list)      
        predictions_test_dis = torch.max(output[1], 1)[1].to(device)
        predictions_test_dis_list = predictions_test_dis.cpu().numpy()
        predictions_test_dis_list_full = np.append(predictions_test_dis_list_full,predictions_test_dis_list)
        predictions_test_cls_2 = torch.max(output[2], 1)[1].to(device)
        predictions_test_dis_2 = torch.max(output[3], 1)[1].to(device)

        # Calculate accuracy        
        correct_test_cls += (predictions_test_cls == labels_cls).sum()
        correct_test_dis += (predictions_test_dis == labels_dis).sum() 
        correct_test_cls_2 += (predictions_test_cls_2 == labels_cls).sum()
        correct_test_dis_2 += (predictions_test_dis_2 == labels_dis).sum()
        
        # Post-calculation for plant disease identification prediction        
        for x in range(len(labels_cls)):
            if (predictions_test_cls_2[x] == labels_cls[x]):
                if (predictions_test_dis_2[x] == labels_dis[x]):
                    correct_test_combine += 1        
        total_test += len(labels_cls)

    # Total accuracy and losses calculations        
    c = total_loss1 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
    d = total_loss2 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
    cd = total_loss3 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
    cd2 = total_loss4 / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))
    tcd = total_loss_combine / ((len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0))       
    accuracy_test_cls  = correct_test_cls * 100 / total_test
    accuracy_test_dis  = correct_test_dis * 100 / total_test
    accuracy_test_cls_2  = correct_test_cls_2 * 100 / total_test
    accuracy_test_dis_2  = correct_test_dis_2 * 100 / total_test    
    accuracy_test_combine  = correct_test_combine * 100 / total_test
    print("")
    print('Original crop acc: {:.4f}'.format(accuracy_test_cls))
    print('Original Disease acc: {:.4f}'.format(accuracy_test_dis))
    print('Synthetic crop acc: {:.4f}'.format(accuracy_test_cls_2))
    print('Synthetic Disease acc: {:.4f}'.format(accuracy_test_dis_2))    
    print('Combined plant disease acc: {:.4f}'.format(accuracy_test_combine))
    print('Total testing loss: {0}'.format(tcd))  
    print("")

    # Unseen test dataset evaluation 
    print(f"\nUnseen Testing")
    total_test = 0
    correct_test_cls = 0
    correct_test_dis = 0
    correct_test_cls_2 = 0
    correct_test_dis_2 = 0
    correct_test_combine = 0
    model_cls.eval()
    model_dis.eval()
    model_classifier.eval()
  
    for images, labels_cls, labels_dis in tqdm(test_loader2):
        images, labels_cls, labels_dis = images.to(device), labels_cls.to(device) , labels_dis.to(device)
        # To obtain the combined labels from individual concepts       
        label_combine_test = (labels_cls+((num_disease-1)*labels_cls))+labels_dis
        label_combine_test_list = label_combine_test.cpu().numpy()

        # Obtain the output for original crop, original disease, synthetic crop and synthetic disease        
        output = model_classifier(images,images)
        
        # Obtain the predicted label
        predictions_test_cls = torch.max(output[0], 1)[1].to(device)
        predictions_test_cls_list = predictions_test_cls.cpu().numpy()
        predictions_test_cls_list_full = np.append(predictions_test_cls_list_full,predictions_test_cls_list)      
        predictions_test_dis = torch.max(output[1], 1)[1].to(device)
        predictions_test_dis_list = predictions_test_dis.cpu().numpy()
        predictions_test_dis_list_full = np.append(predictions_test_dis_list_full,predictions_test_dis_list)
        predictions_test_cls_2 = torch.max(output[2], 1)[1].to(device)
        predictions_test_dis_2 = torch.max(output[3], 1)[1].to(device)

        # Calculate accuracy        
        correct_test_cls += (predictions_test_cls == labels_cls).sum()
        correct_test_dis += (predictions_test_dis == labels_dis).sum() 
        correct_test_cls_2 += (predictions_test_cls_2 == labels_cls).sum()
        correct_test_dis_2 += (predictions_test_dis_2 == labels_dis).sum()
        
        # Post-calculation for plant disease identification prediction        
        for x in range(len(labels_cls)):
            if (predictions_test_cls_2[x] == labels_cls[x]):
                if (predictions_test_dis_2[x] == labels_dis[x]):
                    correct_test_combine += 1        
        total_test += len(labels_cls)

    # Total accuracy calculations        
    accuracy_test_cls  = correct_test_cls * 100 / total_test
    accuracy_test_dis  = correct_test_dis * 100 / total_test
    accuracy_test_cls_2  = correct_test_cls_2 * 100 / total_test
    accuracy_test_dis_2  = correct_test_dis_2 * 100 / total_test  
    accuracy_test_combine  = correct_test_combine * 100 / total_test
    print("")
    print('Original crop acc: {:.4f}'.format(accuracy_test_cls))
    print('Original Disease acc: {:.4f}'.format(accuracy_test_dis))
    print('Synthetic crop acc: {:.4f}'.format(accuracy_test_cls_2))
    print('Synthetic Disease acc: {:.4f}'.format(accuracy_test_dis_2))    
    print('Combined plant disease acc: {:.4f}'.format(accuracy_test_combine)) 
    print("")

    # Model and optimizer saving    
    if ((epoch+1) % 1 == 0):
        print("Saving Model")
        torch.save(model_classifier.state_dict(), os.path.join(Save_model_path,'{}_{}e_{}L_{:.4f}_{:.4f}_FFViT_model.pth'
                                                                .format(model_name,epoch+1,learning_rate_classifier,accuracy_train,accuracy_test_combine)))

        print("Saving optimizer")
        torch.save(optimizer_classifier.state_dict(), os.path.join(Save_model_path,'{}_{}e_{}L_{:.4f}_{:.4f}_FFViT_optimizer.pth'
                                                                    .format(model_name,epoch+1,learning_rate_classifier,accuracy_train,accuracy_test_combine)))

        print("Saving done")

print("Training done") 






