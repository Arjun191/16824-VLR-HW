from __future__ import print_function

import imageio
import numpy as np
import os
import xml.etree.ElementTree as ET

import torch
import torch.nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    INV_CLASS = {}
    for i in range(len(CLASS_NAMES)):
        INV_CLASS[CLASS_NAMES[i]] = i

    def __init__(self, split, size, data_dir='data/VOCdevkit/VOC2007/'):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.size = size
        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')

        split_file = os.path.join(data_dir, 'ImageSets/Main', split + '.txt')
        with open(split_file) as fp:
            self.index_list = [line.strip() for line in fp]

        self.anno_list = self.preload_anno()

    @classmethod
    def get_class_name(cls, index):
        return cls.CLASS_NAMES[index]

    @classmethod
    def get_class_index(cls, name):
        return cls.INV_CLASS[name]

    def __len__(self):
        return len(self.index_list)    


    def preload_anno(self):
        """
        :return: a list of labels. each element is in the form of [class, weight],
         where both class and weight are a numpy array in shape of [20],
        """
        label_list = []
        for index in self.index_list:
            fpath = os.path.join(self.ann_dir, index + '.xml')
            tree = ET.parse(fpath)
            root = tree.getroot()
            # i = 0
                        
            #######################################################################
            # TODO: Insert your code here to preload labels
            # Hint: the folder Annotations contains .xml files with class labels
            # for objects in the image. The `tree` variable contains the .xml
            # information in an easy-to-access format (it might be useful to read
            # https://docs.python.org/3/library/xml.etree.elementtree.html)
            # Loop through the `tree` to find all objects in the image
            #######################################################################
            # print('enter')
            #  The class vector should be a 20-dimensional vector with class[i] = 1 if an object of class i is present in the image and 0 otherwise
            class_vec = torch.zeros(20)
            weight_vec = torch.ones(20)

            for obj in root.findall('object'):
                obj_class = obj.find('name').text
                # print(obj_class)
                i = self.get_class_index(obj_class)
                class_vec[i] = 1
                obj_diff = int(obj.find('difficult').text)
                weight_vec[i] = 0 if obj_diff == 1 else 1


            # The weight vector should be a 20-dimensional vector with weight[i] = 0 iff an object of class i has the `difficult` attribute set to 1 in the XML file and 1 otherwise
            # The difficult attribute specifies whether a class is ambiguous and by setting its weight to zero it does not contribute to the loss during training 

            # for obj in tree.findall('object'):
            #     obj_class = obj.find('name').text
            #     print(obj_class)
            #     obj_diff = int(obj.find('difficult').text)
            #     i = self.get_class_index(obj_class)
            #     if obj_diff == 1:
            #         weight_vec[i] = 0
            # print('end')         
            ######################################################################
            #                            END OF YOUR CODE                        #
            ######################################################################
            # print(index)

            label_list.append((class_vec, weight_vec))

        return label_list

    def get_random_augmentations(self):
        ######################################################################
        # TODO: Return a list of random data augmentation transforms here
        # NOTE: make sure to not augment during test and replace random crops
        # with center crops. Hint: There are lots of possible data
        # augmentations. Some commonly used ones are random crops, flipping,
        # and rotation. You are encouraged to read the docs, which is found
        # at https://pytorch.org/vision/stable/transforms.html
        # Depending on the augmentation you use, your final image size will
        # change and you will have to write the correct value of `flat_dim`
        # in line 46 in simple_cnn.py
        ######################################################################
        pass
        if self.split == 'test':
            return []
        else:
            random_augments = [
                transforms.RandomHorizontalFlip(p=0.5)
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomCrop(self.size)
                ]

        # if self.split=='trainval':
        #   return [transforms.RandomHorizontalFlip(p=0.5),
        #   transforms.RandomVerticalFlip(p=0.5),
        # #   transforms.RandomRotation(degrees=(-11, 11)),
        #   transforms.RandomResizedCrop(self.size)] # Target size 64x64
        #   # return [transforms.CenterCrop(self.size)]
        # else:
        #   return [transforms.CenterCrop(self.size)]


        return random_augments
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def __getitem__(self, index):
        """
        :param index: a int generated by Dataloader in range [0, __len__()]
        :return: index-th element
        image: FloatTensor in shape of (C, H, W) in scale [-1, 1].
        label: LongTensor in shape of (Nc, ) binary label
        weight: FloatTensor in shape of (Nc, ) difficult or not.
        """
        findex = self.index_list[index]
        fpath = os.path.join(self.img_dir, findex + '.jpg')

        img = Image.open(fpath)
        # print(fpath)
        # print("Enter")
        if self.size == None:
            print(self.size)
        trans = transforms.Compose([
            transforms.Resize((self.size,self.size)),
            *self.get_random_augmentations(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.457, 0.407], std=[0.5, 0.5, 0.5]),
        ])
        # print("Exit")

        img = trans(img)
        lab_vec, wgt_vec = self.anno_list[index] 
        image = torch.FloatTensor(img)
        # print(image.shape)
        label = torch.FloatTensor(lab_vec)
        wgt = torch.FloatTensor(wgt_vec)

        return image, label, wgt
