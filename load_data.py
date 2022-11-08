import mindspore.dataset as ds
import pandas as pd
import cv2
import os
import mindspore.numpy as np
from Onehot import onehot
from mindspore.dataset.vision import RandomHorizontalFlip, RandomRotation, HWC2CHW, Inter, RandomColorAdjust
from mindspore import dtype, Tensor
from mindspore.dataset.transforms import TypeCast
import random


class DatasetGenerator:
    def __init__(self,img,labels):
        self.data = img
        self.label1 = labels[:, 0]
        self.label2 = labels[:, 1]
        self.label3 = labels[:, 2]
        self.label4 = labels[:, 3]
        self.label5 = labels[:, 4]
        self.label6 = labels[:, 5]

    def __getitem__(self, index):
        return self.data[index], self.label1[index], self.label2[index], self.label3[index], self.label4[index], self.label5[index], self.label6[index],

    def __len__(self):
        return len(self.data)

def Dataset(csv_fname, img_dir, batch_size):
    imgs = []
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('images')
    i=0
    for img_name, target in csv_data.iterrows():
        img = cv2.imread(os.path.join(img_dir, f'{img_name}'))
        img = cv2.resize(img, (224, 224))
        img = img / 255
        imgs.append(img)

    labels = onehot(csv_data.labels)
    dataset_generator = DatasetGenerator(imgs, labels)
    dataset = ds.GeneratorDataset(dataset_generator,
                                  ["data", "label1", "label2", "label3", "label4", "label5", "label6"], shuffle=False)
    
    random_hor_flip_op = RandomHorizontalFlip()
    random_rotation_op = RandomRotation(10, resample=Inter.BICUBIC)
    type_cast_op_image = TypeCast(dtype.float32)
    hwc2chw = HWC2CHW()

    preprocess_operation = [random_hor_flip_op,
                            random_rotation_op,
                            type_cast_op_image,
                            hwc2chw]

    dataset = dataset.map(operations=preprocess_operation,
                            input_columns="data")
    dataset = dataset.map(type_cast_op_image,
                            input_columns="label1")
    dataset = dataset.map(type_cast_op_image,
                            input_columns="label2")
    dataset = dataset.map(type_cast_op_image,
                            input_columns="label3")
    dataset = dataset.map(type_cast_op_image,
                            input_columns="label4")
    dataset = dataset.map(type_cast_op_image,
                            input_columns="label5")
    dataset = dataset.map(type_cast_op_image,
                            input_columns="label6")
    dataset = dataset.batch(batch_size)
    return dataset