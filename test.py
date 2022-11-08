from mindspore import load_checkpoint, load_param_into_net, Model, ops
from Resnet50_net import ResNet50
from train import MultiClassDiceLoss
from load_data import Dataset
import mindspore.nn as nn
import mindspore.numpy as np
import matplotlib.pyplot as plt


import os

class_num=6
input_size=224
batch_size=50
test_dir='plant_dataset/test/images/'
csv_fname_test='plant_dataset/test/test_label.csv'



if __name__ == '__main__':
    #得到测试集数据
    dataset_test=Dataset(csv_fname_test,test_dir,batch_size)
    
    net=ResNet50(class_num=class_num, input_size=input_size)
    param_dict = load_checkpoint("modelpath/ResNet50/best.ckpt")
    load_param_into_net(net, param_dict)
    acc=0
    num=0
    for data in dataset_test.create_dict_iterator():
        op= ops.Concat(1)
        label=op((np.reshape(data['label1'],[batch_size,1]), np.reshape(data['label2'],[batch_size,1]), np.reshape(data['label3'],[batch_size,1]), 
        np.reshape(data['label4'],[batch_size,1]), np.reshape(data['label5'],[batch_size,1]), np.reshape(data['label6'],[batch_size,1])))
    
        res = net(data['data'])
        res=(res > 0.5).astype(int)
        metric = nn.Accuracy('multilabel')
        metric.clear()
        metric.update(res, label)
        accuracy = metric.eval()
        acc=acc+accuracy
        num=num+1
        '''结果可视化'''
        tag=['scab','healthy','frog_eye_leaf_spot','rust','complex','powdery_mildew']
        groudtruth=[]
        for i in range(batch_size):
            label0=[]
            for j in range(6):
                if res[i, j] == 1:
                    label0.append(tag[j])
            groudtruth.append(' '.join(label0))
        print(groudtruth)
    print('Accuracy of the network on the test images: %d %%' % (100 * acc/num))