import load_data
from train import train
import mindspore.numpy as np

train_dir='plant_dataset/train/images/'
test_dir='plant_dataset/test/images/'
val_dir='plant_dataset/val/images/'
csv_fname_train='plant_dataset/train/train_label.csv'
csv_fname_test='plant_dataset/test/test_label.csv'
csv_fname_val='plant_dataset/val/val_label.csv'

batch_size=25

if __name__ == '__main__':

    # 加载数据
    dataset_train=load_data.Dataset(csv_fname_train,train_dir,batch_size)
    dataset_val=load_data.Dataset(csv_fname_val,val_dir,batch_size)
    train(dataset_train,dataset_val)
