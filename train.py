import mindspore.nn as nn
from mindspore import Model, ops
from Resnet50_net import ResNet50
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, Callback
import logging
import mindspore.numpy as np
import mindspore

class_num=6
input_size=224
eval_per_epoch = 1
epochs=20
batch=25
lr=1e-3

class EvalCallBack(Callback):
    """自定义评估函数"""
    # define the operator required
    def __init__(self, models, eval_dataset, eval_per_epochs, epochs_per_eval):
        super(EvalCallBack, self).__init__()
        self.models = models
        self.eval_dataset = eval_dataset
        self.eval_per_epochs = eval_per_epochs
        self.epochs_per_eval = epochs_per_eval
        self.max_valid_acc = 0
        self.local_best_pt = 'modelpath/ResNet50/best.ckpt'

    # define operator function in epoch end
    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        network = cb_param.network
        
        if cur_epoch % self.eval_per_epochs == 0:
            acc = self.models.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epochs_per_eval["epoch"].append(cur_epoch)
            self.epochs_per_eval["acc"].append(acc["Accuracy"])
            if acc["Accuracy"] > self.max_valid_acc:
                self.max_valid_acc =acc["Accuracy"]
                mindspore.save_checkpoint(network, self.local_best_pt)
                logging.info('Save the best state to {}'.format(self.local_best_pt))
            print(acc)

class MyAccuracy(nn.Metric):
    '''自定义评价函数Metrics'''
    def __init__(self):
        super(MyAccuracy, self).__init__()
        self.clear()

    def clear(self):
        """初始化变量_abs_error_sum和_samples_num"""
        self._samples_num = 0    # 累计数据量

    def update(self, data, label1, label2, label3, label4, label5, label6):
        """更新_abs_error_sum和_samples_num"""
        op= ops.Concat(1)
        label=op((np.reshape(label1,[batch,1]), np.reshape(label2,[batch,1]), np.reshape(label3,[batch,1]), 
                  np.reshape(label4,[batch,1]), np.reshape(label5,[batch,1]), np.reshape(label6,[batch,1])))
        data=(data > 0.5).astype(int)
        self.accuracy = nn.Accuracy('multilabel')
        self.accuracy.clear()
        self.accuracy.update(data, label)

    def eval(self):
        """计算最终评估结果"""
        return self.accuracy.eval()


class CustomWithEvalCell(nn.Cell):
    """自定义多标签评估网络"""

    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label1, label2, label3, label4, label5, label6):
        output = self.network(data)
        return output, label1, label2, label3, label4, label5, label6

class MultiClassDiceLoss(nn.LossBase):
    '''自定义多标签损失函数'''
    def __init__(self, reduction="mean"):
        super(MultiClassDiceLoss, self).__init__(reduction)
        self.loss = nn.BCELoss()

    def construct(self, base, target1, target2, target3, target4, target5, target6):
        op= ops.Concat(1)
        label=op((np.reshape(target1,[batch,1]), np.reshape(target2,[batch,1]), np.reshape(target3,[batch,1]), 
                  np.reshape(target4,[batch,1]), np.reshape(target5,[batch,1]), np.reshape(target6,[batch,1])))
        output=self.loss(base,label)
        return self.get_loss(output)


class CustomWithLossCell(nn.Cell):
    '''将前向网络与损失函数连接起来'''
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label1, label2, label3, label4, label5, label6):
        output = self._backbone(data)
        return self._loss_fn(output, label1, label2, label3, label4, label5, label6)


def train(dataset_train,dataset_val):
    logging.basicConfig(level=logging.DEBUG,
                        format=r'[ResNet Train Log]%(asctime)-15s - %(levelname)s - %(message)s',
                        )

    ckpoint_cb=ModelCheckpoint(config=CheckpointConfig(save_checkpoint_steps=20),
                                                        directory='modelpath/ResNet50/')
    
    net = ResNet50(class_num=class_num, input_size=input_size)
    eval_net = CustomWithEvalCell(net)
    optim = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)
    loss = MultiClassDiceLoss(reduction="mean")
    loss_net = CustomWithLossCell(net, loss)
    model = Model(network=loss_net, optimizer=optim, eval_network=eval_net, metrics={"Accuracy": MyAccuracy()})
    epoch_per_eval = {"epoch": [], "acc": []}
    eval_cb = EvalCallBack(model, dataset_val, eval_per_epoch, epoch_per_eval)
    logging.info('Train begin:\nepoch {}\tlearning rate {}\tbatch size {}'.format(epochs, lr, batch))
    model.train(epochs, dataset_train, callbacks=[ckpoint_cb, LossMonitor(), eval_cb], dataset_sink_mode=False)
    logging.info('Train complete.')