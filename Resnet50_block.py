from mindspore import nn


#定义网络结构
class ResBlock3DS(nn.Cell):
    def __init__(self, C, W, C1, S):
        super(ResBlock3DS, self).__init__()
        self.seq = nn.SequentialCell(
            [
                nn.Conv2d(C, C1, 1, stride=S,data_format='NCHW'),
                nn.BatchNorm2d(C1),
                nn.ReLU(),
                nn.Conv2d(C1, C1, 3, stride=1,data_format='NCHW'),
                nn.BatchNorm2d(C1),
                nn.ReLU(),
                nn.Conv2d(C1, C1 * 4, 1, stride=1,data_format='NCHW'),
                nn.BatchNorm2d(C1 * 4),
            ])
        self.shortup = nn.SequentialCell(
            [
                nn.Conv2d(C, C1 * 4, 1, stride=S,data_format='NCHW'),
                nn.BatchNorm2d(C1 * 4),
            ])
        self.relu = nn.ReLU()
        self.se=nn.SequentialCell(
            [
                nn.AvgPool2d(kernel_size=W // S, stride=1),
                nn.Conv2d(C1 * 4, C1 // 4, 1, stride=1),
                nn.ReLU(),
                nn.Conv2d(C1 // 4, C1 * 4, 1, stride=1),
                nn.Sigmoid()
            ])

    def construct(self, x):
        y1 = self.seq(x)
        y2 = self.se(y1)
        y1 = y1 * y2
        y1 = y1 + self.shortup(x)
        return self.relu(y1)


class ResBlock3RT(nn.Cell):
    def __init__(self, C, W):
        super(ResBlock3RT, self).__init__()
        self.seq = nn.SequentialCell(
            [
                nn.Conv2d(C, C // 4, 1, stride=1,data_format='NCHW'),
                nn.BatchNorm2d(C // 4),
                nn.ReLU(),
                nn.Conv2d(C // 4, C // 4, 3, stride=1,data_format='NCHW'),
                nn.BatchNorm2d(C // 4),
                nn.ReLU(),
                nn.Conv2d(C // 4, C, 1, stride=1,data_format='NCHW'),
                nn.BatchNorm2d(C),
            ])
        self.relu = nn.ReLU()
        self.se=nn.SequentialCell(
            [
                nn.AvgPool2d(kernel_size=W, stride=1),
                nn.Conv2d(C, C // 16, 1, stride=1),
                nn.ReLU(),
                nn.Conv2d(C // 16, C, 1, stride=1),
                nn.Sigmoid()
            ])

    def construct(self, x):
        y1 = self.seq(x)
        y2 = self.se(y1)
        y1 = y1 * y2
        return self.relu(x + y1)