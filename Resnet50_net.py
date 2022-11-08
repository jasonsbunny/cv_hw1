from Resnet50_block import *

class ResNet50(nn.Cell):
    def __init__(self, class_num=6, input_size=224):
        super(ResNet50, self).__init__()
        self.STAGE0 = nn.SequentialCell([
            nn.Conv2d(3, 64, 7, stride=2, data_format='NCHW'),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'),
        ])
        self.STAGE1 = nn.SequentialCell([
            ResBlock3DS(64, 56, 64, 1),
            ResBlock3RT(256, input_size // 4),
            ResBlock3RT(256, input_size // 4),
        ])
        self.STAGE2 = nn.SequentialCell([
            ResBlock3DS(256, 56, 128, 2),
            ResBlock3RT(512, input_size // 8),
            ResBlock3RT(512, input_size // 8),
            ResBlock3RT(512, input_size // 8),
        ])
        self.STAGE3 = nn.SequentialCell([
            ResBlock3DS(512, 28, 256, 2),
            ResBlock3RT(1024, input_size // 16),
            ResBlock3RT(1024, input_size // 16),
            ResBlock3RT(1024, input_size // 16),
            ResBlock3RT(1024, input_size // 16),
            ResBlock3RT(1024, input_size // 16),
        ])
        self.STAGE4 = nn.SequentialCell([
            ResBlock3DS(1024, 14, 512, 2),
            ResBlock3RT(2048, input_size // 32),
            ResBlock3RT(2048, input_size // 32),
        ])
        self.AVERPOOL = nn.AvgPool2d(input_size // 32)
        self.FLATTEN = nn.Flatten()
        self.FC = nn.SequentialCell([
            nn.Dense(2048, 1024),
            nn.ReLU(),
            nn.Dense(1024, 512),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(512, class_num),
            nn.Sigmoid(),
        ])

    def construct(self, x):
        x = self.STAGE0(x)
        x = self.STAGE1(x)
        x = self.STAGE2(x)
        x = self.STAGE3(x)
        x = self.STAGE4(x)
        x = self.AVERPOOL(x)
        x = self.FLATTEN(x)
        y = self.FC(x)
        return y