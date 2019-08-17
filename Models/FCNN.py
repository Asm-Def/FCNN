from Models.Model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    '''
    Convolution(with padding=1) + Batch-Norm + ReLU
    '''
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'):
        super(ConvBlock, self).__init__()
        # if in_channel == 3:
        #     groups = 1
        # else:
        #     groups = min(in_channel, out_channel)
        self.CNN1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.CNN2 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.BN = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        y = F.relu(self.BN(self.CNN1(x)), inplace=True)
        return F.relu(self.BN(self.CNN2(y)), inplace=True)


class FCNN(Model):
    def __init__(self, conv_size=3):
        super(FCNN, self).__init__('FCNN')
        self.B1 = ConvBlock(3, 64)
        self.B2 = ConvBlock(64, 128)
        self.B3 = ConvBlock(128, 256)
        self.B4 = ConvBlock(256, 512)
        self.B5 = ConvBlock(512, 1024)
        self.U5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.B6 = ConvBlock(1024, 512)
        self.U6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.B7 = ConvBlock(512, 256)
        self.U7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.B8 = ConvBlock(256, 128)
        self.U8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.B9 = ConvBlock(128, 64)
        self.Out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        out1 = self.B1(x)
        tmp = F.max_pool2d(out1, 2)
        tmp = F.dropout(tmp, 0.25)

        out2 = self.B2(tmp)
        tmp = F.max_pool2d(out2, 2)
        tmp = F.dropout(tmp, 0.5)

        out3 = self.B3(tmp)
        tmp = F.max_pool2d(out3, 2)
        tmp = F.dropout(tmp, 0.5)

        out4 = self.B4(tmp)
        tmp = F.max_pool2d(out4, 2)
        tmp = F.dropout(tmp, 0.5)

        out5 = self.B5(tmp)

        uconv4 = self.U5(out5)
        uconv4 = F.dropout2d(torch.cat((out4, uconv4), 1))
        uconv4 = self.B6(uconv4)

        uconv3 = self.U6(uconv4)
        uconv3 = F.dropout2d(torch.cat((out3, uconv3), 1))
        uconv3 = self.B7(uconv3)

        uconv2 = self.U7(uconv3)
        uconv2 = F.dropout2d(torch.cat((out2, uconv2), 1))
        uconv2 = self.B8(uconv2)

        uconv1 = self.U8(uconv2)
        uconv1 = F.dropout2d(torch.cat((out1, uconv1), 1))
        uconv1 = self.B9(uconv1)

        return torch.sigmoid(self.Out(uconv1))


def test():
    torch.cuda.current_device()
    torch.cuda._initialized = True
    net = FCNN().cuda()
    tensor = torch.randn((4, 3, 512, 512), device=torch.device('cuda'))
    out = net(tensor)
    return net, tensor, out

