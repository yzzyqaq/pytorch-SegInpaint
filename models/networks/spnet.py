import torch
import torch.nn as nn
try:
    from models.networks.architecture import ResnetBlock
except:
    from architecture import ResnetBlock
import torch.nn.functional as F



class Gate(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,padding=1,activation=nn.ELU()):
        super(Gate,self).__init__()
        
        self.conv = nn.Conv2d(in_ch,2*out_ch,kernel_size=ksize,stride=stride,padding=padding,dilation=1)
        self.activation = activation


    def forward(self, x):
        raw = self.conv(x)
        x1 = raw.split(int(raw.shape[1]/2),dim=1)
        gate = F.sigmoid(x1[0])
        out = self.activation(x1[1])*gate
        return out
class SPNet(nn.Module):
    def __init__(self, opt, block=ResnetBlock):
        super().__init__()
        self.n_class = opt.label_nc
        self.resnet_initial_kernel_size = 7
        self.resnet_n_blocks = 9
        ngf = 64
        activation = nn.ReLU(False)

        self.down = nn.Sequential(
            nn.ReflectionPad2d(self.resnet_initial_kernel_size // 2),
            #nn.Conv2d(self.n_class+3, ngf, kernel_size=self.resnet_initial_kernel_size, stride=2, padding=0),
            Gate(in_ch=self.n_class+3,out_ch=ngf,ksize=self.resnet_initial_kernel_size,stride=2,padding=0),
            nn.BatchNorm2d(ngf),
            activation,
            
            Gate(in_ch=ngf,out_ch=ngf*2,ksize=3,stride=2,padding=1),
            #nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            activation,

            Gate(in_ch=ngf*2,out_ch=ngf*4,ksize=3,stride=2,padding=1),
            #nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf*4),
            activation,

            Gate(in_ch=ngf*4,out_ch=ngf*8,ksize=3,stride=2,padding=1),
            #nn.Conv2d(ngf*4, ngf*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf*8),
            activation,
        )

        # resnet blocks
        resnet_blocks = []
        for i in range(self.resnet_n_blocks):
            resnet_blocks += [block(ngf*8, norm_layer=nn.BatchNorm2d, kernel_size=3)]
        self.bottle_neck = nn.Sequential(*resnet_blocks)

        self.up = nn.Sequential(
            #Gate_De(in_ch=ngf*8,out_ch=ngf*8,ksize=3,stride=2,padding=1),
            nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf*8),
            activation,

            #Gate_De(in_ch=ngf*8,out_ch=ngf*4,ksize=3,stride=2,padding=1),
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf*4),
            activation,

            #Gate_De(in_ch=ngf*4,out_ch=ngf*2,ksize=3,stride=2,padding=1),
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf*2),
            activation,

            #Gate_De(in_ch=ngf*2,out_ch=ngf*1,ksize=3,stride=2,padding=1),
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf),
            activation,
        )

        self.out = nn.Sequential(
            nn.ReflectionPad2d(self.resnet_initial_kernel_size // 2),
            Gate(in_ch=ngf,out_ch=self.n_class,ksize=7,padding=0),
            #nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )
        
    def forward(self, x,mask, seg):
        x = self.down(x)
        x = self.bottle_neck(x)
        x = self.up(x)
        out = self.out(x)
        seg_out = seg * mask +out *1-mask
        
        return  seg_out # shape:

    def generate_fake(self, x):
        return self(x)


if __name__ == '__main__':
    class Opt():
        def __init__(self, label_nc=35):
            self.label_nc = label_nc

    label_nc = 35
    nc = 3
    opt = Opt(label_nc=label_nc)
    x = torch.zeros(2, label_nc+nc, 256, 256).cuda()
    model = SPNet(opt)
    model.cuda()

    out = model(x)

