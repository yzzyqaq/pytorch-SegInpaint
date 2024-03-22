import torch
import torch.nn as nn
from inpaint_modules import *
from collections import OrderedDict
import torch.nn.functional as F
from utils import *
try:
    from models.networks.architecture import ResnetBlock
except:
    from architecture import ResnetBlock
from models.networks.network_module import *


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

class Gate_De(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,padding=1,activation=nn.ELU()):
        super(Gate_De,self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2,mode='nearest')
        self.conv = Gate(in_ch=in_ch,out_ch=out_ch,ksize=ksize,stride=stride,padding=padding,activation=activation)

    def forward(self, x):
        x = self.up_sample(x)
        out = self.conv(x)
        return out
    
class SGNet(nn.Module):
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
            Gate(in_ch=ngf,out_ch=3,ksize=7,padding=0),
            #nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )

        self.refine = nn.Sequential(# encoder
            GatedCon2d(4, 48, 7, 1, 3, pad_type = 'zero', activation = 'lrelu', norm = 'none'),
            GatedCon2d(48, 48 * 2, 4, 2, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            GatedCon2d(48 * 2, 48 * 4, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            GatedCon2d(48 * 4, 48 * 4, 4, 2, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            # Bottleneck
            GatedCon2d(48 * 4, 48 * 4, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            GatedCon2d(48 * 4, 48 * 4, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            GatedCon2d(48 * 4, 48 * 4, 3, 1, 2, dilation = 2, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            GatedCon2d(48 * 4, 48 * 4, 3, 1, 4, dilation = 4, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            GatedCon2d(48 * 4, 48 * 4, 3, 1, 8, dilation = 8, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            GatedCon2d(48 * 4, 48 * 4, 3, 1, 16, dilation = 16, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            GatedCon2d(48 * 4, 48 * 4, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            GatedCon2d(48 * 4, 48 * 4, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            # decoder
            TransposeGatedCon2d(48 * 4, 48 * 2, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            GatedCon2d(48 * 2, 48 * 2, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            TransposeGatedCon2d(48 * 2, 48, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in'),
            GatedCon2d(48, 3, 7, 1, 3, pad_type = 'zero', activation = 'tanh', norm = 'none')

        )
        
    def forward(self, x, mask,img):
        x = self.down(x)
        x = self.bottle_neck(x)
        x = self.up(x)
        first_out = self.out(x)
        second_masked_img = img *  mask + first_out *(1 - mask)
        second_in = torch.cat((second_masked_img, mask), 1)     # in: [B, 4, H, W]
        second_out = self.refine(second_in)                 # out: [B, 3, H, W]
        return first_out ,second_out

        
        #return out # shape:

    '''def generate_fake(self, x, mask,img):
        return self(x)'''

if __name__ == '__main__':
    class Opt():
        def __init__(self, label_nc=35):
            self.label_nc = label_nc

    label_nc = 35
    nc = 3
    opt = Opt(label_nc=label_nc)
    x = torch.randn(2, label_nc+nc, 256, 256).cuda()
    model = SGNet(opt)
    model.cuda()

    out = model(x)
    import pdb; pdb.set_trace()