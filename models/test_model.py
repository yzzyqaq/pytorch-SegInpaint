import torch
import torch.nn as nn
import torch.optim as optim
import models.networks as networks

from models.networks.spnet import SPNet
from models.networks.sgnet import SGNet
from models.networks.spade_discriminator import MultiscaleDiscriminator
from models.networks.loss import GANLoss, GANFeatMatchingLoss, VGGLoss

import torchvision.utils as vutils
import torchvision.transforms as transforms
import os
from networks.losses.high_receptive_pl import HRFPL

def normalize(tensor):
    # assum a batch of img tensor
    return (tensor - 0.5)/0.5

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class testInpaintModel(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        
        
        self.SPNet = SPNet(opt=opt)
        self.SGNet = SGNet(opt=opt)
        self.D_seg = MultiscaleDiscriminator(opt, input_nc=opt.label_nc)
        self.D_img = MultiscaleDiscriminator(opt, input_nc=3)
        checkpoint = torch.load(opt.test_model_path)
        # Load model weights
        epoch = checkpoint['epoch']
        print(epoch)
        self.SPNet.load_state_dict(checkpoint['SPNet'])
        self.SGNet.load_state_dict(checkpoint['SGNet'])
        self.D_seg.load_state_dict(checkpoint['D_seg'])
        self.D_img.load_state_dict(checkpoint['D_img'])

        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.SPNet.cuda()
            self.SGNet.cuda()
            self.D_seg.cuda()
            self.D_img.cuda()

        print("=> finish loading model")

        # loss
        self.criterion_GAN = GANLoss(gan_mode=opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
        self.criterion_Feat = GANFeatMatchingLoss(opt=opt)
        self.run_hrfpl = HRFPL(weight=5, weights_path=os.getcwd())
        self.criterion_VGG = VGGLoss()
    
        self.normalize = normalize

    def forward(self, data, mode, fake_seg=None, fake_img=None):
            """
            - image: in SPG-Net, img_size = cx256x256
            - label, instance, image, path = data
            """
            input_semantics, real_image, corrupted_seg, corrupted_img, occ_mask ,output_semantics= self.preprocess_input(data)

            if mode == 'spn':
                spn_loss, generated_seg = self.compute_spn_loss(input_semantics, corrupted_seg, corrupted_img, output_semantics, occ_mask)
                return spn_loss, generated_seg
            elif mode == 'sgn':
                sgn_loss, generated_img, occ_mask, first_out = self.compute_sgn_loss(fake_seg, real_image, corrupted_img, occ_mask)
                return sgn_loss, generated_img, occ_mask, first_out
            elif mode == 'd_seg':
                d_seg_loss = self.compute_d_seg_loss(input_semantics, corrupted_img, corrupted_seg)
                return d_seg_loss
            elif mode == 'd_img':
                d_img_loss = self.compute_d_img_loss(real_image, fake_seg, corrupted_img, occ_mask)
                #return d_img_loss
                return d_img_loss, real_image, corrupted_img
            else:
                raise ValueError("|mode| is invalid")


    def preprocess_input(self, data):
        # copy from SPADE's Pix2PixModel
        # label: 1~33

# move to GPU and change data types
        data['label'] = data['label'].long()
        data['color'] = data['color'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()
            data['mask'] = data['mask'].cuda()
            data['color'] = data['color'].cuda()
           # data['realseg'] = data['realseg'].cuda()

        image = data['image']
        occ_mask = data['mask']
        #label_map = data['label']
        color_map = data['color']
        #realseg = data['realseg']
        
        bs, _, h, w = color_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        
        input_color = self.FloatTensor(bs, nc, h, w).zero_()
        output_semantics = input_color.scatter_(1, color_map, 1.0)
        #print(input_label.shape)
        #
        
        # create one-hot label map
        label_map = data['label']
       # print(label_map.shape)
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc

        # NOTE: let label=0 be masked region
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        #print(input_label.shape)
        input_semantics = input_label.scatter_(1, label_map, 1.0)
        #print(input_semantics.shape)

        corrupted_seg = occ_mask*input_semantics # NOTE haven't checked yet
        corrupted_img = occ_mask * image
        #corrupted_realseg = occ_mask * realseg

        # normalize
        image = self.normalize(image)
        corrupted_img = self.normalize(corrupted_img)
        #realseg = self.normalize(realseg)
        #corrupted_realseg = self.normalize(corrupted_realseg)

        return input_semantics, image, corrupted_seg, corrupted_img, occ_mask, output_semantics#,realseg,corrupted_realseg

    def compute_spn_loss(self, input_semantics, corrupted_seg, corrupted_img,output_semantics,  occ_mask):
        # Generator - SPNet
        G_SPNet_losses = {}

        real_seg = input_semantics

        # for GAN feature matching loss
        pred_real_seg = self.discriminate_seg(real_seg)

        # SP-Net
        input_spn = torch.cat([corrupted_img, corrupted_seg], dim=1) # concate corrupted + output_seg
        fake_seg = self.SPNet(input_spn,occ_mask,input_semantics) # generated fake_seg
        pred_fake_seg = self.discriminate_seg(fake_seg)
        #fake_seg = output_semantics
        G_SPNet_losses['GAN'] = self.criterion_GAN(pred_fake_seg, target_is_real=True, for_discriminator=False)
        G_SPNet_losses['GAN_Feat'] = self.criterion_Feat(pred_fake_seg, pred_real_seg) # or perceptual loss
        #G_SPNet_losses['GAN_Feat'] = 0*self.criterion_Feat(pred_fake_seg, pred_real_seg)
        return G_SPNet_losses, fake_seg
        #return G_SPNet_losses, real_seg
    
    def compute_sgn_loss(self, fake_seg, real_image, corrupted_img, occ_mask):
        # Generator - SGNet
        G_SGNet_losses = {}
        occ_mask = occ_mask.detach()
        occ_mask.requires_grad_()
        # for GAN feature matching loss
        with torch.no_grad():
            pred_real_img = self.D_img(real_image)

        fake_seg = fake_seg.detach()
        fake_seg.requires_grad_()
        input_sgn = torch.cat([corrupted_img, fake_seg], dim=1) # concate corrupted + output_seg
        firstout, fake_image = self.SGNet(input_sgn,occ_mask,real_image) # generated fake_image

        pred_fake_img = self.D_img(fake_image)
        G_SGNet_losses['GAN'] = self.criterion_GAN(pred_fake_img, target_is_real=True, for_discriminator=False)
        G_SGNet_losses['GAN_Feat'] = self.criterion_Feat(pred_fake_img, pred_real_img)
        G_SGNet_losses['hrfpl'] = self.run_hrfpl(fake_image,real_image)
        G_SGNet_losses['VGG'] = self.criterion_VGG(real_image, fake_image)*self.opt.lambda_feat # or alex
        return G_SGNet_losses, fake_image, occ_mask, firstout
    
    def compute_d_seg_loss(self, real_seg, corrupted_img, corrupted_seg):
        # real_seg is input_semantics
        # reference: https://github.com/NVlabs/SPADE/blob/master/models/pix2pix_model.py#L166-L181
        D_seg_losses = {}
        '''with torch.no_grad():
            input_spn = torch.cat([corrupted_img, corrupted_seg], dim=1)
            fake_seg = self.SPNet(input_spn)
            fake_seg = fake_seg.detach()
            fake_seg.requires_grad_()

        pred_fake_seg = self.D_seg(fake_seg)
        D_seg_losses['D_fake'] = self.criterion_GAN(pred_fake_seg, target_is_real=False, for_discriminator=True)

        pred_real_seg = self.D_seg(real_seg)
        D_seg_losses['D_real'] = self.criterion_GAN(pred_real_seg, target_is_real=True, for_discriminator=True)'''
        return D_seg_losses

    def compute_d_img_loss(self, real_image, fake_seg, corrupted_img, occ_mask):
        D_img_losses = {}
        with torch.no_grad():
            input_sgn = torch.cat([corrupted_img, fake_seg], dim=1)
            firstout,fake_image = self.SGNet(input_sgn,occ_mask,real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake_image = self.D_img(fake_image)
        D_img_losses['D_fake'] = self.criterion_GAN(pred_fake_image, target_is_real=False, for_discriminator=True)

        pred_real_image = self.D_img(real_image)
        D_img_losses['D_real'] = self.criterion_GAN(pred_real_image, target_is_real=True, for_discriminator=True)
        return D_img_losses

    def generate_fake_seg(self, input_spn):
        # NOTE: should produce one_hot?
        fake_prob = self.SPNet(input_spn)
        return fake_prob

    def generate_fake_img(self, inpug_sgn):
        return self.SGNet(inpug_sgn)
    
    def generate_test_result(self, fake_seg, corrupted_img, occ_mask,real_image):
        with torch.no_grad():
            input_sgn = torch.cat([corrupted_img, fake_seg], dim=1)
            fake_image = self.SGNet(input_sgn,occ_mask,real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        return fake_image
        

    def discriminate_seg(self, seg):
        pred_seg = self.D_seg(seg)
        return pred_seg

    def discriminate_img(self, img):
        pred_img = self.D_img(img)
        return pred_img

    def create_optimizers(self, opt):
        SPNet_param = list(self.SPNet.parameters())
        SGNet_param = list(self.SGNet.parameters())
        D_seg_param = list(self.D_seg.parameters())
        D_img_param = list(self.D_img.parameters())
    
        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_SPNet = optim.Adam(SPNet_param, lr=G_lr, betas=(beta1, beta2))
        optimizer_SGNet = optim.Adam(SGNet_param, lr=G_lr, betas=(beta1, beta2))
        optimizer_D_seg = optim.Adam(D_seg_param, lr=D_lr, betas=(beta1, beta2))
        optimizer_D_img = optim.Adam(D_img_param, lr=D_lr, betas=(beta1, beta2))

        return optimizer_SPNet, optimizer_SGNet, optimizer_D_seg, optimizer_D_img


    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def save(self, path, epoch):
        states = {

            'SPNet': self.SPNet.cpu().state_dict(),
            'SGNet': self.SGNet.cpu().state_dict(),
            'D_seg': self.D_seg.cpu().state_dict(),
            'D_img': self.D_img.cpu().state_dict(),

            # 'optimizer_SPNet': self.optimizer_SPNet.state_dict(),
            # 'optimizer_SGNet': self.optimizer_SGNet.state_dict(),
            # 'optimizer_D_seg': self.optimizer_D_seg.state_dict(),
            # 'optimizer_D_img': self.optimizer_D_img.state_dict(),
            'epoch': epoch,
        }
        torch.save(states, path)
        if len(self.opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.SPNet.cuda()
            self.SGNet.cuda()
            self.D_seg.cuda()
            self.D_img.cuda()

    def load(self, path):
        checkpoint = torch.load(path)
        
    
        # Load model weights
        self.SPNet.load_state_dict(checkpoint['SPNet'])
        self.SGNet.load_state_dict(checkpoint['SGNet'])
        self.D_seg.load_state_dict(checkpoint['D_seg'])
        self.D_img.load_state_dict(checkpoint['D_img'])

        assert(torch.cuda.is_available())
        self.SPNet.cuda()
        self.SGNet.cuda()
        self.D_seg.cuda()
        self.D_img.cuda()

        # Load epoch information
        epoch = checkpoint['epoch']
        print(epoch)

        return epoch, self.SPNet, self.SGNet  # 或者你想要返回的其他模型相关的对象
    

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0