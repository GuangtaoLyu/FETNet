from modules.vgg16 import VGG16FeatureExtractor
import os
import time
import numpy as np
import torch
from modules.Losses import *
import torch.optim as optim
from torchvision.utils import make_grid
from torchvision.utils import save_image
from modules.Discriminator import Discriminator
from modules.FETNet import FETNet
from utils.log import get_logger
from utils.erode import *

class FETNetModel():
    def __init__(self):
        self.G = None
        self.lossNet = None
        self.epoch = None
        self.optm_G = None
        self.device = None
        self.real_B = None
        self.fake_B = None
        self.grey = None
        self.logger = get_logger()
        self.epoch =0

    def initialize_model(self, path_g=None, path_d=None, train=True):
        self.lr = 1e-3
        self.G = FETNet(3)
        self.optm_G = optim.Adam(self.G.parameters(), lr=self.lr)
        self.lossNet = VGG16FeatureExtractor()
        self.D = Discriminator(3)
        self.optm_G = optim.Adam(self.G.parameters(), lr=self.lr )
        self.optm_D = optim.Adam(self.D.parameters(), lr=2 * self.lr)
        self.adversarial_loss = AdversarialLoss()
        self.style_loss = style_loss
        self.l1_loss = l1_loss
        self.preceptual_loss = preceptual_loss
        self.diceLoss = diceLoss
        self.iter = 0
        self.total_loss_d = 0
        self.total_loss_g = 0

        try:
            ckpt_dict = torch.load(path_g)
            self.G.load_state_dict(ckpt_dict)
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr=self.lr)
                self.optm_D = optim.Adam(self.D.parameters(), lr=2 * self.lr)

        except:
            self.epoch = 0
            print('No trained model, train from beginning')

    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            self.G.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
                self.D.cuda()
                self.adversarial_loss.cuda()
        else:
            self.device = torch.device("cpu")

    def adjust_learning_rate(self,optimizer, epoch):
        lr = self.lr * (0.5 ** (epoch // 8))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, train_loader, save_path, finetune=False, epochs=500):
        self.epoch = 0
        if finetune:
            self.optm_G = optim.Adam(filter(lambda p:p.requires_grad, self.G.parameters()), lr = 5e-5)
            self.optm_D = optim.Adam(self.D.parameters(), lr = 5e-6)
        print("Starting training from epoch:{:d}".format(self.epoch))
        s_time = time.time()
        while self.epoch < epochs:
            if self.epoch <57:
                self.adjust_learning_rate(self.optm_G, self.epoch)
                self.adjust_learning_rate(self.optm_D, self.epoch)
            for items in train_loader:
                gt_images, masks,text = self.__cuda__(*items)
                masks = tensor_erode(masks,3)
                self.forward(text, masks, gt_images)
                self.update_parameters()
                self.iter+=1

                if self.iter%50==0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    self.logger.info("iter:%6d, g_loss:%.4f, d_loss:%.4f, time_taken:%.2f" % (self.iter, self.total_loss_g/50,self.total_loss_d/50, int_time))
                    self.total_loss_g = 0
                    self.total_loss_d = 0
                    s_time = time.time()

            self.epoch = self.epoch+1
            if self.epoch % 5 == 0:
                if not os.path.exists('{:s}'.format(save_path)):
                    os.makedirs('{:s}'.format(save_path))

                torch.save(self.G.state_dict(), '{:s}/g_{:d}.pth'.format(save_path, self.epoch))
                torch.save(self.D.state_dict(), '{:s}/d_{:d}.pth'.format(save_path, self.epoch))

        self.logger.info("Train finished!")

    def test(self, test_loader, result_save_path):
        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False

        for gt_images, masks, text, name in test_loader:
            text = text.cuda()
            fake_B, masks_out = self.G(text)
            comp_B = fake_B * (1 - masks_out) + text * masks_out
            if not os.path.exists('{:s}/'.format(result_save_path)):
                os.makedirs('{:s}/'.format(result_save_path))
            if not os.path.exists('{:s}/erase/'.format(result_save_path)):
                os.makedirs('{:s}/erase/'.format(result_save_path))
            if not os.path.exists('{:s}/mask/'.format(result_save_path)):
                os.makedirs('{:s}/mask/'.format(result_save_path))

            for k in range(comp_B.size(0)):
                grid = make_grid(comp_B[k:k + 1])
                file_path = '{:s}/erase/{:s}.png'.format(result_save_path, name[k])
                save_image(grid, file_path)

                grid = make_grid(masks_out[k:k + 1])
                file_path = '{:s}/mask/{:s}.png'.format(result_save_path, name[k])
                save_image(1-grid, file_path)


    def forward(self, text,mask, gt_image):
        self.real_B = gt_image
        self.mask = mask
        fake_B, fake_mask = self.G(text)
        self.fake_mask = fake_mask
        self.fake_B = fake_B
        self.com_B = fake_B*(1-self.fake_mask)+text*self.fake_mask

    def update_parameters(self):
        self.update_D()
        self.update_G()

    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        self.total_loss_g += loss_G.detach().item()
        loss_G.backward()
        self.optm_G.step()

    def update_D(self):
        self.optm_D.zero_grad()
        loss_D = self.get_d_loss()
        self.total_loss_d += loss_D.detach().item()
        loss_D.backward()
        self.optm_D.step()

    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        com_B  = self.com_B

        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        com_B_feats = self.lossNet(com_B)

        # adv_loss
        pred_fake = self.D(fake_B,self.mask*fake_B)
        loss_D = self.adversarial_loss(pred_fake, True)

        style_loss = self.style_loss(real_B_feats, fake_B_feats) +  self.style_loss(real_B_feats, com_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) +  self.preceptual_loss(real_B_feats, com_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask) + self.l1_loss(real_B, com_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask)) +  self.l1_loss(real_B, com_B, (1 - self.mask))
        mask_loss = self.diceLoss(self.mask,self.fake_mask)

        loss_G = (style_loss * 120
                  + preceptual_loss * 0.05
                  + valid_loss * 2
                  + hole_loss * 10
                  + mask_loss * 3
                  + loss_D * 0.1)
        if self.iter % 50 == 0:
            self.logger.info("    lossG:%.4f\nstyle_loss:%.4f\npreceptual_loss:%.4f\nvalid_loss:%.4f\nhole_loss:%.4f\nmask_loss:%.4f\nloss_D:%.4f\n" % (loss_G,style_loss,preceptual_loss,valid_loss,hole_loss,mask_loss,loss_D))

        return loss_G

    def get_d_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B.detach()

        pred_real = self.D(real_B,self.mask*real_B)
        pred_fake = self.D(fake_B,self.mask*fake_B)

        loss_D = (self.adversarial_loss(pred_real, True, True)  + self.adversarial_loss(pred_fake, False, True))/2
        if self.iter % 50 == 0:
            self.logger.info("iter:{:<6d} d_loss:{:.4f}".format(self.iter,loss_D))
        return loss_D


    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)