import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
from torchvision import models
import torch.nn.init as init
import torch.nn.functional as F
from resnet import resnet50

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 1
        self.noise_dim = 100
        self.embed_dim = 30
        self.latent_dim = self.noise_dim + self.embed_dim
        self.ngf = 64

        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 8 x 8
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 16 x 16
        )

    def forward(self, embed_vector, z):
        embed_vector = embed_vector.unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([embed_vector, z], 1)
        output = self.netG(latent_vector)
        return output


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 1
        self.embed_dim = 30
        # self.projected_embed_dim = 128
        self.ndf = 64
        self.B_dim = 128
        self.C_dim = 16

        self.netD_1 = nn.Sequential(
            # input is (nc) x 16 x 16
            nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 8 x 8
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

        self.netD_2 = nn.Sequential(
            # state size. (ndf*2) x 4 x 4
            # nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
            nn.Conv2d(self.ndf * 2 + self.embed_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp, embed):
        # print(embed.size())
        x_intermediate = self.netD_1(inp)
        replicated_embed = embed.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        # print(replicated_embed.size())
        # print(x_intermediate.size())
        x = torch.cat([x_intermediate, replicated_embed], 1)
        x = self.netD_2(x)

        return x.view(-1, 1).squeeze(1), x_intermediate


class attribute(nn.Module):
    def __init__(self,**kwargs):
        super(attribute,self).__init__()
        self.num_att = 30
        self.last_conv_stride = 2
        self.base = resnet50(pretrained=True, last_conv_stride=self.last_conv_stride)
        self.classifier = nn.Linear(2048, 256)
        self.classifier_2=nn.Linear(256, self.num_att)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)

    def forward(self, x):
        x = self.base(x)
        #print("x.shape[2:]",x.shape[2:])
        #print(x.size())
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        x_intermediate = self.classifier(x) #size 16 x 16 =256
        x = self.classifier_2(x_intermediate)
        #x = self.softmax(x)
        return x,x_intermediate


class DeepMAR_ResNet50_ExtractFeature(object):
    """
    A feature extraction function
    """
    def __init__(self, model, **kwargs):
        self.model = model

    def __call__(self, imgs):
        old_train_eval_model = self.model.training

        # set the model to be eval
        self.model.eval()

        # imgs should be Variable
        if not isinstance(imgs, Variable):
            print 'imgs should be type: Variable'
            raise ValueError
        score = self.model(imgs)
        score=F.sigmoid(score)
        score = score.data.cpu().numpy()

        self.model.train(old_train_eval_model)

        return score




