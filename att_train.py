import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import model
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import os
from utils import Utils, Logger
#from Text2ImgDataset import Text2ImgDataSet
from torchvision import transforms
from datafolder.folder import Attribute_Dataset


class Trainer(object):
    def __init__(self,dataset_path,batch_size, num_workers, epochs,save_path):
        self.attribute = torch.nn.DataParallel(model.attribute().cuda())
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = epochs
        self.dataset = Attribute_Dataset(dataset_path, dataset_name='Market-1501')
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers)
        self.checkpoints = 'attributes'
        self.save_path = save_path

    def train(self):
        iteration = 0
        criterion = F.binary_cross_entropy_with_logits
        finetuned_params = []
        new_params = []
        for n, p in self.attribute.named_parameters():
            if n.find('classifier') >= 0:
                new_params.append(p)
            else:
                finetuned_params.append(p)
        param_groups = [{'params': finetuned_params, 'lr': 0.001},
                        {'params': new_params, 'lr': 0.001}]

        optimizer = optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=0.0005)
        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                iteration += 1
                data, label, id, name = sample
                imgs = Variable(data).cuda()
                targets = Variable(label).cuda()
                outputs, intermediate = self.attribute(imgs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch) % 10 == 0:
                path = os.path.join(self.checkpoints, self.save_path)
                if not os.path.exists(path):
                    os.makedirs(path)

                torch.save(self.attribute.state_dict(), '{0}/attr_{1}.pth'.format(path, epoch))
