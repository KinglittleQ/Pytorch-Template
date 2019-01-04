import torch
import torch.optim as optim
import torchvision.transforms as tfs
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import tensorboardX as tX
from model import LeNet
# import os.path as osp
from torchT import TemplateModel


lr = 0.001
batch_size = 16
n_epochs = 1
eval_per_epoch = 1
log_per_step = 100
device = torch.device('cpu')
log_dir = 'log'
ckpt_dir = 'checkpoint'

class Model(TemplateModel):

    def __init__(self, args=None):
        super().__init__()

        self.writer = tX.SummaryWriter(log_dir=log_dir, comment='LeNet')
        self.train_logger = None
        self.eval_logger = None
        self.args = args

        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device('cpu')

        self.model = LeNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric = metric

        transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize((0.1307,), (0.3081,))])
        train_dataset = MNIST(root='MNIST', train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = MNIST(root='MNIST', train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.ckpt_dir = ckpt_dir
        self.log_per_step = log_per_step
        # self.eval_per_epoch = None

        self.check_init()


def metric(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct_num = torch.sum(pred == target).item()
    total_num = target.size(0)
    accuracy = correct_num / total_num
    return 1. - accuracy


def main():
    model = Model()

    for i in range(n_epochs):
        model.train()
        if model.epoch % eval_per_epoch == 0:
            model.eval()

    print('Done!!!')


if __name__ == '__main__':
    main()
