import torch
import torch.optim as optim
import torchvision.transforms as tfs
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import tensorboardX as tX
from model import LeNet
import os.path as osp


lr = 0.001
batch_size = 16
n_epochs = 10
eval_per_epoch = 1
log_per_step = 100
device = torch.device('cpu')
log_dir = 'log'
ckpt_dir = 'checkpoint'

class Model():

    def __init__(self, args=None):
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
        self.log_per_step = 100
        # self.eval_per_epoch = None

    def load_state(self, fname):
        state = torch.load(fname)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.step = state['step']
        self.epoch = state['epoch']
        self.best_error = state['best_error']
        print('load model from {}'.format(fname))

    def save_state(self, fname):
        state = {}
        state['model'] = self.model.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        state['step'] = self.step
        state['epoch'] = self.epoch
        state['best_error'] = self.best_error
        torch.save(state, fname)
        print('save model at {}'.format(fname))

    def train(self):
        self.model.train()
        self.epoch += 1
        for batch in self.train_loader:
            self.step += 1
            self.optimizer.zero_grad()

            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)

            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            if self.step % self.log_per_step == 0: 
                self.writer.add_scalar('loss', loss.item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, loss.item()))
                if self.train_logger:
                    self.train_logger(self.writer, x, y, pred)

    def eval(self):
        self.model.eval()
        xs = []
        ys = []
        preds = []
        for batch in self.test_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)

            xs.append(x.cpu())
            ys.append(y.cpu())
            preds.append(pred.cpu())

        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        preds = torch.cat(preds, dim=0)

        error = self.metric(preds, ys)
        if error < self.best_error:
            self.best_error = error
            self.save_state(osp.join(self.ckpt_dir, 'best.pth.tar'))
        self.save_state(osp.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))

        self.writer.add_scalar('error', error, self.epoch)
        print('epoch {}\terror {:.3}'.format(self.epoch, error))

        if self.train_logger:
            self.train_logger(self.writer, xs, ys, preds)

        return error



def metric(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct_num = torch.sum(pred == target).item()
    total_num = target.size(0)
    accuracy = correct_num / total_num
    return 1. - accuracy


def main():
    model = Model()

    for epoch in range(n_epochs):
        model.train()
        if (epoch + 1) % eval_per_epoch == 0:
            model.eval()

    print('Done!!!')


if __name__ == '__main__':
    main()
