import torch
import tensorboardX as tX
import os.path as osp


class Model():

    def __init__(self, args):
        self.writer = tX.SummaryWriter(log_dir=None, comment='')
        self.train_logger = None
        self.eval_logger = None
        self.args = args

        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.metric = None

        self.train_loader = None
        self.test_loader = None

        self.device = None

        self.ckpt_dir = None
        self.log_per_step = None
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

    def inference(self, x):
        x = x.to(self.device)
        return self.model(x)
