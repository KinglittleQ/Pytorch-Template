import torch
import os
import os.path as osp


class TemplateModel():

    def __init__(self):
        self.writer = None
        self.train_logger = None  # not neccessary
        self.eval_logger = None  # not neccessary
        self.args = None  # not neccessary

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

    def check_init(self):
        assert self.model
        assert self.optimizer
        assert self.criterion
        assert self.metric
        assert self.train_loader
        assert self.test_loader
        assert self.device
        assert self.ckpt_dir
        assert self.log_per_step

        if not osp.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

    def load_state(self, fname, optim=True):
        state = torch.load(fname)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state['model'])
        else:
            self.model.load_state_dict(state['model'])

        if optim and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        self.step = state['step']
        self.epoch = state['epoch']
        self.best_error = state['best_error']
        print('load model from {}'.format(fname))

    def save_state(self, fname, optim=True):
        state = {}

        if isinstance(self.model, torch.nn.DataParallel):
            state['model'] = self.model.module.state_dict()
        else:
            state['model'] = self.model.state_dict()

        if optim:
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

            loss, others = self.train_loss(batch)

            loss.backward()
            self.optimizer.step()

            if self.step % self.log_per_step == 0:
                self.writer.add_scalar('loss', loss.item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, loss.item()))
                if self.train_logger:
                    self.train_logger(self.writer, others)

    def train_loss(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        pred = self.model(x)
        loss = self.criterion(pred, y)

        return loss, None

    def eval(self):
        self.model.eval()
        
        error, others = self.eval_error()

        if error < self.best_error:
            self.best_error = error
            self.save_state(osp.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(osp.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))

        self.writer.add_scalar('error', error, self.epoch)
        print('epoch {}\terror {:.3}'.format(self.epoch, error))

        if self.eval_logger:
            self.eval_logger(self.writer, others)

        return error

    def eval_error(self):
        xs, ys, preds = [], [], []
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

        return error, None

    def inference(self, x):
        x = x.to(self.device)
        return self.model(x)

    def num_parameters(self):
        return sum([p.data.nelement() for p in self.model.parameters()])
