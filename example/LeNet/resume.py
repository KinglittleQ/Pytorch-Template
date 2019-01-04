from train import Model


model_path = 'checkpoint/9.pth.tar'
n_epochs = 5
eval_per_epoch = 1


def resume(model_path):
    model = Model()
    if model_path:
        model.load_state(model_path)

    for i in range(n_epochs):
        model.train()

        if model.epoch % eval_per_epoch == 0:
            model.eval()


if __name__ == '__main__':
    resume(model_path)
