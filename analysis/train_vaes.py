if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms, datasets
    import torchbearer
    from torchbearer import Trial, callbacks, metrics
    from torchbearer.callbacks import init
    import torch
    from torch import distributions

    from .vae import VAE, LATENT
    from implementations.torchbearer_implementation import FMix

    import argparse

    parser = argparse.ArgumentParser(description='VAE Training')
    parser.add_argument('--mode', default='base', type=str, help='name of run')
    parser.add_argument('--i', default=1, type=int, help='iteration')
    parser.add_argument('--var', default=1, type=float, help='iteration')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--dir', default='vaes', type=str, help='directory')
    args = parser.parse_args()

    KL = torchbearer.state_key('KL')
    NLL = torchbearer.state_key('NLL')
    SAMPLE = torchbearer.state_key('SAMPLE')

    # Data
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    inv_normalize = transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010))
    transform_base = [transforms.ToTensor(), normalize]

    transform = [transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), transforms.RandomHorizontalFlip()] + transform_base

    transform_train = transforms.Compose(transform)
    transform_test = transforms.Compose(transform_base)

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, 128, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_set, 100, shuffle=False, num_workers=5)

    # KL Divergence

    def kld(prior):
        @callbacks.add_to_loss
        def loss(state):
            res = distributions.kl_divergence(state[LATENT], prior).sum().div(state[LATENT].loc.size(0))
            state[KL] = res.detach()
            return res
        return loss

    # Negative Log Likelihood

    def nll(state):
        y_pred, y_true = state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE]
        res = - y_pred.log_prob(y_true).sum().div(y_true.size(0))
        state[NLL] = res.detach()
        return res

    # Generate Some Images

    @torchbearer.callbacks.on_forward
    @torchbearer.callbacks.on_forward_validation
    def sample(state):
        state[SAMPLE] = state[torchbearer.Y_PRED].loc

    # Train VAEs

    aug = []
    mode = args.mode
    if mode == 'mix':
        aug = [callbacks.Mixup()]
    if mode == 'cutmix':
        aug = [callbacks.CutMix(1, classes=10)]
    if mode == 'fmix':
        aug = [FMix(alpha=1, decay_power=3)]

    model = VAE(64, var=args.var)
    trial = Trial(model, optim.Adam(model.parameters(), lr=5e-2), nll,
                  metrics=[
                      metrics.MeanSquaredError(pred_key=SAMPLE),
                      metrics.mean(NLL),
                      metrics.mean(KL),
                      'loss'
                  ],
                  callbacks=[
                      sample,
                      kld(distributions.Normal(0, 1)),
                      init.XavierNormal(targets=['Conv']),
                      callbacks.MostRecent(args.dir + '/' + mode + '_' + str(args.i) + '.pt'),
                      callbacks.MultiStepLR([40, 80]),
                      callbacks.TensorBoard(write_graph=False, comment=mode + '_' + str(args.i), log_dir='vae_logs'),
                      *aug
                  ])

    if mode in ['base', 'mix', 'cutmix']:
        trial = trial.load_state_dict(torch.load('vaes/' + '/' + mode + '_' + str(args.i) + '.pt'))

    trial.with_generators(train_loader, test_loader).to('cuda').run(args.epochs, verbose=1)
