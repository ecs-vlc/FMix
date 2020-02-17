if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms, datasets
    import torchbearer
    from torchbearer import Trial
    import torch
    from torch import distributions

    from .vae import VAE, MINetwork

    import argparse

    parser = argparse.ArgumentParser(description='VAE MI')
    parser.add_argument('--vae1', default='base_5', type=str, help='VAE 1')
    parser.add_argument('--vae2', default='cutmix_5', type=str, help='VAE 2')
    parser.add_argument('--upper', default=False, type=bool, help='if True, use upper bound, else lower')
    args = parser.parse_args()

    # Data
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    inv_normalize = transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010))
    transform_base = [transforms.ToTensor(), normalize]

    transform_test = transforms.Compose(transform_base)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    test_loader = DataLoader(test_set, 100, shuffle=True, num_workers=5)

    # MI Loss

    def kld(state):
        y_pred = state[torchbearer.Y_PRED]
        marginal = distributions.Normal(0, 1)
        return distributions.kl_divergence(y_pred, marginal).sum().div(y_pred.loc.size(0))

    # Train VAEs

    vae1 = VAE(64)
    vae1.load_state_dict(torch.load('vaes3/' + args.vae1 + '.pt')[torchbearer.MODEL])

    for param in vae1.parameters():
        param.requires_grad = False

    vae2 = VAE(64)
    vae2.load_state_dict(torch.load('vaes3/' + args.vae2 + '.pt')[torchbearer.MODEL])

    for param in vae2.parameters():
        param.requires_grad = False

    model = MINetwork(vae1, vae2, upper=args.upper)
    trial = Trial(model, criterion=kld,
                  metrics=[
                      'loss'
                  ])

    trial.with_generators(test_generator=test_loader).to('cuda').evaluate(data_key=torchbearer.TEST_DATA)
