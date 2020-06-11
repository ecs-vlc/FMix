import torch
from torch import nn
import torchbearer
from torchbearer import metrics, Trial
from datasets.datasets import bengali
import argparse
from models import se_resnext50_32x4d

parser = argparse.ArgumentParser(description='Bengali Evaluate')

parser.add_argument('--dataset', type=str, default='bengali', help='Optional dataset path')
parser.add_argument('--dataset-path', type=str, default=None, help='Optional dataset path')
parser.add_argument('--fold-path', type=str, default='./data/folds.npz', help='Path to object storing fold ids. Run-id 0 will regen this if not existing')
parser.add_argument('--fold', type=str, default='test', help='One of [1, ..., n] or "test"')
parser.add_argument('--run-id', type=int, default=0, help='Run id')

parser.add_argument('--model-r', type=str, default=None, help='Root model')
parser.add_argument('--model-v', type=str, default=None, help='Vowel model')
parser.add_argument('--model-c', type=str, default=None, help='Consonant model')

args = parser.parse_args()

_, __, testset = bengali(args)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=6)


class BengaliModelWrapper(nn.Module):
    def __init__(self, model_r, model_v, model_c):
        super().__init__()

        self.model_r = model_r
        self.model_v = model_v
        self.model_c = model_c

    def forward(self, x):
        return self.model_r(x), self.model_v(x), self.model_c(x)


@metrics.default_for_key('grapheme')
@metrics.mean
class GraphemeAccuracy(metrics.Metric):
    def __init__(self):
        super().__init__('grapheme_acc')
        
    def process(self, *args):
        state = args[0]
        r_pred, v_pred, c_pred = state[torchbearer.PREDICTION]
        r_true, v_true, c_true = state[torchbearer.TARGET]

        _, r_pred = torch.max(r_pred, 1)
        _, v_pred = torch.max(v_pred, 1)
        _, c_pred = torch.max(c_pred, 1)

        r = (r_pred == r_true)
        v = (v_pred == v_true)
        c = (c_pred == c_true)
        return torch.stack((r, v, c), dim=1).all(1).float()


model_r = se_resnext50_32x4d(168, 1)
model_v = se_resnext50_32x4d(11, 1)
model_c = se_resnext50_32x4d(7, 1)

model_r.load_state_dict(torch.load(args.model_r, map_location='cpu')[torchbearer.MODEL])
model_v.load_state_dict(torch.load(args.model_v, map_location='cpu')[torchbearer.MODEL])
model_c.load_state_dict(torch.load(args.model_c, map_location='cpu')[torchbearer.MODEL])

model = BengaliModelWrapper(model_r, model_v, model_c)

trial = Trial(model, criterion=lambda state: None, metrics=['grapheme']).with_test_generator(testloader).to('cuda')
trial.evaluate(data_key=torchbearer.TEST_DATA)
