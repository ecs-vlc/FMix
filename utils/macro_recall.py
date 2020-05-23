from torchbearer.metrics import default_for_key, to_dict, EpochLambda


@default_for_key('macro_recall')
@to_dict
class MacroRecall(EpochLambda):
    def __init__(self):
        from sklearn import metrics

        super(MacroRecall, self).__init__('macro_recall', lambda y_pred, y_true: metrics.recall_score(y_true.cpu().numpy(), y_pred.detach().cpu().numpy(), average='macro'))
