import torch
from torch import nn


class Bert(nn.Module):
    def __init__(self, num_classes, nc=None, bert_model='bert-base-cased'):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(bert_model, output_hidden_states=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, x, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(x.long(),
                            attention_mask=(x != 0).float(),
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        cls_output = torch.cat(outputs[2], dim=1).mean(dim=1)
        cls_output = self.classifier(cls_output)  # batch, 6
        return cls_output
