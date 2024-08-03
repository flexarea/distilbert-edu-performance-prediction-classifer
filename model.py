import torch
import torch.nn as nn
from transformers import AutoModel


class PerformancepredictionModel (nn.Module):
    def __init__(self, freeze_bert: bool = False):
        super(PerformancepredictionModel, self).__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.bert.config.hidden_size, 1)])

        self.freeze_bert = freeze_bert

        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        a = cls_embedding.unsqueeze(0)
        for i, layer in enumerate(self.linear_layers):
            z = layer(a)
            a = torch.sigmoid(z)
        return a
