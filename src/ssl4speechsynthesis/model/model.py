from transformers import BertModel, BertConfig
from transformers.modeling_outputs import MaskedLMOutput
from torch import nn
import torch


class DACBert(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        config = BertConfig(**hparams.bert_config)
        print(config.vocab_size)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.lm_heads = nn.ModuleList()
        self.input_linear = nn.Linear(hparams.input_size, config.hidden_size)
        self.loss = nn.CrossEntropyLoss()
        for i in range(hparams.n_heads):
            self.lm_heads.append(nn.Linear(config.hidden_size, hparams.vocab_size))

    def forward(self, x, targets=None):
        x = self.input_linear(x)
        x = self.bert(inputs_embeds=x).last_hidden_state
        lm_heads_outputs = []
        for lm_head in self.lm_heads:
            lm_heads_outputs.append(lm_head(x))
        if targets is None:
            loss = None
        else:
            lm_heads_outputs = torch.stack(lm_heads_outputs, dim=2)
            loss = self.loss(lm_heads_outputs.permute(0, 3, 1, 2), targets)

        output = MaskedLMOutput(loss=loss, logits=lm_heads_outputs)
        return output
