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
        del self.bert.embeddings.word_embeddings
        self.lm_heads = nn.ModuleList()
        self.input_linear = nn.Linear(hparams.input_size, config.hidden_size)
        self.loss = nn.CrossEntropyLoss()
        self.alpha = 1
        for i in range(hparams.n_heads):
            self.lm_heads.append(nn.Linear(config.hidden_size, hparams.vocab_size))

    def forward(self, x, targets=None,mask=None):
        x = self.input_linear(x)
        bert_output = self.bert(inputs_embeds=x, output_hidden_states=True)
        last_hidden_state = bert_output.last_hidden_state
        lm_heads_outputs = []
        for lm_head in self.lm_heads:
            lm_heads_outputs.append(lm_head(last_hidden_state))
        if targets is None:
            loss = None
        else:
            if mask ==None:
                lm_heads_outputs = torch.stack(lm_heads_outputs, dim=2).permute(0, 3, 1, 2)
                loss = self.loss(lm_heads_outputs, targets)
            else:
                outputs = torch.stack(lm_heads_outputs, dim=2)
                print(outputs.size(),torch.masked_select(outputs,mask).size(),mask.size())
                masked = outputs[~mask].permute(0,3,1,2)
                non_masked = outputs[mask].permute(0,3,1,2)
                non_masked_loss = self.loss(masked,targets[mask])
                masked_loss = self.loss(non_masked,targets[~mask])
                loss = alpha * masked_loss + (1-self.alpha) * non_masked_loss
        output = MaskedLMOutput(loss=loss, logits=lm_heads_outputs, hidden_states=bert_output.hidden_states)
        return output
