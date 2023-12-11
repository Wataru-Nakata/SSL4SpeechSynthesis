from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import MaskedLMOutput
from .configuration_necobert import NecoBertConfig
from .model import DACBert
from omegaconf import DictConfig
from dac import DAC
from torch import nn

class NecoBertModel(PreTrainedModel):
    config_class = NecoBertConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        model_config = {
            "input_size": config.input_size,
            "vocab_size": config.vocab_size,
            "bert_config": {
                "hidden_size": config.hidden_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "intermediate_size": config.intermediate_size,
                "hidden_act": config.hidden_act,
                "hidden_dropout_prob": config.hidden_dropout_prob,
                "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
                "max_position_embeddings": config.max_position_embeddings,
                "initializer_range": config.initializer_range,
                "layer_norm_eps": config.layer_norm_eps,
            },
            "n_heads": config.n_heads,
        }
        self.model = DACBert(DictConfig(model_config))
        self.preprocessor = NecoBertTokenizer(config)
    def forward(self, inputs,sample_rate=None):
        if sample_rate == None:
            sample_rate = self.config.dac_sample_rate
        z_q, codes, z = self.preprocessor(inputs['x'], sample_rate)
        z = z.transpose(1, 2)
        codes = codes.transpose(1, 2)
        output = self.model.forward(**{"x": z, "targets": codes.clone()}) 
        return output
class NecoBertTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dac = DAC(
            encoder_dim=config.dac_encoder_dim,
            encoder_rates=config.dac_encoder_rates,
            decoder_dim=config.dac_decoder_dim,
            latent_dim=config.dac_latent_dim,
            decoder_rates=config.dac_decoder_rates,
            n_codebooks=config.dac_n_codebooks,
            codebook_size=config.dac_codebook_size,
            codebook_dim=config.dac_codebook_dim,
            quantizer_dropout=config.dac_quantizer_dropout,
            sample_rate=config.dac_sample_rate
        )
    def forward(self, x,sample_rate):
        if sample_rate == None:
            sample_rate = self.config.dac_sample_rate
        x = self.dac.preprocess(x, sample_rate)
        z = self.dac.encoder(x)
        z_q, codes, latents, commitment_loss, codebook_loss = self.dac.quantizer(z, self.config.n_heads)
        return z_q, codes, z