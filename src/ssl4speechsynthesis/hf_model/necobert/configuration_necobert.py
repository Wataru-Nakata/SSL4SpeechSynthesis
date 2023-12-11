"""NecoBERT model configuration"""

from transformers.configuration_utils import PretrainedConfig
from typing import List

class NecoBertConfig(PretrainedConfig):
    model_type = "necobert"
    def __init__(
        self, 
        input_size: int = 2048,
        vocab_size: int = 1024,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        n_heads: int = 1,
        dac_encoder_dim: int = 64,
        dac_encoder_rates: List[int] = [2,3,4,4,5],
        dac_latent_dim: int = None,
        dac_decoder_dim: int = 1536,
        dac_decoder_rates: List[int] = [5,4,4,3,2],
        dac_n_codebooks: int = 9,
        dac_codebook_size: int =1024,
        dac_quantizer_dropout: float = 1.0,
        dac_codebook_dim: int = 8,
        dac_sample_rate: int = 24000,
        **kwargs,
    ):
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob  
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.n_heads = n_heads
        self.dac_encoder_dim = dac_encoder_dim
        self.dac_encoder_rates = dac_encoder_rates
        self.dac_latent_dim = dac_latent_dim
        self.dac_decoder_dim = dac_decoder_dim
        self.dac_decoder_rates = dac_decoder_rates
        self.dac_n_codebooks = dac_n_codebooks
        self.dac_codebook_size = dac_codebook_size
        self.dac_quantizer_dropout = dac_quantizer_dropout
        self.dac_codebook_dim = dac_codebook_dim
        self.dac_sample_rate = dac_sample_rate
        super().__init__(**kwargs)