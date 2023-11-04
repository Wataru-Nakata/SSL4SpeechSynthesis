from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
import torch
from transformers.modeling_outputs import MaskedLMOutput
from lightning.pytorch import LightningModule
from dac import DAC
from ssl4speechsynthesis.model.model import DACBert
from itertools import chain
import transformers
from .maksing import span_masking

class FeatureExtractor():
    def __init__(self,hparams) -> None:
        self.dac = DAC.load(hparams.dac_path).eval()
    @torch.inference_mode()
    def __call__(self, x,n_quantizers=2):
        z, codes, latents, _, _ = self.dac.encode(
            x, n_quantizers=n_quantizers
        )
        return z, codes, latents
    def to(self,device:torch.device):
        self.dac.to(device)


class DACBertLightningModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = DACBert(hparams.dac_bert)
        self.mask_embedding = nn.Parameter(
            torch.randn(hparams.dac_bert.input_size), requires_grad=True
        )
        self.feature_extractor = FeatureExtractor(hparams)
        self.save_hyperparameters()

    def forward(self, x):
        output: MaskedLMOutput = self.model.forward(**x)
        return output

    def training_step(self, batch, batch_idx):
        wavs, wav_names = batch
        z,codes,latents = self.feature_extractor(wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads))
        latents = latents.transpose(1, 2)
        codes = codes.transpose(1, 2)
        latents = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.08,
            span_length=10,
        )
        output = self.forward({"x": latents, "targets": codes.clone()})
        self.log("train/loss", output.loss)
        return output.loss

    def validation_step(self, batch, batch_idx):
        wavs, wav_names = batch
        z,codes,latents = self.feature_extractor(wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads))
        latents = latents.transpose(1, 2)
        codes = codes.transpose(1, 2)
        latents = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.08,
            span_length=10,
        )
        output = self.forward({"x": latents, "targets": codes.clone()})
        self.log("val/loss", output.loss,sync_dist=True)
        return output.loss
    def on_fit_start(self) -> None:
        self.feature_extractor.to(self.device)
        return super().on_fit_start()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = transformers.get_constant_schedule_with_warmup(optimizer, 10000)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
