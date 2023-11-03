from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
import torch
from transformers.modeling_outputs import MaskedLMOutput
from lightning.pytorch import LightningModule
from dac import DAC
from ssl4speechsynthesis.model.model import DACBert
from itertools import chain
from .maksing import span_masking


class DACBertLightningModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.dac = DAC.load(hparams.dac_path).eval()
        self.model = DACBert(hparams.dac_bert)
        self.mask_embedding = nn.Parameter(
            torch.randn(hparams.dac_bert.input_size), requires_grad=True
        )
        self.save_hyperparameters()

    def forward(self, x):
        output: MaskedLMOutput = self.model.forward(**x)
        return output

    def training_step(self, batch, batch_idx):
        wavs, wav_names = batch
        with torch.inference_mode():
            self.dac.eval()
            z, codes, latents, _, _ = self.dac.encode(
                wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads)
            )
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
        z, codes, latents, _, _ = self.dac.encode(
            wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads)
        )
        latents = latents.transpose(1, 2)
        codes = codes.transpose(1, 2)
        latents = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.08,
            span_length=10,
        )
        output = self.forward({"x": latents, "targets": codes.clone()})
        self.log("val/loss", output.loss)
        return output.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)
