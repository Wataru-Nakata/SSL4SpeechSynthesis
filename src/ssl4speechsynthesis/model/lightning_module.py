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
import torchmetrics

class FeatureExtractor():
    def __init__(self,cfg) -> None:
        self.dac = DAC.load(cfg.dac_path).eval()
    @torch.inference_mode()
    def __call__(self, x,n_quantizers=2):
        z, codes, latents, _, _ = self.dac.encode(
            x, n_quantizers=n_quantizers
        )
        return z, codes, latents
    def to(self,device:torch.device):
        self.dac.to(device)


class DACBertLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = DACBert(cfg.dac_bert)
        self.mask_embedding = nn.Parameter(
            torch.randn(cfg.dac_bert.input_size), requires_grad=True
        )
        self.feature_extractor = FeatureExtractor(cfg)
        self.top_10accuracy= torchmetrics.Accuracy("multiclass",num_classes=cfg.dac_bert.vocab_size,top_k=10)
        self.top_1accuracy= torchmetrics.Accuracy("multiclass",num_classes=cfg.dac_bert.vocab_size,top_k=1)
        self.cfg = cfg
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
            mask_probability=0.16,
            span_length=10,
        )
        output = self.forward({"x": latents, "targets": codes.clone()})
        self.log("train/loss", output.loss)
        for i in range(self.cfg.dac_bert.n_heads):
            self.log(f"train/head{i+1}/top_10accuracy", self.top_10accuracy(output.logits[:,:,i,:].permute(0,2,1), codes[:,:,i]),sync_dist=True)
            self.log(f"train/head{i+1}/top_1accuracy", self.top_1accuracy(output.logits[:,:,i,:].permute(0,2,1), codes[:,:,i]),sync_dist=True)  
        return output.loss

    def validation_step(self, batch, batch_idx):
        wavs, wav_names = batch
        z,codes,latents = self.feature_extractor(wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads))
        latents = latents.transpose(1, 2)
        codes = codes.transpose(1, 2)
        latents = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.16,
            span_length=10,
        )
        output = self.forward({"x": latents, "targets": codes.clone()})
        self.log("val/loss", output.loss,sync_dist=True)
        for i in range(self.cfg.dac_bert.n_heads):
            self.log(f"val/head{i+1}/top_10accuracy", self.top_10accuracy(output.logits[:,:,i,:].permute(0,2,1), codes[:,:,i]),sync_dist=True)
            self.log(f"val/head{i+1}/top_1accuracy", self.top_1accuracy(output.logits[:,:,i,:].permute(0,2,1), codes[:,:,i]),sync_dist=True)  
        return output.loss
    def on_fit_start(self) -> None:
        self.feature_extractor.to(self.device)
        return super().on_fit_start()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 40_000, 500_000)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
