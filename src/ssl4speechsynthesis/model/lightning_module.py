from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
import torch
from transformers.modeling_outputs import MaskedLMOutput
from lightning.pytorch import LightningModule
from dac import DAC
from ssl4speechsynthesis.model.model import DACBert, DACUnitBert
from itertools import chain
import transformers
from .maksing import span_masking
import torchmetrics
import numpy as np
from dac.model.dac import Encoder

class FeatureExtractor():
    def __init__(self,cfg) -> None:
        self.dac = DAC.load(cfg.dac_path).eval()
        self.downsmaple_rate = int(np.prod(self.dac.encoder_rates))
    @torch.inference_mode()
    def __call__(self, x,n_quantizers=2):
        x = self.dac.preprocess(x, 24000)
        z = self.dac.encoder(x)
        z_q, codes, latents, commitment_loss, codebook_loss = self.dac.quantizer(z, n_quantizers)
        return z_q, codes, z
    def to(self,device:torch.device):
        self.dac.to(device)


class DACBertLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = DACBert(cfg.model.dac_bert)
        self.mask_embedding = nn.Parameter(
            torch.randn(cfg.model.dac_bert.input_size), requires_grad=True
        )
        self.feature_extractor = FeatureExtractor(cfg.model)
        print(cfg.model.dac_bert)
        if cfg.model.dac_bert.use_pretrained_encoder is False:
            self.encoder = Encoder(
                self.feature_extractor.dac.encoder_dim,
                self.feature_extractor.dac.encoder_rates,
                self.feature_extractor.dac.latent_dim
            )
        self.top_10accuracy= torchmetrics.Accuracy("multiclass",num_classes=cfg.model.dac_bert.vocab_size+1,top_k=10,ignore_index=cfg.model.dac_bert.vocab_size)
        self.top_1accuracy= torchmetrics.Accuracy("multiclass",num_classes=cfg.model.dac_bert.vocab_size+1,top_k=1,ignore_index= cfg.model.dac_bert.vocab_size)
        self.pad_idx = cfg.model.dac_bert.vocab_size
        self.cfg = cfg
        self.save_hyperparameters()

    def forward(self, x):
        output: MaskedLMOutput = self.model.forward(**x)
        return output

    def training_step(self, batch, batch_idx):
        wavs, wav_names,lengths = batch
        out_lengths = torch.tensor([length // self.feature_extractor.downsmaple_rate for length in lengths],device=self.device)
        z,codes,latents = self.feature_extractor(wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads))
        latents = latents.transpose(1, 2).clone()
        codes = codes.transpose(1, 2).clone()
        if hasattr(self,"encoder"):
            latents = self.encoder(wavs.unsqueeze(1))
        latents,span_mask = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.08,
            span_length=10,
        )
        span_mask = span_mask.to(self.device)
        attention_mask = torch.arange(latents.size(1),device=self.device).expand(latents.size(0), -1) < out_lengths.unsqueeze(1)
        codes = codes.masked_fill_(~attention_mask.unsqueeze(-1).repeat(1,1,codes.size(-1)),self.pad_idx)
        output = self.forward({"x": latents, "targets": codes,'lens':attention_mask,"mask": span_mask})
        self.log("train/loss", output.loss)
        for i in range(self.cfg.model.dac_bert.n_heads):
            masked_targets = codes.clone()
            unmasked_targets = codes.clone()
            masked_targets[span_mask] = self.pad_idx
            unmasked_targets[~span_mask] = self.pad_idx
            self.log(f"train/head{i+1}/masked/top_10accuracy", self.top_10accuracy(output.logits[:,:,:,i], masked_targets[:,:,i]),sync_dist=True)
            self.log(f"train/head{i+1}/unmasked/top_10accuracy", self.top_10accuracy(output.logits[:,:,:,i], unmasked_targets[:,:,i]),sync_dist=True)
            self.log(f"train/head{i+1}/masked/top_1accuracy", self.top_1accuracy(output.logits[:,:,:,i], masked_targets[:,:,i]),sync_dist=True)  
            self.log(f"train/head{i+1}/unmasked/top_1accuracy", self.top_1accuracy(output.logits[:,:,:,i], unmasked_targets[:,:,i]),sync_dist=True)  
        return output.loss

    def validation_step(self, batch, batch_idx):
        wavs, wav_names,lengths = batch
        out_lengths = torch.tensor([length // self.feature_extractor.downsmaple_rate for length in lengths],device=self.device)
        z,codes,latents = self.feature_extractor(wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads))
        latents = latents.transpose(1, 2).clone()
        codes = codes.transpose(1, 2).clone()
        latents,span_mask = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.08,
            span_length=10,
        )
        span_mask = span_mask.to(self.device)
        attention_mask = torch.arange(latents.size(1),device=self.device).expand(latents.size(0), -1) < out_lengths.unsqueeze(1)
        codes = codes.masked_fill_(~attention_mask.unsqueeze(-1).repeat(1,1,codes.size(-1)),self.pad_idx)
        output = self.forward({"x": latents, "targets": codes,'lens':attention_mask,"mask": span_mask})
        self.log("val/loss", output.loss,sync_dist=True)
        for i in range(self.cfg.model.dac_bert.n_heads):
            masked_targets = codes.clone()
            unmasked_targets = codes.clone()
            masked_targets[span_mask] = self.pad_idx
            unmasked_targets[~span_mask] = self.pad_idx
            self.log(f"val/head{i+1}/masked/top_10accuracy", self.top_10accuracy(output.logits[:,:,:,i], masked_targets[:,:,i]),sync_dist=True)
            self.log(f"val/head{i+1}/unmasked/top_10accuracy", self.top_10accuracy(output.logits[:,:,:,i], unmasked_targets[:,:,i]),sync_dist=True)
            self.log(f"val/head{i+1}/masked/top_1accuracy", self.top_1accuracy(output.logits[:,:,:,i], masked_targets[:,:,i]),sync_dist=True)  
            self.log(f"val/head{i+1}/unmasked/top_1accuracy", self.top_1accuracy(output.logits[:,:,:,i], unmasked_targets[:,:,i]),sync_dist=True)  
        return output.loss
    def on_fit_start(self) -> None:
        self.feature_extractor.to(self.device)
        return super().on_fit_start()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-2,eps=1e-6)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 40_000, 500_000)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

class DACBertUnitsLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = DACUnitBert(cfg.model.dac_bert)
        self.mask_embedding = nn.Parameter(
            torch.randn(cfg.model.dac_bert.input_size), requires_grad=True
        )
        self.feature_extractor = FeatureExtractor(cfg.model)
        self.top_10accuracy= torchmetrics.Accuracy("multiclass",num_classes=cfg.model.dac_bert.vocab_size+1,top_k=10,ignore_index=cfg.model.dac_bert.vocab_size)
        self.top_1accuracy= torchmetrics.Accuracy("multiclass",num_classes=cfg.model.dac_bert.vocab_size+1,top_k=1,ignore_index= cfg.model.dac_bert.vocab_size)
        self.pad_idx = cfg.model.dac_bert.vocab_size
        self.cfg = cfg
        self.save_hyperparameters()

    def forward(self, x):
        output: MaskedLMOutput = self.model.forward(**x)
        return output

    def training_step(self, batch, batch_idx):
        wavs, wav_names,lengths = batch
        out_lengths = torch.tensor([length // self.feature_extractor.downsmaple_rate for length in lengths],device=self.device)
        z,codes,latents = self.feature_extractor(wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads))
        latents = latents.transpose(1, 2).clone()
        codes = codes.transpose(1, 2).clone()
        latents,span_mask = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.08,
            span_length=10,
        )
        span_mask = span_mask.to(self.device)
        attention_mask = torch.arange(latents.size(1),device=self.device).expand(latents.size(0), -1) < out_lengths.unsqueeze(1)
        codes = codes.masked_fill_(~attention_mask.unsqueeze(-1).repeat(1,1,codes.size(-1)),self.pad_idx)
        output = self.forward({"x": latents, "targets": codes,'lens':attention_mask,"mask": span_mask})
        self.log("train/loss", output.loss)
        for i in range(self.cfg.model.dac_bert.n_heads):
            masked_targets = codes.clone()
            unmasked_targets = codes.clone()
            masked_targets[span_mask] = self.pad_idx
            unmasked_targets[~span_mask] = self.pad_idx
            self.log(f"train/head{i+1}/masked/top_10accuracy", self.top_10accuracy(output.logits[:,:,:,i], masked_targets[:,:,i]),sync_dist=True)
            self.log(f"train/head{i+1}/unmasked/top_10accuracy", self.top_10accuracy(output.logits[:,:,:,i], unmasked_targets[:,:,i]),sync_dist=True)
            self.log(f"train/head{i+1}/masked/top_1accuracy", self.top_1accuracy(output.logits[:,:,:,i], masked_targets[:,:,i]),sync_dist=True)  
            self.log(f"train/head{i+1}/unmasked/top_1accuracy", self.top_1accuracy(output.logits[:,:,:,i], unmasked_targets[:,:,i]),sync_dist=True)  
        return output.loss

    def validation_step(self, batch, batch_idx):
        wavs, wav_names,lengths = batch
        out_lengths = torch.tensor([length // self.feature_extractor.downsmaple_rate for length in lengths],device=self.device)
        z,codes,latents = self.feature_extractor(wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads))
        latents = latents.transpose(1, 2).clone()
        codes = codes.transpose(1, 2).clone()
        latents,span_mask = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.08,
            span_length=10,
        )
        span_mask = span_mask.to(self.device)
        attention_mask = torch.arange(latents.size(1),device=self.device).expand(latents.size(0), -1) < out_lengths.unsqueeze(1)
        codes = codes.masked_fill_(~attention_mask.unsqueeze(-1).repeat(1,1,codes.size(-1)),self.pad_idx)
        output = self.forward({"x": latents, "targets": codes,'lens':attention_mask,"mask": span_mask})
        self.log("val/loss", output.loss,sync_dist=True)
        for i in range(self.cfg.model.dac_bert.n_heads):
            masked_targets = codes.clone()
            unmasked_targets = codes.clone()
            masked_targets[span_mask] = self.pad_idx
            unmasked_targets[~span_mask] = self.pad_idx
            self.log(f"val/head{i+1}/masked/top_10accuracy", self.top_10accuracy(output.logits[:,:,:,i], masked_targets[:,:,i]),sync_dist=True)
            self.log(f"val/head{i+1}/unmasked/top_10accuracy", self.top_10accuracy(output.logits[:,:,:,i], unmasked_targets[:,:,i]),sync_dist=True)
            self.log(f"val/head{i+1}/masked/top_1accuracy", self.top_1accuracy(output.logits[:,:,:,i], masked_targets[:,:,i]),sync_dist=True)  
            self.log(f"val/head{i+1}/unmasked/top_1accuracy", self.top_1accuracy(output.logits[:,:,:,i], unmasked_targets[:,:,i]),sync_dist=True)  
        return output.loss
    def on_fit_start(self) -> None:
        self.feature_extractor.to(self.device)
        return super().on_fit_start()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-2,eps=1e-6)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 40_000, 500_000)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]