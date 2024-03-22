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
import hydra
from transformers.models.hubert.modeling_hubert import HubertFeatureEncoder
from Amphion.models.codec.ns3_codec.facodec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download


class DACFeatureExtractor:
    def __init__(self, cfg) -> None:
        self.dac = DAC.load(cfg.dac_path).eval()
        self.downsample_rate = int(np.prod(self.dac.encoder_rates))

    @torch.inference_mode()
    def __call__(self, x, n_quantizers=2):
        x = self.dac.preprocess(x, 24000)
        z = self.dac.encoder(x)
        z_q, codes, latents, commitment_loss, codebook_loss = self.dac.quantizer(
            z, n_quantizers
        )
        return z_q, codes, z, x

    def to(self, device: torch.device):
        self.dac.to(device)


class FACodecFeatureExtractor:
    def __init__(self, cfg) -> None:
        self.fa_encoder = FACodecEncoder(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )
        self.fa_decoder = FACodecDecoder(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )

        encoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin"
        )
        decoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin"
        )
        self.fa_encoder.load_state_dict(torch.load(encoder_ckpt))
        self.fa_encoder.eval()
        self.fa_decoder.load_state_dict(torch.load(decoder_ckpt))
        self.fa_decoder.eval()
        self.downsample_rate = 200


    @torch.inference_mode()
    def __call__(self, x, n_quantizers=2):
        enc_out = self.fa_encoder(x)

        vq_post_emb, vq_id, _, quantized, spk_embs = self.fa_decoder(
            enc_out, eval_vq=False, vq=True
        )
        return quantized, vq_id[:n_quantizers,:,:].transpose(0,1), enc_out, x

    def to(self, device: torch.device):
        self.fa_encoder.to(device)
        self.fa_decoder.to(device)


class DACBertLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = DACBert(cfg.model.dac_bert)
        self.mask_embedding = nn.Parameter(
            torch.randn(cfg.model.dac_bert.input_size), requires_grad=True
        )
        self.feature_extractor = hydra.utils.instantiate(cfg.model.feature_extractor)
        if cfg.model.dac_bert.use_pretrained_encoder is False:
            self.encoder = HubertFeatureEncoder(cfg.model.dac_bert.encoder)
        self.top_10accuracy = torchmetrics.Accuracy(
            "multiclass",
            num_classes=cfg.model.dac_bert.vocab_size + 1,
            top_k=10,
            ignore_index=cfg.model.dac_bert.vocab_size,
        )
        self.top_1accuracy = torchmetrics.Accuracy(
            "multiclass",
            num_classes=cfg.model.dac_bert.vocab_size + 1,
            top_k=1,
            ignore_index=cfg.model.dac_bert.vocab_size,
        )
        self.pad_idx = cfg.model.dac_bert.vocab_size
        self.cfg = cfg
        self.save_hyperparameters()

    def forward(self, x):
        output: MaskedLMOutput = self.model.forward(**x)
        return output

    def training_step(self, batch, batch_idx):
        wavs, wav_names, lengths = batch
        out_lengths = torch.tensor(
            [length // self.feature_extractor.downsample_rate for length in lengths],
            device=self.device,
        )
        z, codes, latents, x = self.feature_extractor(
            wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads)
        )
        latents = latents.transpose(1, 2).clone()
        codes = codes.transpose(1, 2).clone()
        if hasattr(self, "encoder"):
            latents = self.encoder(x.clone().squeeze(1)).transpose(1, 2)
        latents, span_mask = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.08,
            span_length=10,
        )
        span_mask = span_mask.to(self.device)
        attention_mask = torch.arange(latents.size(1), device=self.device).expand(
            latents.size(0), -1
        ) < out_lengths.unsqueeze(1)
        codes = codes.masked_fill_(
            ~attention_mask.unsqueeze(-1).repeat(1, 1, codes.size(-1)), self.pad_idx
        )
        output = self.forward(
            {"x": latents, "targets": codes, "lens": attention_mask, "mask": span_mask}
        )
        self.log("train/loss", output.loss)
        for i in range(self.cfg.model.dac_bert.n_heads):
            masked_targets = codes.clone()
            unmasked_targets = codes.clone()
            masked_targets[span_mask] = self.pad_idx
            unmasked_targets[~span_mask] = self.pad_idx
            self.log(
                f"train/head{i+1}/masked/top_10accuracy",
                self.top_10accuracy(output.logits[:, :, :, i], masked_targets[:, :, i]),
                sync_dist=True,
            )
            self.log(
                f"train/head{i+1}/unmasked/top_10accuracy",
                self.top_10accuracy(
                    output.logits[:, :, :, i], unmasked_targets[:, :, i]
                ),
                sync_dist=True,
            )
            self.log(
                f"train/head{i+1}/masked/top_1accuracy",
                self.top_1accuracy(output.logits[:, :, :, i], masked_targets[:, :, i]),
                sync_dist=True,
            )
            self.log(
                f"train/head{i+1}/unmasked/top_1accuracy",
                self.top_1accuracy(
                    output.logits[:, :, :, i], unmasked_targets[:, :, i]
                ),
                sync_dist=True,
            )
        return output.loss

    def validation_step(self, batch, batch_idx):
        wavs, wav_names, lengths = batch
        out_lengths = torch.tensor(
            [length // self.feature_extractor.downsample_rate for length in lengths],
            device=self.device,
        )
        z, codes, latents, x = self.feature_extractor(
            wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads)
        )
        latents = latents.transpose(1, 2).clone()
        codes = codes.transpose(1, 2).clone()
        if hasattr(self, "encoder"):
            latents = self.encoder(x.squeeze(1)).transpose(1, 2)
        latents, span_mask = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.08,
            span_length=10,
        )
        span_mask = span_mask.to(self.device)
        attention_mask = torch.arange(latents.size(1), device=self.device).expand(
            latents.size(0), -1
        ) < out_lengths.unsqueeze(1)
        codes = codes.masked_fill_(
            ~attention_mask.unsqueeze(-1).repeat(1, 1, codes.size(-1)), self.pad_idx
        )
        output = self.forward(
            {"x": latents, "targets": codes, "lens": attention_mask, "mask": span_mask}
        )
        self.log("val/loss", output.loss, sync_dist=True)
        for i in range(self.cfg.model.dac_bert.n_heads):
            masked_targets = codes.clone()
            unmasked_targets = codes.clone()
            masked_targets[span_mask] = self.pad_idx
            unmasked_targets[~span_mask] = self.pad_idx
            self.log(
                f"val/head{i+1}/masked/top_10accuracy",
                self.top_10accuracy(output.logits[:, :, :, i], masked_targets[:, :, i]),
                sync_dist=True,
            )
            self.log(
                f"val/head{i+1}/unmasked/top_10accuracy",
                self.top_10accuracy(
                    output.logits[:, :, :, i], unmasked_targets[:, :, i]
                ),
                sync_dist=True,
            )
            self.log(
                f"val/head{i+1}/masked/top_1accuracy",
                self.top_1accuracy(output.logits[:, :, :, i], masked_targets[:, :, i]),
                sync_dist=True,
            )
            self.log(
                f"val/head{i+1}/unmasked/top_1accuracy",
                self.top_1accuracy(
                    output.logits[:, :, :, i], unmasked_targets[:, :, i]
                ),
                sync_dist=True,
            )
        return output.loss

    def on_fit_start(self) -> None:
        self.feature_extractor.to(self.device)
        return super().on_fit_start()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=2e-4, weight_decay=1e-2, eps=1e-6
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, 40_000, 500_000
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class DACBertUnitsLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = DACUnitBert(cfg.model.dac_bert)
        self.mask_embedding = nn.Parameter(
            torch.randn(cfg.model.dac_bert.input_size), requires_grad=True
        )
        self.feature_extractor = FeatureExtractor(cfg.model)
        self.top_10accuracy = torchmetrics.Accuracy(
            "multiclass",
            num_classes=cfg.model.dac_bert.vocab_size + 1,
            top_k=10,
            ignore_index=cfg.model.dac_bert.vocab_size,
        )
        self.top_1accuracy = torchmetrics.Accuracy(
            "multiclass",
            num_classes=cfg.model.dac_bert.vocab_size + 1,
            top_k=1,
            ignore_index=cfg.model.dac_bert.vocab_size,
        )
        self.pad_idx = cfg.model.dac_bert.vocab_size
        self.cfg = cfg
        self.save_hyperparameters()

    def forward(self, x):
        output: MaskedLMOutput = self.model.forward(**x)
        return output

    def training_step(self, batch, batch_idx):
        wavs, wav_names, lengths = batch
        out_lengths = torch.tensor(
            [length // self.feature_extractor.downsample_rate for length in lengths],
            device=self.device,
        )
        z, codes, latents = self.feature_extractor(
            wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads)
        )
        latents = latents.transpose(1, 2).clone()
        codes = codes.transpose(1, 2).clone()
        latents, span_mask = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.08,
            span_length=10,
        )
        span_mask = span_mask.to(self.device)
        attention_mask = torch.arange(latents.size(1), device=self.device).expand(
            latents.size(0), -1
        ) < out_lengths.unsqueeze(1)
        codes = codes.masked_fill_(
            ~attention_mask.unsqueeze(-1).repeat(1, 1, codes.size(-1)), self.pad_idx
        )
        output = self.forward(
            {"x": latents, "targets": codes, "lens": attention_mask, "mask": span_mask}
        )
        self.log("train/loss", output.loss)
        for i in range(self.cfg.model.dac_bert.n_heads):
            masked_targets = codes.clone()
            unmasked_targets = codes.clone()
            masked_targets[span_mask] = self.pad_idx
            unmasked_targets[~span_mask] = self.pad_idx
            self.log(
                f"train/head{i+1}/masked/top_10accuracy",
                self.top_10accuracy(output.logits[:, :, :, i], masked_targets[:, :, i]),
                sync_dist=True,
            )
            self.log(
                f"train/head{i+1}/unmasked/top_10accuracy",
                self.top_10accuracy(
                    output.logits[:, :, :, i], unmasked_targets[:, :, i]
                ),
                sync_dist=True,
            )
            self.log(
                f"train/head{i+1}/masked/top_1accuracy",
                self.top_1accuracy(output.logits[:, :, :, i], masked_targets[:, :, i]),
                sync_dist=True,
            )
            self.log(
                f"train/head{i+1}/unmasked/top_1accuracy",
                self.top_1accuracy(
                    output.logits[:, :, :, i], unmasked_targets[:, :, i]
                ),
                sync_dist=True,
            )
        return output.loss

    def validation_step(self, batch, batch_idx):
        wavs, wav_names, lengths = batch
        out_lengths = torch.tensor(
            [length // self.feature_extractor.downsample_rate for length in lengths],
            device=self.device,
        )
        z, codes, latents = self.feature_extractor(
            wavs.unsqueeze(1), n_quantizers=len(self.model.lm_heads)
        )
        latents = latents.transpose(1, 2).clone()
        codes = codes.transpose(1, 2).clone()
        latents, span_mask = span_masking(
            latents,
            mask_embedding=self.mask_embedding,
            mask_probability=0.08,
            span_length=10,
        )
        span_mask = span_mask.to(self.device)
        attention_mask = torch.arange(latents.size(1), device=self.device).expand(
            latents.size(0), -1
        ) < out_lengths.unsqueeze(1)
        codes = codes.masked_fill_(
            ~attention_mask.unsqueeze(-1).repeat(1, 1, codes.size(-1)), self.pad_idx
        )
        output = self.forward(
            {"x": latents, "targets": codes, "lens": attention_mask, "mask": span_mask}
        )
        self.log("val/loss", output.loss, sync_dist=True)
        for i in range(self.cfg.model.dac_bert.n_heads):
            masked_targets = codes.clone()
            unmasked_targets = codes.clone()
            masked_targets[span_mask] = self.pad_idx
            unmasked_targets[~span_mask] = self.pad_idx
            self.log(
                f"val/head{i+1}/masked/top_10accuracy",
                self.top_10accuracy(output.logits[:, :, :, i], masked_targets[:, :, i]),
                sync_dist=True,
            )
            self.log(
                f"val/head{i+1}/unmasked/top_10accuracy",
                self.top_10accuracy(
                    output.logits[:, :, :, i], unmasked_targets[:, :, i]
                ),
                sync_dist=True,
            )
            self.log(
                f"val/head{i+1}/masked/top_1accuracy",
                self.top_1accuracy(output.logits[:, :, :, i], masked_targets[:, :, i]),
                sync_dist=True,
            )
            self.log(
                f"val/head{i+1}/unmasked/top_1accuracy",
                self.top_1accuracy(
                    output.logits[:, :, :, i], unmasked_targets[:, :, i]
                ),
                sync_dist=True,
            )
        return output.loss

    def on_fit_start(self) -> None:
        self.feature_extractor.to(self.device)
        return super().on_fit_start()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=2e-4, weight_decay=1e-2, eps=1e-6
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, 40_000, 500_000
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
