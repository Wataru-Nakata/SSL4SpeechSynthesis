from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import seed_everything
from ssl4speechsynthesis.model.lightning_module import DACBertLightningModule
from ssl4speechsynthesis.data.datamodule import AudioDataModule
from lightning.pytorch.loggers import WandbLogger


@hydra.main(version_base="1.3", config_name="config", config_path="config")
def main(cfg: DictConfig):
    seed_everything(1234)
    torch.set_float32_matmul_precision('medium')
    callbacks = [LearningRateMonitor(logging_interval="step")]
    loggers = WandbLogger(project="ssl4speechsynthesis", log_model=True)
    trainer = hydra.utils.instantiate(
        cfg.train.trainer, logger=loggers, callbacks=callbacks
    )
    lightning_module = hydra.utils.instantiate(cfg.model.lightning_module,cfg=cfg)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule,hparams=cfg.data)
    trainer.fit(lightning_module, datamodule, ckpt_path=cfg.train.ckpt_path)
if __name__ == "__main__":
    main()
