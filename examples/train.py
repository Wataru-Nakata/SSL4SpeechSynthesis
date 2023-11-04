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
    callbacks = [LearningRateMonitor(logging_interval="step")]
    loggers = WandbLogger(project="ssl4speechsynthesis", log_model=True)
    trainer = hydra.utils.instantiate(
        cfg.train.trainer, logger=loggers, callbacks=callbacks
    )
    lightning_module = DACBertLightningModule(cfg.model)
    datamodule = AudioDataModule(cfg.data)
    trainer.fit(lightning_module, datamodule, ckpt_path=cfg.train.ckpt_path)


if __name__ == "__main__":
    main()
