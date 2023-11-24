import torch
from torch.utils.data.dataset import Dataset
import torchaudio
from pathlib import Path
import random
import string
from lightning.pytorch import LightningDataModule
from torch.utils.data import DistributedSampler


class AudioDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.cfg = hparams
        self.train_dataset = GlobWavDataset(hparams.train.roots, hparams.train.patterns)
        self.val_dataset = GlobWavDataset(hparams.val.roots, hparams.val.patterns)

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset,drop_last=True)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            drop_last=True,
            persistent_workers=True,
            collate_fn=lambda batch: self.collate_fn(batch, crops_second=20),
            sampler=sampler,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.val.batch_size,
            num_workers=self.cfg.val.num_workers,
            drop_last=True,
            collate_fn=lambda batch: self.collate_fn(batch, crops_second=20),
        )

    def collate_fn(self, batch, crops_second=None):
        wavs = []
        wav_names = []
        for sample in batch:
            wav_name, (wav, sr), wav_path = sample
            if sr != self.cfg.sample_rate:
                wav = torchaudio.transforms.Resample(sr, self.cfg.sample_rate)(wav)
            if crops_second is not None:
                wav = wav[:, : int(crops_second * self.cfg.sample_rate)]
            wavs.append(wav.view(-1))
            wav_names.append(wav_name)
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        return wavs, wav_names




def generate_random_string(length):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(length))


class GlobWavDataset(Dataset):
    def __init__(
        self, roots, patterns, shuffled: bool = True, add_random_string=True
    ) -> None:
        self.wav_files = []
        for root, pattern in zip(roots, patterns):
            self.root = Path(root)
            self.wav_files.extend(list(self.root.glob(pattern)))
        if shuffled:
            random.shuffle(self.wav_files)
        self.add_random_string = add_random_string

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        if self.add_random_string:
            return (
                wav_path.stem + generate_random_string(5),
                torchaudio.load(wav_path),
                str(wav_path),
            )
        else:
            return wav_path.stem, torchaudio.load(wav_path), str(wav_path)
