from functools import partial
import torch
from torch.utils.data.dataset import Dataset
import torchaudio
from pathlib import Path
import random
import string
from lightning.pytorch import LightningDataModule
from torch.utils.data import DistributedSampler
import math
import webdataset as wds

def my_split_by_worker(urls):
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        return urls
    else:
        return urls[wi.id::wi.num_workers]
def my_split_by_node(urls):
    node_id, node_count = torch.distributed.get_rank(), torch.distributed.get_world_size()
    return urls[node_id::node_count]

class AudioDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.cfg = hparams
        self.train_dataset = wds.WebDataset(self.cfg.train.dataset,nodesplitter=wds.split_by_node).shuffle(1000).decode(wds.torch_audio).repeat(100).with_length(50_000*self.cfg.train.batch_size)
        self.val_dataset = wds.WebDataset(self.cfg.val.dataset,nodesplitter=wds.split_by_node).decode(wds.torch_audio).repeat(100).with_length(256*self.cfg.val.batch_size)

    def train_dataloader(self):
        loader = wds.WebLoader(
            self.train_dataset,
            num_workers=self.cfg.train.num_workers,
        ).batched(
            batchsize=self.cfg.train.batch_size,
            collation_fn=partial(self.collate_fn, crops_second=self.cfg.train.crop_second),
        )
        loader.length = 50_000*self.cfg.train.batch_size
        loader= loader.with_length(50_000*self.cfg.train.batch_size)
        return loader
    def val_dataloader(self):
        loader: wds.WebLoader =  wds.WebLoader(
            self.val_dataset,
            num_workers=self.cfg.val.num_workers,
        ).batched(
            batchsize=self.cfg.val.batch_size,
            collation_fn=partial(self.collate_fn, crops_second=self.cfg.val.crop_second),
        )
        loader.length = 256*self.cfg.val.batch_size
        loader = loader.with_length(256*self.cfg.val.batch_size)
        return loader

    def collate_fn(self, batch, crops_second=None):
        wavs = []
        wav_names = []
        lengths = []
        for sample in batch:
            wav_name, (wav, sr), wav_path = sample['__key__'], sample['sample.flac'], sample['wav_path.txt']
            if sr != self.cfg.sample_rate:
                wav = torchaudio.transforms.Resample(sr, self.cfg.sample_rate)(wav)
            if crops_second is not None:
                wav = wav[:, : int(crops_second * self.cfg.sample_rate)]
            wavs.append(wav.view(-1))
            wav_names.append(wav_name)
            length = wav.view(-1).size(0)
            right_pad = math.ceil(length / self.cfg.hop_length) * self.cfg.hop_length - length
            lengths.append(length + right_pad)  
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        return wavs, wav_names,lengths




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
        with open(wav_path, "rb") as f:
            bytes = f.read()
        return str(wav_path),bytes 
            
