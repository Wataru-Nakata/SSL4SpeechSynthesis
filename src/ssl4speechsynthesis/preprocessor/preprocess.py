from typing import Any
from tqdm import tqdm
import torch
import webdataset as wds
import itertools
from multiprocessing.managers import SyncManager
from multiprocessing import current_process, Pool
import json
from pathlib import Path
import random
from functools import partial


class MyShardWriter(wds.ShardWriter):
    def __init__(self, pattern, maxcount=100000, maxsize=3e9, post=None, start_shard=0, **kw):
        super().__init__(pattern, maxcount, maxsize, post, start_shard)
        self.verbose = False

    def get_shards(self):
        return self.shard

    def get_count(self):
        return self.count if self.count < self.maxcount else 0

    def get_total(self):
        return self.total
class MyManager(SyncManager):
    pass

def worker(file_path,lock,pbar,sink):
    worker_id = int(current_process().name.split('-')[-1])
    wav_path = file_path
    key = Path(wav_path).stem
    with open(wav_path,"rb") as f:
        bytes = f.read()
    
    with lock:
        sample = {
            "__key__": key,
            "sample.flac": bytes,
            "wav_path.txt":str(wav_path),
            "json": json.dumps(
                {
                    "key":current_process().name,
                    "count": sink.get_count(),
                }),
        }
        sink.write(sample)
        pbar.update(1)
        pbar.set_postfix_str(
            f'shard {sink.get_shards()} '
            f'worker {worker_id}'
        )
    


class Preprocessor():
    def __init__(self, cfg,split="train"):
        self.cfg = cfg
        if split=="train":
            dataset = cfg.train.dataset
            self.pattern = cfg.train.save_path
        elif split=="valid":
            dataset = cfg.valid.dataset
            self.pattern = cfg.valid.save_path
        else:
            raise NotImplementedError
        Path(self.pattern).parent.mkdir(parents=True,exist_ok=True)
        self.dataset_paths = list(itertools.chain.from_iterable([ list(Path(root).glob(pattern)) for root, pattern in zip(dataset.roots,dataset.patterns)]))
        self.split = split
    
    def preprocess(self):
        random.shuffle(self.dataset_paths)
        n_samples = len(self.dataset_paths)
        MyManager.register('Tqdm', tqdm)
        MyManager.register('Sink', MyShardWriter)
        with MyManager() as manager:
            lock = manager.Lock()
            sink = manager.Sink(
                pattern=self.pattern,
                maxsize = self.cfg.maxsize,
                maxcount = self.cfg.maxcount,
            )
            pbar = manager.Tqdm(
                total=n_samples,
                desc='Main process',
            )
            worker_with_args = partial(
                worker,lock=lock,pbar=pbar,sink=sink
            )
            with Pool(
                self.cfg.num_workers,
            ) as pool:
                pool.map(
                    worker_with_args,
                    self.dataset_paths,
                    chunksize= n_samples // self.cfg.num_workers,
                )
            pbar.close()
            sink.close()
            print('\nPreprocessing finished')
