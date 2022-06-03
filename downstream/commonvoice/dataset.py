import torch
import torchaudio
import numpy as np 
from librosa.util import find_files
from torchaudio import load
from torch import nn
from datasets import load_dataset, Audio
import librosa
import os 
import random
import pickle
import torchaudio
import sys
import time
import glob
import tqdm
import pytorch_lightning
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, random_split, DataLoader, Dataset


CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')


# Language Identification
class LanguageClassifiDataset(Dataset):
    def __init__(self, mode, file_path, meta_data, max_timestep=None):

        self.root = file_path
        self.speaker_num = 1251
        self.meta_data =meta_data
        self.max_timestep = max_timestep
        #self.usage_list = open(self.meta_data, "r").readlines()
        #languages = ["pl", "it", "ru", "zh-CN", "ar", "es", "fr", "en", "de", "fa"]
        languages = ["ru", "zh-CN", "ar"]
        sample_rate = 16000
        maxLen = sample_rate * 5
        maxCount = {"train": 1000, "validation": 150}

        print(mode)
        cache_path = os.path.join(CACHE_PATH, f'{mode}.pkl')
        if os.path.isfile(cache_path):
            print(f'Loading file paths from {cache_path}')
            with open(cache_path, 'rb') as cache:
                dataset, label = pickle.load(cache)
        else:
            dataset, label = self.generate_dataset(languages, maxLen, maxCount[mode], split=mode)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as cache:
                pickle.dump((dataset, label), cache)
        print(f'there are {len(dataset)} files found')

        self.dataset = dataset
        self.label = label
    
    def generate_dataset(self, languages, maxLen, maxCount, split='train'):
        datasets = []
        taken = False
        data = " "
        labels = " "
        procedDatasets = []
        procedLabels = []
        #for x in languages:
        #  datasets.append(load_dataset("common_voice", x, split=split, streaming = True).cast_column("audio", Audio(sampling_rate=16_000)))
        for y in range(len(languages)): 
            # to sidestep the issue of data with max length, we simply ignore all sentences that are fewer 
            # than five seconds in length.
            audio = []
            averageLength = 0
            data = load_dataset("common_voice", languages[y], split=split, streaming = True).cast_column("audio", Audio(sampling_rate=16_000))
    
    
            startCount = 0
            for x in data:
                if(x["audio"]["array"].shape[0] <= maxLen):
                    base = x["audio"]["array"]
                    averageLength += len(base)
                    audio.append(librosa.util.fix_length(base, size = maxLen, axis = 0))
                    startCount += 1
                    if(startCount == maxCount): break
            print(averageLength/ (16000 * startCount))
            concatenation = np.stack(audio, axis = 0)
            #the feature extractor accepts samples at 16k hertz
            concatenation = torch.from_numpy(concatenation)
            print(concatenation.shape)
            nulabels = torch.full((concatenation.shape[0], 1), y)
            print(nulabels)
            #datasets.append(TensorDataset(concatenation, nulabels))

            data1 = TensorDataset(concatenation, nulabels)
            
            procedDatasets.append(concatenation)
            procedLabels.append(nulabels)
    

        return ConcatDataset(procedDatasets), ConcatDataset(procedLabels)


        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], self.label[idx]
        
    def collate_fn(self, samples):        
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels


class DownstreamDataModule(LightningDataModule):
    def __init__(self, data_config, dataloader_config, **kwargs):
        super().__init__()

        self.datarc = data_config
        self.dataloaderrc = dataloader_config
        root_dir = Path(self.datarc['file_path'])
        self.train_batch_size = self.dataloaderrc.pop("train_batch_size")
        self.eval_batch_size = self.dataloaderrc.pop("eval_batch_size")
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):        
        root_dir = Path(self.datarc['file_path'])

        self.train_dataset = LanguageClassifiDataset('train', root_dir, self.datarc['meta_data'], self.datarc['max_timestep'])
        self.dev_dataset = LanguageClassifiDataset('validation', root_dir, self.datarc['meta_data'])
        # self.test_dataset = LanguageClassifiDataset('test', root_dir, self.datarc['meta_data'])

    def train_dataloader(self):

        return DataLoader(self.train_dataset, **self.dataloaderrc, batch_size=self.train_batch_size, collate_fn=self.train_dataset.collate_fn)
    
    def val_dataloader(self):
        
        return DataLoader(self.dev_dataset, **self.dataloaderrc, batch_size=self.eval_batch_size, collate_fn=self.dev_dataset.collate_fn)

    # def test_dataloader(self):
        
    #     return DataLoader(self.test_dataset, **self.dataloaderrc, batch_size=self.eval_batch_size, collate_fn=self.dev_dataset.collate_fn)
