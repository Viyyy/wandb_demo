import os

import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.functional as F

class AudioDataset(Dataset):
    def __init__(self, X, y, data_dir, sr:int, duration:float, device, transform=None):
        '''
        :param X: list of audio paths
        :param y: list of labels
        :param data_dir: directory where audio files are stored
        :param sr: sampling rate
        :param duration: duration of each audio clip in seconds
        :param device: device to load audio on
        :param transform: optional transform to be applied on a sample
        '''
        assert len(X) == len(y)
        super().__init__()
        self.X = X
        self.y = y
        self.data_dir = data_dir
        self.sr = sr
        self.duration = duration
        self.target_samples = int(self.sr * self.duration)
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __get_audio_path__(self, index):
        audio_path = os.path.join(self.data_dir, self.X[index])
        return audio_path

    def __get_label__(self, index):
        return self.y[index]

    def __getitem__(self, index):
        signal, sr = torchaudio.load(self.__get_audio_path__(index))
        signal = signal.to(self.device)
        # resample to desired sampling rate
        signal = F.resample(signal,orig_freq=sr, new_freq=self.sr)
        # mix down to mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        # cut/pad audio to desired duration
        if signal.shape[1]>self.target_samples:
            signal = signal[:,:self.target_samples]
        else:
            paddings = self.target_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, paddings)) # pad with zeros
        if self.transform:
            feature = self.transform(signal)
        else:
            feature = signal
        label = self.__get_label__(index)
        return feature, label, index