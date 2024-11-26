import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torchaudio

np.random.seed(123)

PAD_INDEX = 0
EPS = 1e-10
SAMPLING_RATE = 16000


def is_nan(t):
    inf = torch.isinf(t).any()
    nan = torch.isnan(t).any()
    if inf or nan:
        return True
    return False


def pad_noise(speech, noise, is_random_noise_cut=False):
    """
    Cuts noise vector if speech vec is shorter
    Adds noise if speech vector is longer
    """
    noise_len = noise.shape[1]
    speech_len = speech.shape[1]

    if speech_len > noise_len:
        repeat = (speech_len // noise_len) + 1
        noise = torch.tile(noise, (1, repeat))
        diff = speech_len - noise.shape[1]
        noise = noise[:, : noise.shape[1] + diff]

    elif speech_len < noise_len:
        if is_random_noise_cut:
            start_range = noise_len - speech_len
            start_idx = np.random.randint(0, start_range)
            noise = noise[:, start_idx : start_idx + speech_len]
        else:
            noise = noise[:, :speech_len]
    return noise


def mix_signals(speech, noise, desired_snr):
    # calculate energies
    energy_s = torch.sum(speech**2, dim=-1, keepdim=True)
    energy_n = torch.sum(noise**2, dim=-1, keepdim=True)

    b = torch.sqrt((energy_s / energy_n) * (10 ** (-desired_snr / 10.0)))
    return speech + b * noise


class SamplerFixNoise:
    def __init__(self, csv_path, musan_data_path, spk_noise_set_path, mode="train", is_random_noise_cut=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_random_noise_cut = is_random_noise_cut
        self.noise_path = musan_data_path
        self.noise_dict = self._load_soundbible_noise()

        self.data = pd.read_csv(csv_path).astype({"spk": "str"})  # change speaker dtype to str

        # Load spk noise set
        with open(spk_noise_set_path, "r") as f:
            self.spk_noise_set = json.load(f)

    def _load_soundbible_noise(self):
        noise_dict = {}
        noise_files = os.listdir(self.noise_path)
        for noise_file in noise_files:
            if "ANNOTATIONS" in noise_file:
                continue
            if "LICENSE" in noise_file:
                continue
            noise, fs = torchaudio.load(os.path.join(self.noise_path, noise_file))
            if fs != SAMPLING_RATE:
                noise = torchaudio.functional.resample(noise, fs, SAMPLING_RATE)
            noise_dict[os.path.basename(noise_file)] = noise
        return noise_dict

    def _pad_source(self, source, total_len):
        repeat = (total_len // source.shape[1]) + 1
        source = torch.tile(source, (1, repeat))
        source = source[:, :total_len]
        return source

    def get_data_len(self, spk_id, mode):
        return len(self.data[(self.data["spk"] == str(spk_id)) & (self.data["split"] == mode)])

    def sample_batch(self, spk_id, batch_size, mode="train", mix_snr=None):
        if mix_snr is not None:
            assert mode != "train" and mode != "val", "mix_snr is only for test mode"
        np.random.seed(42)

        data = self.data
        files = data[(data["spk"] == str(spk_id)) & (data["split"] == mode)]
        noise_files = self.spk_noise_set[str(spk_id)]

        sec = 4
        fs = 16000

        total_len = fs * sec

        mixtures = []
        targets = []

        noise_cnt = [0] * len(noise_files)
        while batch_size:
            # Load sources
            idx = random.randint(0, len(files) - 1)
            source = files.iloc[idx]["file"]
            source, fs = torchaudio.load(source)
            if fs != SAMPLING_RATE:
                source = torchaudio.functional.resample(source, fs, SAMPLING_RATE)
            if source.shape[1] < total_len:
                source = self._pad_source(source, total_len)
            elif source.shape[1] > total_len:
                start_range = source.shape[1] - SAMPLING_RATE * sec
                start_idx = random.randint(0, start_range - 1)
                source = source[:, start_idx : start_idx + total_len]

            # Load noise
            if mode != "test":
                # Randomly choose noise
                idx = random.randint(0, len(noise_files) - 1)
                noise_cnt[idx] += 1
                noise, fs = torchaudio.load(os.path.join(self.noise_path, noise_files[idx]))
            else:
                # Fix noise in test mode
                noise, fs = torchaudio.load(os.path.join(self.noise_path, noise_files[self.curr_noise_idx]))
                self.curr_noise_idx += 1
            if fs != SAMPLING_RATE:
                noise = torchaudio.functional.resample(noise, fs, SAMPLING_RATE)

            source = source.to(self.device)
            noise = noise.to(self.device)
            noise = pad_noise(source, noise, self.is_random_noise_cut)

            # Mix noise
            if mix_snr is None:
                SNR = random.randrange(-5, 5)
            else:
                SNR = mix_snr
            mixture = mix_signals(source, noise, SNR)
            assert (
                mixture.shape[1] == total_len
            ), f"Mixture dim does not match. Mixture {mixture.shape}, \
                required size {total_len}"
            mixtures.append(mixture)
            targets.append(source)
            batch_size -= 1

        return {
            "x": torch.stack(mixtures, dim=1).squeeze(0),
            "t": torch.stack(targets, dim=1).squeeze(0),
        }
