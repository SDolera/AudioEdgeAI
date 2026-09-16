import torch
from torch.utils.data import Dataset


import random
import wave
import numpy as np



def random_segment_startpoint(n_frames):

    one_segment = 16000
    max_allowed_frame = n_frames  - one_segment
    return random.randint(0, max_allowed_frame)


def get_one_segment(noisy, clean):

    with wave.open(str(noisy), "rb") as n, wave.open(str(clean), "rb") as c:
        startpoint = random_segment_startpoint(n.getnframes())
        n.setpos(startpoint)
        noisy_frames = n.readframes(16000)
        c.setpos(startpoint)
        clean_frames = c.readframes(16000)

    noisy_segment, clean_segment = pcm_to_float32(noisy_frames, clean_frames)

    return noisy_segment, clean_segment


def pcm_to_float32(noisy_frames, clean_frames):

    noisy_int16 = np.frombuffer(noisy_frames, '<i2', count=-1)
    noisy_float32 = noisy_int16.astype(np.float32)

    clean_int16 = np.frombuffer(clean_frames, '<i2', count=-1)
    clean_float32 = clean_int16.astype(np.float32)

    return noisy_float32, clean_float32


def apply_scale(noisy_segment, clean_segment, scale):

    scaled_noisy = noisy_segment * scale
    scaled_clean = clean_segment * scale

    return scaled_noisy, scaled_clean


class custom_denoiser_dataset(Dataset):

    def __init__(self, valid_pair_rms_scale):
        self.audio_samples = valid_pair_rms_scale

    def __len__(self):

        return len(self.audio_samples)
    
    def __getitem__(self, index):

        noisy_path, clean_path, _, scale = self.audio_samples[index]
        noisy_segment, clean_segment = get_one_segment(noisy_path, clean_path)
        scaled_noisy, scaled_clean = apply_scale(noisy_segment, clean_segment, scale)

        noisy_tensor = torch.from_numpy(scaled_noisy)
        clean_tensor = torch.from_numpy(scaled_clean)

        final_noisy_tensor = noisy_tensor.unsqueeze(0)
        final_clean_tensor = clean_tensor.unsqueeze(0)

        return final_noisy_tensor, final_clean_tensor