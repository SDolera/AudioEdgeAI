import wave
from pathlib import Path
import numpy as np

from tqdm import tqdm


def retrieve_subfolders(dataset_path, set_keyword):
    required_subfolder = ("clean", "noisy")
    subfolders_path = []

    if dataset_path.is_dir():
        for content in tqdm(dataset_path.iterdir(), desc="Retrieve subfolders path."):
            if content.is_dir():
                if (any(keyword in content.name for keyword in required_subfolder) 
                and set_keyword in content.name) :
                    subfolders_path.append(content)
    else:
        print(f"{dataset_path} is not a path to a directory.")

    return subfolders_path


def arrange_subfolders_content(subfolders_path):
    arrange_subfolder_file = {}

    for each in tqdm(subfolders_path, desc="Arranging subfolders"):
        arrange_subfolder_file[each] = []
        for file in each.iterdir():
            if file.suffix.lower() == ".wav":
                arrange_subfolder_file[each].append(file)
    
    return arrange_subfolder_file


def filter_valid_file(arrange_subfolder_file):
    clean_set = set()
    noisy_set = set()
    filtered_subfolder_file = {}

    for subfolder, filepaths in tqdm(arrange_subfolder_file.items(), desc="Assigning sets"):
        for filepath in filepaths:
            if valid_audio_specs(filepath):
                if "clean" in subfolder.name:
                    clean_set.add(filepath.name)
                else:
                    noisy_set.add(filepath.name)

    intersection_set = clean_set.intersection(noisy_set)

    for subfolder, filepaths in tqdm(arrange_subfolder_file, desc="Filtering"):
        filtered_subfolder_file[subfolder] = []      
        for filepath in filepaths:
            if filepath.name in intersection_set:
                filtered_subfolder_file[subfolder].append(filepath)

    return filtered_subfolder_file


def valid_audio_specs(filepath):
    with wave.open(str(filepath), "rb") as wav:
        return (wav.getframerate() == 16000 and wav.getsampwidth() == 2 and wav.getnchannels() == 1)
    

def equal_nframes(filtered_subfolder_file):
    clean = []
    noisy = []
    valid_frame_pair = []

    for subfolder, _ in filtered_subfolder_file.item():
        if "clean" in subfolder.name:
            clean = filtered_subfolder_file.get(subfolder)
        else:
            noisy = filtered_subfolder_file.get(subfolder)

    for clean_path in tqdm(clean, desc="Checking nframes in pair."):
        for noisy_path in noisy:
            if clean_path.name == noisy_path.name:
                with wave.open(str(clean_path), "rb") as c, wave.open(str(noisy_path), "rb") as n:
                    if c.getnframes() == n.getnframes():
                        valid_frame_pair.append((noisy_path, clean_path))

    return valid_frame_pair


def compute_rms_list_and_median(valid_frame_pair):
    valid_pair_with_rms = []
    rms_only = []

    for noisy_audio_path, clean_audio_path in tqdm(valid_frame_pair, desc="computing RMS list."):
        with wave.open(str(clean_audio_path)) as clean:
            current_audio = clean.readframes(clean.getnframes())
            samples = np.frombuffer(current_audio, '<i2', count=-1)
            converted_sample = samples.astype(np.float32)
            clean_audio_rms = np.sqrt(np.mean(np.square(converted_sample)))

            valid_pair_with_rms.append((noisy_audio_path, clean_audio_path, clean_audio_rms))
            rms_only.append(clean_audio_rms)
    rms_median = np.median(np.array(rms_only, dtype=np.float32))
    
    return valid_pair_with_rms, rms_median


def rms_scaled(valid_pair_with_rms, rms_median):
    valid_pair_with_rms_scaled = []
    for noisy, clean, rms in tqdm(valid_pair_with_rms, desc="Calculating RMS scale."):
        if rms == 0:
            scale = 1
        else:
            scale = rms_median / rms
        valid_pair_with_rms_scaled.append((noisy, clean, rms, scale))
    
    return valid_pair_with_rms_scaled


def prepare_dataset(dataset_path, set_keyword):

    subfolders = retrieve_subfolders(dataset_path, set_keyword)
    arrange_subfolders = arrange_subfolders_content(subfolders)
    filter_file_by_specs = filter_valid_file(arrange_subfolders)
    filter_file_by_frames = equal_nframes(filter_file_by_specs)
    pair_with_rms, rms_median = compute_rms_list_and_median(filter_file_by_frames)

    return rms_scaled(pair_with_rms, rms_median)