#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as transforms
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Define augmentations using audiomentations library
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),  # Add Gaussian noise
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),  # Stretch or compress time
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),  # Shift pitch
])

# Define paths for data and output
data_path = 'UrbanSound8K/audio'
metadata_path = 'UrbanSound8K/metadata/UrbanSound8K.csv'
output_path = 'preprocessed_data'
fixed_size = (64, 512)  # Fixed size for spectrograms

# Create output directories for each fold if they don't exist
for fold in range(1, 11):
    fold_path = os.path.join(output_path, f'fold{fold}')
    os.makedirs(fold_path, exist_ok=True)

# Load metadata from the UrbanSound8K dataset
metadata = pd.read_csv(metadata_path)

def preprocess_audio(file_path, n_mels=64, n_fft=2048, hop_length=512, augmentations=None):
    """
    Preprocess an audio file to create a mel spectrogram.

    Args:
    - file_path (str): Path to the audio file.
    - n_mels (int): Number of mel bands to generate.
    - n_fft (int): FFT window size.
    - hop_length (int): Number of samples between successive frames.
    - augmentations (Compose): Augmentations to apply to the waveform.

    Returns:
    - np.array: Mel spectrogram in dB.
    """
    waveform, sample_rate = torchaudio.load(file_path, normalize=True)
    
    # Convert to mono if the audio is stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Apply augmentations if provided
    if augmentations:
        waveform = torch.tensor(augmentations(samples=waveform.numpy()[0], sample_rate=sample_rate)).unsqueeze(0)
    
    # Create mel spectrogram
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mel_spectrogram = mel_spectrogram_transform(waveform)
    
    # Convert to decibel scale
    mel_spectrogram_db_transform = T.AmplitudeToDB()
    mel_spectrogram_db = mel_spectrogram_db_transform(mel_spectrogram)
    
    return mel_spectrogram_db.squeeze().numpy()

def pad_spectrogram(spectrogram, size=fixed_size):
    """
    Pad the spectrogram to a fixed size.

    Args:
    - spectrogram (np.array): Input spectrogram.
    - size (tuple): Target size (height, width).

    Returns:
    - np.array: Padded or trimmed spectrogram.
    """
    # Ensure the spectrogram is 2D
    if spectrogram.ndim == 3:
        spectrogram = spectrogram[0]  # Remove the channel dimension if present

    if spectrogram.ndim != 2:
        raise ValueError(f"Expected a 2D spectrogram, but got {spectrogram.ndim}D array")

    target_height, target_width = size
    height, width = spectrogram.shape

    # Pad or trim width
    if width < target_width:
        pad_width = target_width - width
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spectrogram = spectrogram[:, :target_width]

    # Pad or trim height
    if height < target_height:
        pad_height = target_height - height
        spectrogram = np.pad(spectrogram, ((0, pad_height), (0, 0)), mode='constant')
    else:
        spectrogram = spectrogram[:target_height, :]
    
    return spectrogram

def spectrogram_to_tensor(spectrogram):
    """
    Convert a mel spectrogram to a tensor.

    Args:
    - spectrogram (np.array): Input spectrogram.

    Returns:
    - Tensor: Spectrogram as a PyTorch tensor.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor = transform(spectrogram)
    
    return tensor

def process_and_save_data(metadata, data_path, output_path, augmentations, spectrogram_file_name, augmented_spectrogram_file_name):
    """
    Process and save audio files as mel spectrogram tensors.

    Args:
    - metadata (DataFrame): Metadata for the dataset.
    - data_path (str): Path to the audio files.
    - output_path (str): Path to save the processed data.
    - augmentations (Compose): Augmentations to apply.
    - spectrogram_file_name (str): Filename for non-augmented spectrograms.
    - augmented_spectrogram_file_name (str): Filename for augmented spectrograms.
    """
    for fold in range(1, 11):
        spectrograms = []
        labels = []
        augmented_spectrograms = []
        
        fold_metadata = metadata[metadata['fold'] == fold]
        for index, row in fold_metadata.iterrows():
            file_name = row['slice_file_name']
            file_path = os.path.join(data_path, f'fold{fold}', file_name)
            
            # Process audio file without augmentation
            spectrogram = preprocess_audio(file_path, augmentations=None)
            spectrogram = pad_spectrogram(spectrogram)
            tensor = spectrogram_to_tensor(spectrogram)
            
            # Append tensor and label to lists
            spectrograms.append(tensor)

            # Process audio file with augmentation
            augmented_spectrogram = preprocess_audio(file_path, augmentations=augmentations)
            augmented_spectrogram = pad_spectrogram(augmented_spectrogram)
            augmented_tensor = spectrogram_to_tensor(augmented_spectrogram)
            
            # Append augmented tensor and label to lists
            augmented_spectrograms.append(augmented_tensor)

            labels.append(row['classID'])
            
            if index % 100 == 0:
                print(f'Processed {index + 1}/{len(fold_metadata)} files in fold {fold}')
        
        # Save lists as .pt files
        torch.save(spectrograms, os.path.join(output_path, f'fold{fold}', spectrogram_file_name))
        torch.save(labels, os.path.join(output_path, f'fold{fold}', 'labels.pt'))
        torch.save(augmented_spectrograms, os.path.join(output_path, f'fold{fold}', augmented_spectrogram_file_name))

if __name__ == "__main__":
    # Define filenames for saving the spectrograms
    spectrogram_file_name = 'spectrograms.pt'
    augmented_spectrogram_file_name = 'spectrograms_augmented.pt'
    
    # Process and save the data
    process_and_save_data(metadata, data_path, output_path, augment, spectrogram_file_name, augmented_spectrogram_file_name)
    print('Data preprocessing completed successfully.')


# In[ ]:




