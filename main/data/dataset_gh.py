import torch
import os
import json
import pandas as pd
import random
import glob
import torchvision.transforms as transforms
from natsort import natsorted
from tqdm import tqdm
import torchaudio
import pickle

import librosa

from PIL import Image

from stable_audio_tools.data.utils import Stereo, Mono, PhaseFlipper



class GreatestHitsDataset(torch.utils.data.Dataset):
    """
    Dataset to train onset detection model on Greatest Hits dataset. 
    Split videos into chunks of N seconds.
    Annotate each chunk with onset labels (1 if onset, 0 otherwise) for each video frame.
    """

    def __init__(
        self,
        root_dir,
        split_file_path,
        split='train',
        data_to_use=1.0,
        chunk_length_in_seconds=8.0,
        frames_transforms=None,
        sr=16000,
        audio_file_suffix='.resampled.wav',
        annotations_file_suffix=".times.csv",
        metadata_file_suffix=".metadata.json",
        frame_file_suffix=".jpg",
        force_channels="stereo"
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split_file_path = split_file_path
        self.split = split
        self.data_to_use = data_to_use
        self.chunk_length_in_seconds = chunk_length_in_seconds
        self.sr = sr
        self.audio_file_suffix = audio_file_suffix
        self.annotations_file_suffix = annotations_file_suffix
        self.metadata_file_suffix = metadata_file_suffix
        self.frame_file_suffix = frame_file_suffix
        self.force_channels = force_channels

        if frames_transforms is not None:
            self.frames_transforms = frames_transforms
        else:
            self.frames_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        
        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )


        # load list chunks from pickle if exists
        if os.path.exists(f"{self.root_dir}/list_chunks_{self.split}_prova6.pkl"):
            with open(f"{self.root_dir}/list_chunks_{self.split}_prova6.pkl", "rb") as f:
                self.list_chunks = pickle.load(f)
        
        else: 
            # read list of samples
            with open(split_file_path, "r") as f:
                self.list_samples = f.read().splitlines()

            # subset
            if data_to_use < 1.0:
                # shuffle list
                random.shuffle(self.list_samples)
                self.list_samples = self.list_samples[0:int(len(self.list_samples) * data_to_use)]
                self.list_samples = natsorted(self.list_samples)  # natural sorting (e.g. 1, 2, 10 instead of 1, 10, 2)

            self.list_chunks = []
            self.total_time_in_minutes = 0.0

            for sample in tqdm(self.list_samples, total=len(self.list_samples), desc=f"Loading {split} dataset"):
                # get metadata
                metadata_path = os.path.join(root_dir, sample, f"{sample}{metadata_file_suffix}")
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                frame_rate = metadata["processed"]["video_frame_rate"]
                duration = metadata["processed"]["video_duration"]

                # compute number of chunks
                num_chunks = int(duration / chunk_length_in_seconds)
                end_time = num_chunks * chunk_length_in_seconds

                audio = os.path.join(root_dir, sample, "audio", f"{sample}{audio_file_suffix}")
                audio, sr_native = torchaudio.load(audio)
                if sr_native != self.sr:
                    audio = torchaudio.functional.resample(audio, orig_freq=sr_native, new_freq=self.sr)

                # get annotations (onsets) up to end time
                annotations_path = os.path.join(root_dir, sample, f"{sample}{annotations_file_suffix}")
                annotations = pd.read_csv(annotations_path, header=None, names=["times", "labels"])
                onset_times = annotations["times"].values
                onset_times = onset_times[onset_times < end_time]

                self.total_time_in_minutes += end_time

                # get chunks
                chunk_length_in_frames = int(chunk_length_in_seconds * frame_rate)
                for i in range(num_chunks):
                    # chunk start and end
                    chunk_start_time = i * chunk_length_in_seconds
                    chunk_end_time = chunk_start_time + chunk_length_in_seconds
                    chunk_start_frame = int(chunk_start_time * frame_rate)
                    chunk_end_frame = int(chunk_end_time * frame_rate)
                    chunk_start_sr = int(chunk_start_time * self.sr)
                    chunk_end_sr = int(chunk_end_time * self.sr)

                    audio_chunk = audio[:, chunk_start_sr:chunk_end_sr]
                    if self.augs is not None:
                        audio_chunk = self.augs(audio_chunk)
                    # Normalize audio amplitude between -1 and 1
                    audio_chunk = audio_chunk / torch.max(torch.abs(audio_chunk))
                    # audio_chunk = audio_chunk.clamp(-1,1)
                    if self.encoding is not None:
                        audio_chunk = self.encoding(audio_chunk)

                    # extract onset times for this chunk
                    chunk_onsets_times = annotations[(annotations["times"] >= chunk_start_time) & (annotations["times"] < chunk_end_time)]["times"].values

                    # normalize to chunk start
                    chunk_onsets_times = chunk_onsets_times - chunk_start_time

                    # convert onset times to frames
                    chunk_onsets_frames = (chunk_onsets_times * frame_rate).astype(int)

                    # compute frames labels
                    labels = torch.zeros(chunk_length_in_frames)
                    labels[chunk_onsets_frames] = 1

                    # append chunk
                    self.list_chunks.append({
                        "video_name": sample,
                        "frames_path": os.path.join(root_dir, sample, "frames"),
                        "start_time": chunk_start_time,
                        "end_time": chunk_end_time,
                        "start_frame": chunk_start_frame,
                        "end_frame": chunk_end_frame,
                        "audio": audio_chunk,
                        "labels": labels,
                        "frame_rate": frame_rate,
                        "sample_rate": self.sr,
                        "seconds_start": 0.0,
                        "seconds_total": self.chunk_length_in_seconds,
                    })

            self.total_time_in_minutes /= 60.0
            with open(f"{self.root_dir}/list_chunks_{self.split}_prova6.pkl", "wb") as f:
                pickle.dump(self.list_chunks, f)



    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, index):
        try:
            chunk = self.list_chunks[index]
            frames_list = glob.glob(f"{chunk['frames_path']}/*{self.frame_file_suffix}")
            frames_list = natsorted(frames_list)

            # get frames
            frames_list = frames_list[chunk["start_frame"]:chunk["end_frame"]]
            frames = self.read_image_and_apply_transforms(frames_list)
        # except Exception as e:
        #     print(f"Error reading frames for chunk {index}: {e}")
        #     print(chunk)
        #     print('---')
        #     return self.__getitem__(index + 1)
        # get labels
            labels = chunk["labels"]

        # get audio
            audio = chunk["audio"]

        # get mel spectrogram
            # spec = chunk["spec"]

            seconds_start = chunk["seconds_start"]
            seconds_total = chunk["seconds_total"]

            item = {
                "video_name": chunk["video_name"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "start_frame": chunk["start_frame"],
                "end_frame": chunk["end_frame"],
                "label": labels,
                "frame_rate": chunk["frame_rate"],
                "sample_rate": chunk["sample_rate"],
            }

        except Exception as e:
            print(f"Error reading frames for chunk {index}: {e}")
            print(chunk)
            print('---')
            return self.__getitem__(index + 1)

        return audio, frames, seconds_start, seconds_total, [item]

    def read_image_and_apply_transforms(self, frame_list):
        imgs = []
        #convert_tensor = transforms.ToTensor()
        for img_path in frame_list:
            image = Image.open(img_path).convert('RGB')
            #image = convert_tensor(image)
            if self.frames_transforms is not None:
                image = self.frames_transforms(image)
            imgs.append(image.unsqueeze(0))
        # (T, C, H ,W)
        imgs = torch.cat(imgs, dim=0).squeeze()
        # if self.frames_transforms is not None:
        #     imgs = self.frames_transforms(imgs)
        imgs = imgs.permute(1, 0, 2, 3)
        # (C, T, H ,W)
        return imgs
    
    def amplitude_envelope(self, signal):
        """Calculate the amplitude envelope of a signal with a given frame size nad hop length."""
        amplitude_envelope = []
    
        # calculate amplitude envelope for each frame
        for i in range(0, len(signal), self.hop_length): 
            amplitude_envelope_current_frame = max(signal[i:i+self.frame_size]) 
            amplitude_envelope.append(amplitude_envelope_current_frame)
            
        return torch.tensor(amplitude_envelope)

    def print(self):
        print(f"\nGreatesthit {self.split} dataset:")
        print(f"num {self.split} chunks: {len(self.list_chunks)}")
        audio, frames, seconds_start, seconds_total, item = self[0]
        print(f"chunk frames size: {frames.shape}")
        print(f"chunk audio size: {audio.shape}")

if __name__ == '__main__':
    dataset = GreatestHitsDataset(
        root_dir="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed",
        split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/val.txt",
        split='val',
        data_to_use=0.1,
        chunk_length_in_seconds=2.0,
        frames_transforms=[
            transforms.ToTensor(),
            transforms.Resize((240, 240), antialias=True),
            transforms.RandomCrop((224, 224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        audio_file_suffix=".resampled.wav",
        annotations_file_suffix=".times.csv",
        metadata_file_suffix=".metadata.json",
        frame_file_suffix=".jpg",
    )
    dataset.print()
