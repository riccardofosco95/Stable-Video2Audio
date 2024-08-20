import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from subprocess import CalledProcessError
import logging
import csv
import os
from tqdm import tqdm
pylog = logging.getLogger(__name__)

def _get_youtube_link(youtube_id: str, start_time: Optional[int]) -> str:
    link = f"https://www.youtube.com/watch?v={youtube_id}"
    if start_time is None:
        return link
    else:
        start_time_in_seconds = start_time // 1000
        return f"{link}&t={start_time_in_seconds}s"

def download_id(youtube_id= "---g-f_I2yQ" ,start_time =1):

    verbose=3
    # Get audio download link with yt-dlp, without start time
    link = _get_youtube_link(youtube_id, start_time)
    get_url_command = [
        "yt-dlp",
        "--youtube-skip-dash-manifest",
        "-g",
        link,
    ]
    try:
        output = subprocess.check_output(get_url_command)
    except (CalledProcessError, PermissionError) as err:
        if verbose >= 2:
            print(err)
        return False

    output = output.decode()
    lines = output.split("\n")
    if len(lines) < 1:
        return False
    _video_link = lines[0]
    audio_link = lines[1] if len(lines) > 1 else lines[0]

    audio_format = "flac"
    acodec= "flac"
    audio_duration=10
    audio_n_channels= 1
    sr= 44_100

    fpath_out = f"/import/c4dm-datasets-ext/DIFF-SFX/AudioSet-Strong/samples/{youtube_id}_{start_time}.mp4"

    start_time_in_seconds = start_time // 1000

    # Extract video
    extract_command = [
        "ffmpeg",
        # Input
        "-i",
        _video_link,
        # Get only 10s of the clip after start_time
        "-ss",
        str(start_time_in_seconds),
        "-t",
        str(audio_duration),
        fpath_out,
    ]
    try:
        if verbose < 3:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        else:
            stdout = None
            stderr = None
        exitcode = subprocess.check_call(extract_command, stdout=stdout, stderr=stderr)

    except (CalledProcessError, PermissionError) as err:
        if verbose >= 2:
            print(err)
        return False

    # Extract audio
    fpath_out = f"/import/c4dm-datasets-ext/DIFF-SFX/AudioSet-Strong/samples/{youtube_id}_{start_time}.mp3"
    # Download and extract audio from audio_link to fpath_out with ffmpeg
    extract_command = [
        "ffmpeg",
        # Input
        "-i",
        audio_link,
        # Remove video
        "-vn",
        # Format (flac)
        "-f",
        audio_format,
        # Audio codec (flac)
        "-acodec",
        acodec,
        # Get only 10s of the clip after start_time
        "-ss",
        str(start_time_in_seconds),
        "-t",
        str(audio_duration),
        # Resample to a specific rate (default to 32 kHz)
        "-ar",
        str(sr),
        # Compute mean of 2 channels
        "-ac",
        str(audio_n_channels),
        fpath_out,
    ]
    try:
        if verbose < 3:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        else:
            stdout = None
            stderr = None
        exitcode = subprocess.check_call(extract_command, stdout=stdout, stderr=stderr)
        return exitcode == 0

    except (CalledProcessError, PermissionError) as err:
        if verbose >= 2:
            print(err)
        return False

if __name__ == "__main__":
    with open('/homes/rfg543/Documents/Stable-Video2Audio/main/data/audioset_strong/audioset_eval_strong.tsv') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        dl_list = [row[:4] for row in reader]

    for row in tqdm(dl_list):
        segment_id, start_time_seconds, end_time_seconds, label = row
        video_id, start_time = segment_id.rsplit('_', 1)
        start_time = int(start_time)
        if not os.path.exists(f"/import/c4dm-datasets-ext/DIFF-SFX/AudioSet-Strong/samples/{segment_id}.mp3") and not os.path.exists(f"/import/c4dm-datasets-ext/DIFF-SFX/AudioSet-Strong/samples/{segment_id}.mp4"):
            download_id(video_id, start_time)