# Stable-Video2Audio

Stable-Video2Audio generates audio that is semantically and temporally aligned with a reference video input. 

The architecture is divided into two distinct parts: a Video2RMS (not yet available) that maps the input video to a continuous RMS envelope representative of the audio to be generated, and Stable-Foley, a generative model that produces the final audio output guided by the RMS and conditioned by 1. a sample representative of the semantics desired for the generated audio, 2. features representative of the video frames, and 3. the length of the audio to be generated.

Stable-Foley is based on Stable Audio Open and implements a ControlNet to Fine-Tine the Diffusion Transformer (DiT) and guide its generation via a continuous RMS envelope. 
Through such guidance, it is possible to control the timing and intensity over time of the various sounds to be generated in the output audio track. 
In addition, fine-tuning with ControlNet permits the use of only 20% of the DiT layers (5 layers), making the model lightweight and fast. 
The semantics of the audio is controlled by conditioning the model through cross-attention with embeddings of the reference audio generated by CLAP. 
To further improve the semantics and alignment of the produced audio with respect to the video, additional conditioning of the video frame embeddings produced by CAVP (https://diff-foley.github.io/) is used, which is a video encoder that extracts features from the frames that are relevant to the audio associated with those frames.

## Usage

Clone this repository

```
git clone https://github.com/riccardofosco95/Stable-Video2Audio.git

cd Stable-Video2Audio
```

Install the requirements (it is recommended to use Python version 3.8.10)

```
pip install -r requirements_aes.txt
```

## Checkpoints

You can download Stable-Foley, CLAP and CAVP checkpoints through this [Google Drive](https://drive.google.com/drive/folders/1A4b1fKQyIy8h9EOmQGU_8WxLL7Fvfd6j) folder. 
You can download the `\logs` folder and place it in the main directory of this repository.

Afterwards, log in on Hugginface with `huggingface-cli login` (this can be done in the [Inference Notebook](/notebook/inference_gh.ipynb)) using personal token in order to be able to download Stable Audio Open weights.


## Demo

A [Inference Notebook](/notebook/inference_gh.ipynb) (`/Stable-Video2Audio/notebook/inference_gh.ipynb`) is available to generate audios through Stable-Foley.

The model was trained on GreatestHits, a video dataset of people hitting or scratching different objects with a drumstick for more info: [GreatestHits Dataset](https://andrewowens.com/vis/). 

Five test videos can be found in `/Stable-Video2Audio/notebook/gh_data`, where `samples-processed-4fps-44kHz`contains the audio and frames extrcated from each video. 
The audio is resampled at 44100 Hz (this will be the sample rate of the generated stereo audio) and the frames are extracted at 4fps (this is because cavp was trained on video with this frame rate).

!!! First of all, Stable-Foley, CLAP and CAVP checkpoints must be downloaded. 
A `/logs` folder should be placed in the main directory of this reopository and should have the following structure:

```
logs/
    cavp_ckpt/
        # CAVP ckpt
    ckpts/
        # Stable-Foley ckpt
    clap_ckpt
        # CLAP ckpt
```

!!! You will also need to modify paths in `/Stable-Video2Audio/main/controlnet/pretrained.py`:

- `ln:29` : `"config_path": absolute path to "/Stable-Video2Audio/main/CAVP/config/Stage1_CAVP.yaml"`
- `ln:30` : `"ckpt_path": absolute path to your CAVP ckpt`

- `ln:40` : `"clap_ckpt_path": absolute path to your CLAP ckpt`

Also, carefully read the comments in the provided notebook to know where you need to change other paths to checkpoints.




## Train

```
PYTHONUNBUFFERED=1 TAG=gh-controlnet python3 train.py exp=train_gh_controlnet.yaml
```
