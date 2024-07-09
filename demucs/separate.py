from pathlib import Path
import torch as th
# from .api import Separator, save_audio

import subprocess

import torch as th
import torchaudio as ta

from typing import Optional, Callable, Dict, Tuple, Union

from .apply import apply_model
from .audio import AudioFile, save_audio

# my_script.py
import argparse



class Separator:
    def __init__(
        self,
        device= "cpu",
        overlap= 0.25,
    ):
        self._overlap = overlap
        self._device = device
        self._audio_channels = 2
        self._samplerate = 44100
        self.sources = ['drums', 'bass', 'other', 'vocals']

    def separate_audio_file(self, file, model_path):
        wav = AudioFile(file).read(streams=0, samplerate=self._samplerate, channels=self._audio_channels)
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std() + 1e-8
        out = apply_model(
                None,
                wav[None],
                overlap=self._overlap,
                device=self._device,
                model_path=model_path,
            )
        out *= ref.std() + 1e-8
        out += ref.mean()
        wav *= ref.std() + 1e-8
        wav += ref.mean()
        return (wav, dict(zip(self.sources, out[0])))





def main(file_path, dir_out = './output', 
        device= "cuda" if th.cuda.is_available() else "cpu", 
        model_path=None):
    file_path = Path(file_path)
    out = Path(dir_out) 
    out.mkdir(parents=True, exist_ok=True) 

    # separate audio file
    separator = Separator(device=device, overlap=0.25)
    _, res = separator.separate_audio_file(file_path.resolve(), model_path=model_path)

    # save audio
    for name, source in res.items():
        stem = out / f"{file_path.stem}_{name}.wav"
        stem.parent.mkdir(parents=True, exist_ok=True)
        save_audio(source, str(stem), samplerate=separator._samplerate)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--input_audio', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    main(args.input_audio, device=args.device, model_path=args.model_path)

