from pathlib import Path
import torch as th
from .api import Separator, save_audio

def main(file_path):
    print(file_path)

    dir_out = './'
    model_name = "htdemucs"
    device = "cuda" if th.cuda.is_available() else "cpu"
    mp3_bitrate = 320

    separator = Separator(model=model_name, device=device, overlap=0.25)

    file_path = Path(file_path)

    out = Path(dir_out) / model_name
    out.mkdir(parents=True, exist_ok=True)

    _, res = separator.separate_audio_file(file_path.resolve())

    ext = "mp3"
    mp3_preset = 2
    clip_mode = 'rescale'
    float32 = False
    int24 = False

    kwargs = {
        "samplerate": separator._samplerate,
        "bitrate": mp3_bitrate,
        "preset": mp3_preset,
        "clip": clip_mode,
        "as_float": float32,
        "bits_per_sample": 24 if int24 else 16,
    }

    for name, source in res.items():
        stem = out / f"{file_path.stem}_{name}.{ext}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        save_audio(source, str(stem), **kwargs)


if __name__ == "__main__":
    main('test.mp3')