import subprocess

import torch as th
import torchaudio as ta

from dora.log import fatal
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, Union

from .apply import apply_model, _replace_dict
from .audio import AudioFile, convert_audio, save_audio
from .pretrained import get_model, _parse_remote_files, REMOTE_ROOT
from .repo import RemoteRepo, LocalRepo, ModelOnlyRepo, BagOnlyRepo


class LoadAudioError(Exception):
    pass


class LoadModelError(Exception):
    pass


class _NotProvided:
    pass


NotProvided = _NotProvided()


class Separator:
    def __init__(
        self,
        model: str = "htdemucs",
        repo: Optional[Path] = None,
        device: str = "cuda" if th.cuda.is_available() else "cpu",
        overlap: float = 0.25,
    ):
        self._name = model
        self._repo = repo
        self._load_model()
        self._overlap = overlap
        self._device = device


    def _load_model(self):
        self._model = get_model(name=self._name, repo=self._repo)
        if self._model is None:
            raise LoadModelError("Failed to load model")
        self._audio_channels = 2
        self._samplerate = 44100
 
 
 
    def separate_audio_file(self, file: Path):
        wav = AudioFile(file).read(streams=0, samplerate=self._samplerate, channels=self._audio_channels)
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std() + 1e-8
        out = apply_model(
                self._model,
                wav[None],
                overlap=self._overlap,
                device=self._device,
            )
        out *= ref.std() + 1e-8
        out += ref.mean()
        wav *= ref.std() + 1e-8
        wav += ref.mean()
        return (wav, dict(zip(self._model.sources, out[0])))

    # @property
    # def samplerate(self):
    #     return self._samplerate

    # @property
    # def audio_channels(self):
    #     return self._audio_channels

    # @property
    # def model(self):
    #     return self._model


# def list_models(repo: Optional[Path] = None) -> Dict[str, Dict[str, Union[str, Path]]]:
   
#     model_repo: ModelOnlyRepo
#     if repo is None:
#         models = _parse_remote_files(REMOTE_ROOT / 'files.txt')
#         model_repo = RemoteRepo(models)
#         bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
#     else:
#         if not repo.is_dir():
#             fatal(f"{repo} must exist and be a directory.")
#         model_repo = LocalRepo(repo)
#         bag_repo = BagOnlyRepo(repo, model_repo)
#     return {"single": model_repo.list_model(), "bag": bag_repo.list_model()}


if __name__ == "__main__":
    # Test API functions
    # two-stem not supported

    from .separate import get_parser

    args = get_parser().parse_args()
    separator = Separator(
        model=args.name,
        repo=args.repo,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
        split=args.split,
        segment=args.segment,
        jobs=args.jobs,
        callback=print
    )
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    for file in args.tracks:
        separated = separator.separate_audio_file(file)[1]
        if args.mp3:
            ext = "mp3"
        elif args.flac:
            ext = "flac"
        else:
            ext = "wav"
        kwargs = {
            "samplerate": separator.samplerate,
            "bitrate": args.mp3_bitrate,
            "clip": args.clip_mode,
            "as_float": args.float32,
            "bits_per_sample": 24 if args.int24 else 16,
        }
        for stem, source in separated.items():
            stem = out / args.filename.format(
                track=Path(file).name.rsplit(".", 1)[0],
                trackext=Path(file).name.rsplit(".", 1)[-1],
                stem=stem,
                ext=ext,
            )
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(source, str(stem), **kwargs)
