# Customize the following options!
model = "htdemucs"
extensions = ["mp3", "wav", "ogg", "flac"]  # we will look for all those file types.
two_stems = None   # only separate one stems from the rest, for instance
# two_stems = "vocals"

# Options for the output audio.
mp3 = True
mp3_rate = 320
float32 = False  # output as float 32 wavs, unsused if 'mp3' is True.
int24 = False    # output as int24 wavs, unused if 'mp3' is True.
# You cannot set both `float32 = True` and `int24 = True` !!

#@title Useful functions, don't forget to execute
import io
from pathlib import Path
import select
from shutil import rmtree
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO

# def separate(file, outp=None):
    # cmd = ["python3", "-m", "demucs.separate", "-o", str(outp), "-n", model]  + ["--mp3", f"--mp3-bitrate={mp3_rate}"]
    # p = sp.Popen(cmd + [file])
    # p.wait()

# from demucs.separate import main
# args = object()

# main()

# python3 -m demucs.separate -o ./ -n htdemucs --mp3 --mp3-bitrate=320
