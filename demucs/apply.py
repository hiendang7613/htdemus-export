# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Code to apply a model to a mix. It will handle chunking with overlaps and
inteprolation between chunks, as well as the "shift trick".
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import random
from threading import Lock
import typing as tp

import torch as th
from torch import nn
from torch.nn import functional as F
import tqdm

from .demucs import Demucs
from .hdemucs import HDemucs
from .htdemucs import HTDemucs
from .utils import center_trim, DummyPoolExecutor

Model = tp.Union[Demucs, HDemucs, HTDemucs]
import torch


class BagOfModels(nn.Module):
    def __init__(self, models: tp.List[Model],
                 weights: tp.Optional[tp.List[tp.List[float]]] = None,
                 segment: tp.Optional[float] = None):
        """
        Represents a bag of models with specific weights.
        You should call `apply_model` rather than calling directly the forward here for
        optimal performance.

        Args:
            models (list[nn.Module]): list of Demucs/HDemucs models.
            weights (list[list[float]]): list of weights. If None, assumed to
                be all ones, otherwise it should be a list of N list (N number of models),
                each containing S floats (S number of sources).
            segment (None or float): overrides the `segment` attribute of each model
                (this is performed inplace, be careful is you reuse the models passed).
        """
        super().__init__()
        assert len(models) > 0
        first = models[0]
        for other in models:
            assert other.sources == first.sources
            assert other.samplerate == first.samplerate
            assert other.audio_channels == first.audio_channels
            if segment is not None:
                if not isinstance(other, HTDemucs) and segment > other.segment:
                    other.segment = segment

        self.audio_channels = first.audio_channels
        self.samplerate = first.samplerate
        self.sources = first.sources
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [[1. for _ in first.sources] for _ in models]
        else:
            assert len(weights) == len(models)
            for weight in weights:
                assert len(weight) == len(first.sources)
        self.weights = weights

    @property
    def max_allowed_segment(self) -> float:
        max_allowed_segment = float('inf')
        for model in self.models:
            if isinstance(model, HTDemucs):
                max_allowed_segment = min(max_allowed_segment, float(model.segment))
        return max_allowed_segment

    def forward(self, x):
        raise NotImplementedError("Call `apply_model` on this.")


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, th.Tensor)
        return TensorChunk(tensor_or_chunk)


def _replace_dict(_dict: tp.Optional[dict], *subs: tp.Tuple[tp.Hashable, tp.Any]) -> dict:
    if _dict is None:
        _dict = {}
    else:
        _dict = copy.copy(_dict)
    for key, value in subs:
        _dict[key] = value
    return _dict

from fractions import Fraction

def apply_model(model: tp.Union[BagOfModels, Model],
                mix: tp.Union[th.Tensor, TensorChunk],
                num_workers: int = 8,
                overlap: float = 0.25, 
                device: tp.Union[str, th.device] = 'cuda',
                transition_power: float = 1.,
                len_model_sources = 4,
                segment = Fraction(39,5),
                samplerate = 44100
                ) -> th.Tensor:
    
    pool=ThreadPoolExecutor(max_workers=num_workers)
    model_weights = [1.]*len_model_sources
    totals = [0.] * len_model_sources
    batch, channels, length = mix.shape

    import torch
    model = torch.jit.load("/Users/apple/demucs/scriptmodule.pt")
    model.to(device)
    model.eval()

    segment_length: int = int(samplerate * segment)
    stride = int((1 - overlap) * segment_length)
    offsets = range(0, length, stride)
    futures = []
    for offset in offsets:
        print("for offset in offsets:", offset)
        chunk = TensorChunk(mix, offset, segment_length)
        future = pool.submit(run_model, model, chunk, device, samplerate, segment)
        futures.append((future, offset))
        offset += segment_length
    if True:
        scale = float(format(stride / samplerate, ".2f"))
        futures = tqdm.tqdm(futures, unit_scale=scale, ncols=120, unit='seconds')

    out = th.zeros(batch, len_model_sources, channels, length, device=mix.device)
    sum_weight = th.zeros(length, device=mix.device)
    weight = th.cat([th.arange(1, segment_length // 2 + 1, device=device),
                        th.arange(segment_length - segment_length // 2, 0, -1, device=device)])
    weight = (weight / weight.max())**transition_power
    for future, offset in futures:
        chunk_out = future.result()  # type: th.Tensor
        chunk_length = chunk_out.shape[-1]
        out[..., offset:offset + segment_length] += (weight[:chunk_length] * chunk_out).to(mix.device)
        sum_weight[offset:offset + segment_length] += weight[:chunk_length].to(mix.device)
    out /= sum_weight

    for k, inst_weight in enumerate(model_weights):
        out[:, k, :, :] *= inst_weight
        totals[k] += inst_weight
    for k in range(out.shape[1]):
        out[:, k, :, :] /= totals[k]
    return out
   
def run_model(model, mix, device, samplerate, segment):
    length = mix.shape[-1]
    valid_length = int(segment * samplerate)
    mix = TensorChunk(mix)
    padded_mix = mix.padded(valid_length).to(device)
    with th.no_grad():
        out = model(padded_mix)
    return center_trim(out, length)