from concurrent.futures import ThreadPoolExecutor
import copy
import random
from threading import Lock
import typing as tp

import torch as th
from torch import nn
from torch.nn import functional as F
import tqdm

import torch


def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):

    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor

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


from fractions import Fraction

def apply_model(model, mix,
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
    # model = model.models[0]

    model = torch.jit.load("/Users/apple/htdemus-export/scriptmodule.pt", map_location=torch.device(device))
    # model.eval()

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
        # import torch
        # model = torch.jit.trace(model, padded_mix)
        # torch.jit.save(model, 'scriptmodule.pt')
        # return
        out = model(padded_mix)
    return center_trim(out, length)