"""Tiled inference helpers shared by ``test.py`` and ``model_complexity.py``.

``test.py`` historically carried its own copy of the split / merge logic. The
functions here are that same logic, factored out so the profiler measures
*exactly* the inference procedure that produces the reported PSNR/SSIM numbers
rather than an idealised single forward pass.

Tiling rules (unchanged from ``test.py``):

* tile starts step by ``crop_size - overlap_size`` and the last start is
  clamped to ``length - crop_size`` so the border is always covered;
* overlapping tiles are merged with an inverse-distance score map that
  down-weights tile borders;
* the spiking state is reset before every tile, so tiles are independent.

Images smaller than ``crop_size`` are replicate-padded up to ``crop_size`` and
cropped back afterwards, so the caller never has to special-case them.
"""

import torch
import torch.nn.functional as F

__all__ = [
    "tiled_restore", "split_image", "merge_image", "get_scoremap",
    "tile_count", "pad_to_multiple",
]

# The encoder halves the resolution twice and the decoder upsamples by exactly
# 2x, so every spatial size fed to the network must be a multiple of 4.
SIZE_MULTIPLE = 4

_SCOREMAP_CACHE = {}


def _tile_starts(length, tile, stride):
    """Start offsets covering ``length`` with tiles of ``tile`` px."""
    tile = int(min(tile, length))
    stride = int(max(stride, 1))
    last = int(length) - tile
    starts = list(range(0, last + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != last:
        starts.append(last)
    return starts


def get_scoremap(height, width, channels, batch=1, is_mean=False,
                 device=None, dtype=torch.float32):
    """Inverse-distance-to-centre blending weights for one tile.

    Vectorised equivalent of the per-pixel loop that used to live in
    ``test.py``; values are identical.
    """
    key = (height, width, channels, batch, bool(is_mean), str(device), str(dtype))
    cached = _SCOREMAP_CACHE.get(key)
    if cached is not None:
        return cached
    if is_mean:
        score = torch.ones((batch, channels, height, width),
                           device=device, dtype=dtype)
    else:
        rows = torch.arange(height, device=device, dtype=dtype) - height / 2.0
        cols = torch.arange(width, device=device, dtype=dtype) - width / 2.0
        distance = torch.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2 + 1e-3)
        score = (1.0 / distance).expand(batch, channels, height, width).contiguous()
    _SCOREMAP_CACHE[key] = score
    return score


def pad_to_multiple(image, multiple=SIZE_MULTIPLE, minimum=0):
    """Replicate-pad ``image`` so H, W are >= ``minimum`` and multiples of ``multiple``."""
    height, width = int(image.shape[-2]), int(image.shape[-1])
    target_h = max(height, int(minimum))
    target_w = max(width, int(minimum))
    if multiple > 1:
        target_h = ((target_h + multiple - 1) // multiple) * multiple
        target_w = ((target_w + multiple - 1) // multiple) * multiple
    if target_h == height and target_w == width:
        return image, (height, width)
    padded = F.pad(image, (0, target_w - width, 0, target_h - height), mode='replicate')
    return padded, (height, width)


def split_image(image, crop_size, overlap_size):
    """Split ``(B, C, H, W)`` into overlapping tiles; returns tiles and starts."""
    height, width = int(image.shape[-2]), int(image.shape[-1])
    tile_h = min(int(crop_size), height)
    tile_w = min(int(crop_size), width)
    stride = max(int(crop_size) - int(overlap_size), 1)
    tiles, starts = [], []
    for top in _tile_starts(height, tile_h, stride):
        for left in _tile_starts(width, tile_w, stride):
            tiles.append(image[:, :, top:top + tile_h, left:left + tile_w])
            starts.append((top, left))
    return tiles, starts, (tile_h, tile_w)


def merge_image(tiles, starts, tile_size, resolution):
    """Blend overlapping tiles back into a full image with the score map."""
    batch, channels, height, width = resolution
    tile_h, tile_w = tile_size
    reference = tiles[0]
    accumulator = torch.zeros((batch, channels, height, width),
                              device=reference.device, dtype=reference.dtype)
    weights = torch.zeros_like(accumulator)
    scoremap = get_scoremap(tile_h, tile_w, channels, batch=batch,
                            device=reference.device, dtype=reference.dtype)
    for tile, (top, left) in zip(tiles, starts):
        accumulator[:, :, top:top + tile_h, left:left + tile_w] += scoremap * tile
        weights[:, :, top:top + tile_h, left:left + tile_w] += scoremap
    return accumulator / weights


def tile_count(height, width, crop_size, overlap_size):
    """Number of tiles ``tiled_restore`` would run for an image of this size."""
    padded_h = max(int(height), int(crop_size))
    padded_w = max(int(width), int(crop_size))
    if SIZE_MULTIPLE > 1:
        padded_h = ((padded_h + SIZE_MULTIPLE - 1) // SIZE_MULTIPLE) * SIZE_MULTIPLE
        padded_w = ((padded_w + SIZE_MULTIPLE - 1) // SIZE_MULTIPLE) * SIZE_MULTIPLE
    stride = max(int(crop_size) - int(overlap_size), 1)
    tile_h = min(int(crop_size), padded_h)
    tile_w = min(int(crop_size), padded_w)
    return float(len(_tile_starts(padded_h, tile_h, stride))
                 * len(_tile_starts(padded_w, tile_w, stride)))


def tiled_restore(model, image, crop_size=64, overlap_size=8, reset_fn=None,
                  collect_band_counts=False, clamp=False):
    """Restore one image tile by tile.

    Args:
        model: SpikeRain model in eval mode, already on the right device.
        image: ``(B, C, H, W)`` tensor on the model's device.
        crop_size / overlap_size: tiling geometry (``crop_size`` is rounded up
            to a multiple of 4 because of the 2x/2x encoder-decoder).
        reset_fn: called with ``model`` before every tile; defaults to
            ``spikingjelly.activation_based.functional.reset_net``.
        collect_band_counts: also return the per-tile spectral-band count when
            the model exposes one (``model.last_band_count``). Variants without
            an adaptive spectral budget simply return an empty list.
        clamp: clamp the merged result to ``[0, 1]``.

    Returns:
        ``(restored, band_counts)`` where ``restored`` has the same shape as
        ``image``.
    """
    if reset_fn is None:
        from spikingjelly.activation_based import functional as sj_functional
        reset_fn = sj_functional.reset_net

    crop_size = int(crop_size)
    if crop_size % SIZE_MULTIPLE:
        crop_size += SIZE_MULTIPLE - (crop_size % SIZE_MULTIPLE)

    if image.dim() != 4:
        raise ValueError("tiled_restore expects a (B, C, H, W) tensor, got "
                         "{}".format(tuple(image.shape)))

    padded, (height, width) = pad_to_multiple(image, SIZE_MULTIPLE, minimum=crop_size)
    batch, channels = int(padded.shape[0]), int(padded.shape[1])
    tiles, starts, tile_size = split_image(padded, crop_size, overlap_size)

    restored_tiles = []
    band_counts = []
    for tile in tiles:
        reset_fn(model)
        output = model(tile)
        reset_fn(model)
        restored_tiles.append(output)
        if collect_band_counts:
            bands = getattr(model, 'last_band_count', None)
            if bands is not None:
                band_counts.append(float(bands))

    merged = merge_image(restored_tiles, starts, tile_size,
                         (batch, channels, int(padded.shape[-2]), int(padded.shape[-1])))
    merged = merged[:, :, :height, :width]
    if clamp:
        merged = torch.clamp(merged, 0.0, 1.0)
    return merged, band_counts
