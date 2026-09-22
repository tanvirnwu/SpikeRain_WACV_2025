"""Parameter / operation / energy / latency profiler for SpikeRain-SpectraSpike.

This module is the single source of truth for the efficiency numbers reported
in the paper. Everything it prints is *measured* on a real checkpoint with real
test images -- nothing is estimated from a formula alone:

* Parameter counts come from the instantiated model.
* Operation counts come from forward hooks that see the actual tensor shapes
  produced by the actual input, so tiled inference, the octave band bank and
  the adaptive spectral budget (Idea D, which changes how many bands are
  processed per image) are all reflected in the totals.
* Whether a convolution is spike-driven (accumulate-only, SOPs) or a dense ANN
  convolution (multiply-accumulate, MACs) is decided by *inspecting the values
  of its input tensor at run time*, not by a hand-maintained layer list. A
  tensor counts as a spike tensor when all of its values lie on the grid
  {0, 1/D, 2/D, ..., 1} for some small integer D (D = 1 is the binary LIF case,
  D > 1 is the NI-LIF integer-spike case of Idea C).
* Firing rates are the measured mean spike count per input element, so
  SOP_l = MAC_l * mean_spike_count_per_input_element.
* Latency is wall-clock time with CUDA synchronisation, warm-up iterations and
  repeated runs, reported as mean / std / median / min / max.

Energy model
------------
Following Chen et al., AAAI 2026 ("10894-AAAI26.ChenS-CV.pdf", Experimental
Setup) and the prior work it cites (Hu, Tang & Pan 2021; Song et al. 2024), a
45 nm technology node is assumed with

    E_MAC  = 12.5 pJ   per ANN multiply-accumulate operation ("per FLOP")
    E_SOP  = 77   fJ   per synaptic operation (accumulate driven by a spike)
    E_SIGN = 3.7  pJ   per spike-based Sign operation (energy per spike)

    E = E_MAC * MACs_ANN + E_SOP * SOPs + E_SIGN * SignOps

``--sign_op_mode spike`` (default) charges E_SIGN once per *emitted* spike,
which is what "calculated using energy per spike" in that paper means.
``--sign_op_mode neuron`` instead charges it once per neuron update
(T x B x C x H x W per spiking layer); both numbers are always reported.

Two accounting modes are reported side by side:

``full``        every measured operation is charged: convolutions, linear
                layers, normalisation, pooling, element-wise attention
                arithmetic, the Laplacian octave bank and the spectral-budget
                statistics. This is the honest upper bound.
``conv_only``   only convolution / linear operations plus the Sign term, which
                is the convention used by the SNN deraining literature the
                comparison tables are drawn from. Use this number when
                comparing against published Params / FLOPs / Energy columns.

Unit conventions (stated explicitly because the literature is inconsistent):
one MAC is counted as one operation. The ``FLOPs`` field is reported as
2 x MACs (one multiply + one add); the ``MACs`` field is the thop-style count.
The energy model above is defined per MAC, so ``MACs`` -- not ``FLOPs`` -- is
what drives the energy figure.

Architecture coverage
---------------------
The profiler discovers what to count from the model it is handed, so it works
with the released SpikeRain (DSRB / MDSA / Temporal Fusion / ARFE, binary
LIFNode) and stays correct if the SpectraSpike components -- the NI-LIF integer
neuron, the spike-driven cross-band attention, the Laplacian octave bank and
the adaptive spectral budget -- are added later. Those components are imported
optionally: whichever of them exist in ``model/`` get their dedicated hooks,
and anything not present is simply absent from the breakdown. Any leaf module
with no counter is listed under ``unhandled_modules`` in the report, so nothing
is ever silently uncounted.

Standalone use
--------------
    python model_complexity.py \
        --data_path /path/to/Rain200H/test/input \
        --model_version M \
        --weights ./checkpoints/SpikeRain_M/models/.../model_best.pth

Programmatic use (``test.py --profile`` does this after restoration):

    from model_complexity import profile_efficiency, print_efficiency_summary
    report = profile_efficiency(model, image_paths=[...], crop_size=64, ...)
    print_efficiency_summary(report)
"""

import argparse
import inspect
import json
import math
import os
import platform
import statistics
import time
from collections import OrderedDict
from glob import glob

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from inference_utils import SIZE_MULTIPLE, tile_count, tiled_restore

# --------------------------------------------------------------------------
# Energy constants (45 nm, Chen et al. AAAI 2026)
# --------------------------------------------------------------------------
ENERGY_PER_MAC_PJ = 12.5
ENERGY_PER_SOP_PJ = 0.077
ENERGY_PER_SIGN_PJ = 3.7

ENERGY_REFERENCE = (
    "Chen et al., AAAI 2026 (10894-AAAI26.ChenS-CV.pdf): 12.5 pJ per FLOP (MAC), "
    "77 fJ per SOP, 3.7 pJ per spike-based Sign operation, 45 nm technology node."
)

IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")

# Bilinear interpolation is charged 4 multiply-accumulates per output element
# (the four-tap weighted sum); this is stated so the number is auditable.
BILINEAR_MACS_PER_OUTPUT = 4
# Depthwise Gaussian kernel of the Laplacian octave bank is 5x5.
OCTAVE_KERNEL_TAPS = 25


def _is_container(module):
    return isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.Identity))


def _numel(tensor):
    return float(tensor.numel())


class OperationCounter:
    """Accumulates measured operation counts over one or more forward passes.

    All value-derived quantities (SOPs, spike counts) are accumulated as 0-dim
    CUDA tensors so that no host/device synchronisation happens inside the
    profiled forward passes; they are converted to Python floats only once, at
    the end, by :meth:`totals`.
    """

    SHAPE_BUCKETS = (
        "mac_conv_linear", "mac_norm", "mac_pool", "mac_elementwise",
        "mac_octave_bank", "neuron_updates",
    )
    VALUE_BUCKETS = ("sop_conv_linear", "sop_elementwise", "spike_count")

    def __init__(self, device, max_spike_levels=16, classify_calls=2,
                 tolerance=1e-4, min_spike_elements=64):
        self.device = device
        self.max_spike_levels = int(max_spike_levels)
        self.classify_calls = int(classify_calls)
        self.tolerance = float(tolerance)
        self.min_spike_elements = int(min_spike_elements)
        self.unhandled = OrderedDict()
        self.reset()

    def reset(self):
        self.shape_totals = OrderedDict((name, 0.0) for name in self.SHAPE_BUCKETS)
        self.value_totals = OrderedDict(
            (name, torch.zeros((), dtype=torch.float64, device=self.device))
            for name in self.VALUE_BUCKETS
        )
        self.per_module = OrderedDict()
        self._spike_cache = {}
        self._classify_seen = {}
        self.forward_calls = 0

    # -- accumulation helpers ------------------------------------------------
    def add_shape(self, bucket, value, name=None, module_type=None):
        self.shape_totals[bucket] += float(value)
        if name is not None:
            record = self._record(name, module_type)
            record[bucket] = record.get(bucket, 0.0) + float(value)

    def add_value(self, bucket, tensor, name=None, module_type=None):
        tensor = tensor.detach().to(dtype=torch.float64, device=self.device)
        self.value_totals[bucket] = self.value_totals[bucket] + tensor
        if name is not None:
            record = self._record(name, module_type)
            if bucket in record:
                record[bucket] = record[bucket] + tensor
            else:
                record[bucket] = tensor

    def _record(self, name, module_type):
        record = self.per_module.get(name)
        if record is None:
            record = OrderedDict()
            record["type"] = module_type or ""
            record["calls"] = 0
            self.per_module[name] = record
        return record

    def note_call(self, name, module_type=None):
        record = self._record(name, module_type)
        record["calls"] += 1

    # -- spike detection -----------------------------------------------------
    def spike_levels(self, name, tensor):
        """Measured number of spike levels ``D`` of ``tensor``, or ``None``.

        A tensor is a spike tensor when every value lies on {0, 1/D, ..., 1}.
        The classification is measured on the first ``classify_calls`` forward
        passes of each module and then cached, because it is a structural
        property of the network while the *rate* is re-measured every call.
        """
        seen = self._classify_seen.get(name, 0)
        if seen >= self.classify_calls and name in self._spike_cache:
            return self._spike_cache[name]
        self._classify_seen[name] = seen + 1
        levels = self._measure_spike_levels(tensor)
        previous = self._spike_cache.get(name, "unset")
        if previous == "unset" or previous is None:
            self._spike_cache[name] = levels
        elif levels is None:
            # A layer that ever receives a non-spike tensor is not spike-driven.
            self._spike_cache[name] = None
        else:
            self._spike_cache[name] = max(previous, levels)
        return self._spike_cache[name]

    def _measure_spike_levels(self, tensor):
        """Decide whether ``tensor`` is a spike tensor and, if so, return ``D``.

        The test is deliberately strict so that an ordinary activation map can
        never be mistaken for spikes and silently priced at the accumulate
        rate: the tensor must be large enough to be a feature map, entirely
        within ``[0, 1]``, saturated at exactly ``1`` (a spike tensor always
        contains at least one maximal spike at feature-map sizes), and every
        value must sit on the grid ``{0, 1/D, ..., 1}`` for a small integer D.
        """
        if tensor.numel() < self.min_spike_elements or not tensor.is_floating_point():
            return None
        tolerance = self.tolerance
        flat = tensor.detach().float()
        minimum = float(flat.min())
        maximum = float(flat.max())
        if minimum < -tolerance or maximum > 1.0 + tolerance:
            return None
        if maximum <= tolerance:
            return 1  # a spike tensor in which nothing fired
        if abs(maximum - 1.0) > tolerance:
            return None
        smallest_positive = float(
            flat.masked_fill(flat <= tolerance, 1.0).min()
        )
        if smallest_positive <= tolerance:
            return None
        levels = int(round(1.0 / smallest_positive))
        if levels < 1 or levels > self.max_spike_levels:
            return None
        if abs(smallest_positive * levels - 1.0) > 1e-3:
            return None
        scaled = flat * levels
        if float((scaled - scaled.round()).abs().max()) > 1e-3:
            return None
        return levels

    # -- results -------------------------------------------------------------
    def totals(self):
        result = OrderedDict()
        for name, value in self.shape_totals.items():
            result[name] = float(value)
        for name, tensor in self.value_totals.items():
            result[name] = float(tensor.item())
        return result

    def module_table(self):
        table = OrderedDict()
        for name, record in self.per_module.items():
            entry = OrderedDict()
            for key, value in record.items():
                if torch.is_tensor(value):
                    entry[key] = float(value.item())
                else:
                    entry[key] = value
            table[name] = entry
        return table


# --------------------------------------------------------------------------
# Forward hooks
# --------------------------------------------------------------------------
def _first_tensor(inputs):
    for item in inputs:
        if torch.is_tensor(item):
            return item
    return None


def _conv_macs(module, output):
    kernel = 1
    for size in module.kernel_size:
        kernel *= int(size)
    in_per_group = int(module.in_channels) // int(module.groups)
    return _numel(output) * in_per_group * kernel


def _hook_conv(counter, name, module, inputs, output):
    counter.note_call(name, type(module).__name__)
    tensor = _first_tensor(inputs)
    macs = _conv_macs(module, output)
    if module.bias is not None:
        macs += _numel(output)
    levels = counter.spike_levels(name, tensor) if tensor is not None else None
    if levels is None:
        counter.add_shape("mac_conv_linear", macs, name, type(module).__name__)
        return
    # Spike-driven: SOP_l = MAC_l * (mean spike count per input element).
    mean_spike_count = tensor.detach().float().mean() * float(levels)
    counter.add_value("sop_conv_linear", mean_spike_count * macs, name,
                      type(module).__name__)
    record = counter._record(name, type(module).__name__)
    record["dense_macs_if_ann"] = record.get("dense_macs_if_ann", 0.0) + macs
    record["spike_levels"] = levels


def _hook_linear(counter, name, module, inputs, output):
    counter.note_call(name, type(module).__name__)
    tensor = _first_tensor(inputs)
    macs = _numel(output) * float(module.in_features)
    if module.bias is not None:
        macs += _numel(output)
    levels = counter.spike_levels(name, tensor) if tensor is not None else None
    if levels is None:
        counter.add_shape("mac_conv_linear", macs, name, type(module).__name__)
        return
    mean_spike_count = tensor.detach().float().mean() * float(levels)
    counter.add_value("sop_conv_linear", mean_spike_count * macs, name,
                      type(module).__name__)
    record = counter._record(name, type(module).__name__)
    record["dense_macs_if_ann"] = record.get("dense_macs_if_ann", 0.0) + macs
    record["spike_levels"] = levels


def _hook_norm(counter, name, module, inputs, output):
    counter.note_call(name, type(module).__name__)
    scale_and_shift = 2.0 if getattr(module, "affine", True) else 1.0
    counter.add_shape("mac_norm", _numel(output) * scale_and_shift, name,
                      type(module).__name__)


def _hook_pool(counter, name, module, inputs, output):
    counter.note_call(name, type(module).__name__)
    tensor = _first_tensor(inputs)
    if tensor is not None:
        counter.add_shape("mac_pool", _numel(tensor), name, type(module).__name__)


def _hook_activation(counter, name, module, inputs, output):
    counter.note_call(name, type(module).__name__)
    counter.add_shape("mac_elementwise", _numel(output), name, type(module).__name__)


def _hook_neuron(counter, name, module, inputs, output):
    """Spiking-neuron statistics: updates, emitted spikes and firing rate."""
    counter.note_call(name, type(module).__name__)
    counter.add_shape("neuron_updates", _numel(output), name, type(module).__name__)
    levels = float(getattr(module, "D", 1) or 1)
    spikes = output.detach().float().sum() * levels
    counter.add_value("spike_count", spikes, name, type(module).__name__)
    record = counter._record(name, type(module).__name__)
    record["elements"] = record.get("elements", 0.0) + _numel(output)
    record["spike_levels"] = int(levels)


# -- composite modules: only the arithmetic their own ``forward`` performs ---
def _hook_mdsa(counter, name, module, inputs, output):
    """MDSA: two mean reductions over C and three dense re-weightings."""
    counter.note_call(name, type(module).__name__)
    tensor = _first_tensor(inputs)
    if tensor is None:
        return
    counter.add_shape("mac_elementwise", 5.0 * _numel(tensor), name,
                      type(module).__name__)


def _hook_sdcba(counter, name, module, inputs, output):
    """SDCBA: two spike-count poolings and two binary maskings (AND)."""
    counter.note_call(name, type(module).__name__)
    tensor = _first_tensor(inputs)
    if tensor is None:
        return
    counter.add_value(
        "sop_elementwise",
        torch.as_tensor(4.0 * _numel(tensor), dtype=torch.float64,
                        device=counter.device),
        name, type(module).__name__,
    )


def _hook_arfe(counter, name, module, inputs, output):
    """ARFE: 7 element-wise float operations per feature element."""
    counter.note_call(name, type(module).__name__)
    tensor = _first_tensor(inputs)
    if tensor is None:
        return
    counter.add_shape("mac_elementwise", 7.0 * _numel(tensor), name,
                      type(module).__name__)


def _hook_temporal_fusion(counter, name, module, inputs, output):
    """Temporal fusion: one weighting and one reduction per element."""
    counter.note_call(name, type(module).__name__)
    tensor = _first_tensor(inputs)
    if tensor is None:
        return
    counter.add_shape("mac_elementwise", 2.0 * _numel(tensor), name,
                      type(module).__name__)


def _hook_upsampling(counter, name, module, inputs, output):
    """Bilinear interpolation before the spiking convolution."""
    counter.note_call(name, type(module).__name__)
    tensor = _first_tensor(inputs)
    if tensor is None:
        return
    upsampled_elements = _numel(tensor) * float(module.scale_factor) ** 2
    counter.add_shape("mac_elementwise",
                      BILINEAR_MACS_PER_OUTPUT * upsampled_elements,
                      name, type(module).__name__)


def _hook_octave_bank(counter, name, module, inputs, output):
    """(T-1) depthwise 5x5 Gaussian blurs plus (T-1) band subtractions."""
    counter.note_call(name, type(module).__name__)
    tensor = _first_tensor(inputs)
    if tensor is None:
        return
    blurs = max(int(module.T) - 1, 0)
    counter.add_shape(
        "mac_octave_bank",
        blurs * _numel(tensor) * (OCTAVE_KERNEL_TAPS + 1.0),
        name, type(module).__name__,
    )


def _hook_spectral_budget(counter, name, module, inputs, output):
    """Absolute value, mean, cumulative sum and comparison over the band stack."""
    counter.note_call(name, type(module).__name__)
    tensor = _first_tensor(inputs)
    if tensor is None:
        return
    counter.add_shape("mac_elementwise", 3.0 * _numel(tensor), name,
                      type(module).__name__)


def _hook_dsrb(counter, name, module, inputs, output):
    """Residual addition after the attention branch."""
    counter.note_call(name, type(module).__name__)
    tensor = _first_tensor(inputs)
    if tensor is None:
        return
    counter.add_shape("mac_elementwise", _numel(tensor), name, type(module).__name__)


def _hook_network(counter, name, module, inputs, output):
    """Final residual addition with the input image."""
    counter.note_call(name, type(module).__name__)
    counter.forward_calls += 1
    counter.add_shape("mac_elementwise", _numel(output), name, type(module).__name__)


# --------------------------------------------------------------------------
# Hook registry
# --------------------------------------------------------------------------
def _optional(module_path, class_name):
    """Import ``module_path.class_name`` or return ``None`` if it is absent.

    The SpectraSpike additions (NI-LIF neurons, cross-band attention, the
    Laplacian octave bank, the spectral budget) are not part of every SpikeRain
    checkout, so their hooks are registered only when the classes exist.
    """
    try:
        module = __import__(module_path, fromlist=[class_name])
    except ImportError:
        return None
    return getattr(module, class_name, None)


def _build_registry():
    """Map module classes to hooks, importing the project's own modules."""
    from model.modules import (
        ARFE, DSRB, MultiDimensionalAttention, TemporalFusion, UpSampling,
    )
    from model.spikerain import SpikeRain

    nilif_class = _optional('model.modules', 'NILIFNode')
    candidates = [
        (MultiDimensionalAttention, _hook_mdsa),
        (_optional('model.modules', 'SpikeDrivenCrossBandAttention'), _hook_sdcba),
        (ARFE, _hook_arfe),
        (TemporalFusion, _hook_temporal_fusion),
        (UpSampling, _hook_upsampling),
        (_optional('model.octave', 'LaplacianOctaveBank'), _hook_octave_bank),
        (_optional('model.budget', 'SpectralBudget'), _hook_spectral_budget),
        (DSRB, _hook_dsrb),
        (SpikeRain, _hook_network),
    ]
    composites = OrderedDict(
        (klass, hook) for klass, hook in candidates if klass is not None
    )
    return composites, nilif_class


def _neuron_classes(nilif_class):
    classes = [] if nilif_class is None else [nilif_class]
    try:
        from spikingjelly.activation_based.neuron import BaseNode
        classes.append(BaseNode)
    except Exception:  # pragma: no cover - spikingjelly layout changed
        pass
    if not classes:
        raise RuntimeError(
            "No spiking-neuron class could be imported; install spikingjelly "
            "or expose NILIFNode in model/modules.py.")
    return tuple(classes)


_NORM_CLASSES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.LayerNorm, nn.GroupNorm,
)
_POOL_CLASSES = (
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
)
_ACTIVATION_CLASSES = (
    nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.SiLU,
    nn.Softmax, nn.Hardsigmoid, nn.ELU,
)
_IGNORED_CLASSES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.Flatten, nn.Unflatten)


def attach_counters(model, counter):
    """Register every profiling hook and return the handles plus a coverage report."""
    composites, nilif_class = _build_registry()
    neuron_classes = _neuron_classes(nilif_class)
    handles = []
    hooked = set()

    def make(hook, name):
        def wrapper(module, inputs, output):
            if not torch.is_tensor(output):
                return
            hook(counter, name, module, inputs, output)
        return wrapper

    for name, module in model.named_modules():
        display = name or type(module).__name__
        hook = None
        if isinstance(module, neuron_classes):
            hook = _hook_neuron
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hook = _hook_conv
        elif isinstance(module, nn.Linear):
            hook = _hook_linear
        elif isinstance(module, _NORM_CLASSES):
            hook = _hook_norm
        elif isinstance(module, _POOL_CLASSES):
            hook = _hook_pool
        elif isinstance(module, _ACTIVATION_CLASSES):
            hook = _hook_activation
        else:
            for klass, composite_hook in composites.items():
                if type(module) is klass:
                    hook = composite_hook
                    break
        if hook is not None:
            handles.append(module.register_forward_hook(make(hook, display)))
            hooked.add(display)
            continue
        is_leaf = next(module.children(), None) is None
        if is_leaf and not _is_container(module) and not isinstance(module, _IGNORED_CLASSES):
            counter.unhandled[display] = type(module).__name__
    return handles


def detach_counters(handles):
    for handle in handles:
        handle.remove()


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
def count_parameters(model):
    trainable = 0
    non_trainable = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            non_trainable += parameter.numel()
    buffers = sum(buffer.numel() for buffer in model.buffers() if torch.is_tensor(buffer))
    total = trainable + non_trainable
    return OrderedDict([
        ("trainable", int(trainable)),
        ("non_trainable", int(non_trainable)),
        ("total", int(total)),
        ("trainable_M", trainable / 1e6),
        ("non_trainable_M", non_trainable / 1e6),
        ("total_M", total / 1e6),
        ("buffers", int(buffers)),
        ("buffers_M", buffers / 1e6),
        ("model_size_fp32_MB", total * 4 / (1024 ** 2)),
    ])


# --------------------------------------------------------------------------
# Energy
# --------------------------------------------------------------------------
def summarise_operations(totals, sign_op_mode="spike", scale=1.0):
    """Turn raw counter totals into the reported operation / energy block."""
    if scale <= 0:
        raise ValueError("scale must be positive")

    def get(key):
        return float(totals.get(key, 0.0)) / scale

    mac_conv_linear = get("mac_conv_linear")
    mac_norm = get("mac_norm")
    mac_pool = get("mac_pool")
    mac_elementwise = get("mac_elementwise")
    mac_octave = get("mac_octave_bank")
    sop_conv_linear = get("sop_conv_linear")
    sop_elementwise = get("sop_elementwise")
    neuron_updates = get("neuron_updates")
    spike_count = get("spike_count")

    mac_full = mac_conv_linear + mac_norm + mac_pool + mac_elementwise + mac_octave
    sop_full = sop_conv_linear + sop_elementwise
    if sign_op_mode not in ("spike", "neuron"):
        raise ValueError("sign_op_mode must be 'spike' or 'neuron'")
    sign_ops = spike_count if sign_op_mode == "spike" else neuron_updates
    firing_rate = (spike_count / neuron_updates) if neuron_updates > 0 else None

    def energy_uJ(macs, sops, signs):
        picojoules = (ENERGY_PER_MAC_PJ * macs
                      + ENERGY_PER_SOP_PJ * sops
                      + ENERGY_PER_SIGN_PJ * signs)
        return picojoules / 1e6

    energy_full = energy_uJ(mac_full, sop_full, sign_ops)
    energy_conv = energy_uJ(mac_conv_linear, sop_conv_linear, sign_ops)
    ann_equivalent = energy_uJ(mac_full + sop_full, 0.0, 0.0)

    return OrderedDict([
        ("MACs_G", mac_full / 1e9),
        ("FLOPs_G", 2.0 * mac_full / 1e9),
        ("SOPs_G", sop_full / 1e9),
        ("Energy_uJ", energy_full),
        ("Energy_uJ_conv_only", energy_conv),
        ("MACs_G_conv_only", mac_conv_linear / 1e9),
        ("FLOPs_G_conv_only", 2.0 * mac_conv_linear / 1e9),
        ("SOPs_G_conv_only", sop_conv_linear / 1e9),
        ("sign_ops_G", sign_ops / 1e9),
        ("spike_count_G", spike_count / 1e9),
        ("neuron_updates_G", neuron_updates / 1e9),
        ("average_firing_rate", firing_rate),
        ("sign_op_mode", sign_op_mode),
        ("mac_breakdown_G", OrderedDict([
            ("conv_linear", mac_conv_linear / 1e9),
            ("normalisation", mac_norm / 1e9),
            ("pooling", mac_pool / 1e9),
            ("elementwise", mac_elementwise / 1e9),
            ("octave_bank", mac_octave / 1e9),
        ])),
        ("sop_breakdown_G", OrderedDict([
            ("conv_linear", sop_conv_linear / 1e9),
            ("spike_elementwise", sop_elementwise / 1e9),
        ])),
        ("energy_terms_uJ", OrderedDict([
            ("MAC_term", ENERGY_PER_MAC_PJ * mac_full / 1e6),
            ("SOP_term", ENERGY_PER_SOP_PJ * sop_full / 1e6),
            ("Sign_term", ENERGY_PER_SIGN_PJ * sign_ops / 1e6),
        ])),
        ("ann_equivalent_energy_uJ", ann_equivalent),
        ("energy_saving_vs_ann_equivalent",
         (1.0 - energy_full / ann_equivalent) if ann_equivalent > 0 else None),
    ])


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------
def _synchronise(device):
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(callable_fn, device, warmup=5, runs=20, label=""):
    """Wall-clock latency in milliseconds with warm-up and CUDA synchronisation."""
    if runs < 1:
        raise ValueError("runs must be >= 1")
    for _ in range(max(warmup, 0)):
        callable_fn()
    _synchronise(device)
    samples = []
    for _ in range(runs):
        _synchronise(device)
        start = time.perf_counter()
        callable_fn()
        _synchronise(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    mean = statistics.fmean(samples) if hasattr(statistics, "fmean") else sum(samples) / len(samples)
    return OrderedDict([
        ("label", label),
        ("mean_ms", mean),
        ("std_ms", statistics.stdev(samples) if len(samples) > 1 else 0.0),
        ("median_ms", statistics.median(samples)),
        ("min_ms", min(samples)),
        ("max_ms", max(samples)),
        ("throughput_per_s", 1000.0 / mean if mean > 0 else None),
        ("warmup_iterations", int(max(warmup, 0))),
        ("timed_iterations", int(runs)),
    ])


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def list_images(data_path):
    paths = []
    for pattern in IMAGE_PATTERNS:
        paths.extend(glob(os.path.join(data_path, pattern)))
    if not paths:
        raise RuntimeError("Found 0 images in: {}".format(data_path))
    return sorted(paths)


def load_image_tensor(path, device):
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        tensor = TF.to_tensor(image)
    return tensor.unsqueeze(0).to(device)


def _round_to_multiple(size, multiple=SIZE_MULTIPLE):
    """Largest multiple of ``multiple`` not greater than ``size`` (at least one)."""
    size = int(size)
    rounded = (size // multiple) * multiple
    return max(rounded, multiple)


def _fixed_size_input(image, size):
    """Centre crop (or bilinear resize when too small) to ``size`` x ``size``.

    ``size`` is rounded down to a multiple of 4 because the encoder halves the
    resolution twice and the decoder upsamples by exactly 2x; anything else
    makes the skip connections mismatch.
    """
    size = _round_to_multiple(size)
    height, width = image.shape[-2], image.shape[-1]
    if height < size or width < size:
        return torch.nn.functional.interpolate(
            image, size=(size, size), mode="bilinear", align_corners=False)
    top = (height - size) // 2
    left = (width - size) // 2
    return image[:, :, top:top + size, left:left + size].contiguous()


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def profile_efficiency(model, image_paths, crop_size=64, overlap_size=8,
                       device=None, num_images=3, profile_size=128,
                       latency_warmup=5, latency_runs=20, sign_op_mode="spike",
                       reset_fn=None, model_config=None, checkpoint=None,
                       max_spike_levels=16, verbose=True):
    """Measure parameters, operations, energy and latency on real test images.

    Args:
        model: an already-built, already-loaded SpikeRain model on ``device``.
        image_paths: list of test-image paths (real data, not random tensors).
        crop_size / overlap_size: tiling used by ``inference_utils.tiled_restore``
            so the reported per-image cost matches how the model is actually run.
        num_images: how many test images to average the operation counts over.
        profile_size: side length of the extra fixed-resolution single-forward
            measurement used for paper tables (``None`` disables it).
        sign_op_mode: ``'spike'`` charges the Sign energy per emitted spike,
            ``'neuron'`` charges it per neuron update.

    Returns:
        A JSON-serialisable ``OrderedDict`` with ``parameters``,
        ``per_image``, ``per_tile``, ``at_resolution``, ``latency``,
        ``protocol`` and ``environment`` sections.
    """
    if reset_fn is None:
        from spikingjelly.activation_based import functional as sj_functional
        reset_fn = sj_functional.reset_net
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    if not image_paths:
        raise RuntimeError("profile_efficiency needs at least one test image")

    # The encoder halves the resolution twice, so the tile fed to the network
    # must be a multiple of 4; round up rather than fail on e.g. crop_size=50.
    crop_size = int(crop_size)
    if crop_size % SIZE_MULTIPLE:
        crop_size += SIZE_MULTIPLE - (crop_size % SIZE_MULTIPLE)
    overlap_size = int(overlap_size)
    if overlap_size >= crop_size:
        raise ValueError("overlap_size ({}) must be smaller than crop_size ({})"
                         .format(overlap_size, crop_size))

    selected = list(image_paths)[:max(int(num_images), 1)]
    was_training = model.training
    model.eval()

    counter = OperationCounter(device=device, max_spike_levels=max_spike_levels)
    handles = attach_counters(model, counter)
    tiles_total = 0.0
    pixels_total = 0.0
    band_counts_all = []
    try:
        with torch.no_grad():
            for path in selected:
                image = load_image_tensor(path, device)
                pixels_total += float(image.shape[-1] * image.shape[-2])
                before = counter.shape_totals["neuron_updates"]
                restored, band_counts = tiled_restore(
                    model, image, crop_size=crop_size, overlap_size=overlap_size,
                    reset_fn=reset_fn, collect_band_counts=True)
                del restored
                band_counts_all.extend(band_counts)
                tiles_total += _tile_count(image.shape[-2], image.shape[-1],
                                           crop_size, overlap_size)
                if before == counter.shape_totals["neuron_updates"]:
                    raise RuntimeError(
                        "No spiking activity was recorded - the profiler hooks did "
                        "not fire. Check that the model is a SpikeRain instance.")
        raw_totals = counter.totals()
        per_module = counter.module_table()
        unhandled = OrderedDict(counter.unhandled)

        images_profiled = float(len(selected))
        per_image = summarise_operations(raw_totals, sign_op_mode, scale=images_profiled)
        per_tile = summarise_operations(raw_totals, sign_op_mode,
                                        scale=max(tiles_total, 1.0))

        at_resolution = None
        if profile_size:
            counter.reset()
            with torch.no_grad():
                sample = _fixed_size_input(load_image_tensor(selected[0], device),
                                           int(profile_size))
                reset_fn(model)
                model(sample)
                reset_fn(model)
            at_resolution = summarise_operations(counter.totals(), sign_op_mode)
            at_resolution["resolution"] = "{s}x{s}".format(s=int(profile_size))
            at_resolution["note"] = (
                "Single un-tiled forward pass at a fixed resolution; use this for "
                "paper tables that quote FLOPs/SOPs at a stated input size.")
    finally:
        detach_counters(handles)
        reset_fn(model)

    # Latency is measured with the hooks removed so the profiler never inflates it.
    sample_image = load_image_tensor(selected[0], device)

    def run_tiled():
        with torch.no_grad():
            tiled_restore(model, sample_image, crop_size=crop_size,
                          overlap_size=overlap_size, reset_fn=reset_fn,
                          collect_band_counts=False)

    tile_input = _fixed_size_input(sample_image, min(crop_size,
                                                     sample_image.shape[-1],
                                                     sample_image.shape[-2]))

    def run_tile():
        with torch.no_grad():
            model(tile_input)
            reset_fn(model)

    latency = OrderedDict()
    latency["per_image_tiled"] = measure_latency(
        run_tiled, device, warmup=max(latency_warmup // 2, 1),
        runs=max(latency_runs // 4, 3),
        label="end-to-end restoration of one full test image (tiled, batch 1)")
    latency["per_tile"] = measure_latency(
        run_tile, device, warmup=latency_warmup, runs=latency_runs,
        label="single {s}x{s} tile forward pass".format(s=tile_input.shape[-1]))
    if profile_size:
        fixed_input = _fixed_size_input(sample_image, int(profile_size))

        def run_fixed():
            with torch.no_grad():
                model(fixed_input)
                reset_fn(model)

        latency["at_resolution"] = measure_latency(
            run_fixed, device, warmup=latency_warmup, runs=latency_runs,
            label="single {s}x{s} forward pass".format(s=int(profile_size)))
    reset_fn(model)
    if was_training:
        model.train()

    report = OrderedDict()
    report["created_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["checkpoint"] = str(checkpoint) if checkpoint else None
    report["model_config"] = OrderedDict(model_config or {})
    report["parameters"] = count_parameters(model)
    report["per_image"] = per_image
    report["per_tile"] = per_tile
    report["at_resolution"] = at_resolution
    report["latency"] = latency
    report["measurement"] = OrderedDict([
        ("images_profiled", int(images_profiled)),
        ("image_paths", [os.path.basename(path) for path in selected]),
        ("tiles_per_image", tiles_total / images_profiled),
        ("pixels_per_image", pixels_total / images_profiled),
        ("crop_size", int(crop_size)),
        ("overlap_size", int(overlap_size)),
        ("average_bands_per_tile",
         (sum(band_counts_all) / len(band_counts_all)) if band_counts_all else None),
    ])
    report["protocol"] = OrderedDict([
        ("energy_reference", ENERGY_REFERENCE),
        ("E_MAC_pJ", ENERGY_PER_MAC_PJ),
        ("E_SOP_pJ", ENERGY_PER_SOP_PJ),
        ("E_SIGN_pJ", ENERGY_PER_SIGN_PJ),
        ("formula", "E = E_MAC * MACs_ANN + E_SOP * SOPs + E_SIGN * SignOps"),
        ("sign_op_mode", sign_op_mode),
        ("mac_to_flop_convention", "1 MAC = 2 FLOPs; energy is charged per MAC"),
        ("spike_detection",
         "A tensor is treated as spikes when every value lies on the grid "
         "0, 1/D, 2/D, ..., 1 for an integer D <= "
         + str(max_spike_levels) + "; D is measured, not assumed."),
        ("sop_formula",
         "SOP_l = MAC_l * measured mean spike count per input element"),
        ("accounting_modes",
         "'full' charges every measured operation; 'conv_only' charges "
         "convolution/linear operations plus the Sign term, matching the "
         "convention of published SNN deraining tables."),
        ("bilinear_macs_per_output", BILINEAR_MACS_PER_OUTPUT),
        ("octave_kernel_taps", OCTAVE_KERNEL_TAPS),
        ("precision", "float32"),
    ])
    report["environment"] = OrderedDict([
        ("device", str(device)),
        ("gpu_name", torch.cuda.get_device_name(device)
         if device.type == "cuda" and torch.cuda.is_available() else None),
        ("torch_version", torch.__version__),
        ("cuda_version", torch.version.cuda),
        ("cudnn_benchmark", bool(torch.backends.cudnn.benchmark)),
        ("platform", platform.platform()),
    ])
    report["per_module_totals"] = per_module
    report["unhandled_modules"] = unhandled
    if unhandled and verbose:
        print("[model_complexity] WARNING: no operation counter for: {}".format(
            ", ".join("{} ({})".format(k, v) for k, v in unhandled.items())))
    return report


def _tile_count(height, width, crop_size, overlap_size):
    """Tiles ``inference_utils.tiled_restore`` runs for an image of this size."""
    return tile_count(height, width, crop_size, overlap_size)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _format(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf"
    if isinstance(value, float):
        return "{:.{d}f}".format(value, d=digits)
    return str(value)


def print_efficiency_summary(report):
    """Print the headline efficiency table requested for the paper."""
    parameters = report["parameters"]
    width = 74
    print("\nModel complexity, energy and latency (measured)")
    print("=" * width)
    print("{:<34}{:>16}  {:<22}".format("Metric", "Value", "Unit / note"))
    print("-" * width)
    print("{:<34}{:>16,}  {:<22}".format(
        "Trainable params", parameters["trainable"],
        "{:.4f} M".format(parameters["trainable_M"])))
    print("{:<34}{:>16,}  {:<22}".format(
        "Non-trainable params", parameters["non_trainable"],
        "{:.4f} M".format(parameters["non_trainable_M"])))
    print("{:<34}{:>16,}  {:<22}".format(
        "Total params", parameters["total"],
        "{:.4f} M".format(parameters["total_M"])))
    print("{:<34}{:>16,}  {:<22}".format(
        "Non-learnable buffers", parameters["buffers"],
        "{:.4f} M".format(parameters["buffers_M"])))
    print("-" * width)

    blocks = [("Per test image (tiled inference)", report.get("per_image"))]
    if report.get("at_resolution"):
        blocks.append(("At {} (single forward)".format(
            report["at_resolution"].get("resolution", "fixed size")),
            report["at_resolution"]))
    blocks.append(("Per {}x{} tile".format(
        report["measurement"]["crop_size"], report["measurement"]["crop_size"]),
        report.get("per_tile")))

    for title, block in blocks:
        if not block:
            continue
        print(title)
        print("{:<34}{:>16}  {:<22}".format(
            "  FLOPs", _format(block["FLOPs_G"]), "G (= 2 x MACs)"))
        print("{:<34}{:>16}  {:<22}".format(
            "  MAC", _format(block["MACs_G"]), "G"))
        print("{:<34}{:>16}  {:<22}".format(
            "  SOP", _format(block["SOPs_G"]), "G"))
        print("{:<34}{:>16}  {:<22}".format(
            "  Energy", _format(block["Energy_uJ"]), "uJ (full accounting)"))
        print("{:<34}{:>16}  {:<22}".format(
            "  Energy (conv/linear only)", _format(block["Energy_uJ_conv_only"]),
            "uJ (paper convention)"))
        print("{:<34}{:>16}  {:<22}".format(
            "  Average firing rate", _format(block["average_firing_rate"]),
            "spikes / neuron / step"))
        print("-" * width)

    latency = report.get("latency", {})
    for key, title in (("per_image_tiled", "Latency (per image, end-to-end)"),
                       ("at_resolution", "Latency (fixed-size forward)"),
                       ("per_tile", "Latency (per tile)")):
        block = latency.get(key)
        if not block:
            continue
        print("{:<34}{:>16}  {:<22}".format(
            title, _format(block["mean_ms"], 3),
            "ms +/- {:.3f}".format(block["std_ms"])))
    print("-" * width)
    measurement = report["measurement"]
    print("Measured on {} image(s), {:.1f} tiles/image, {:.0f} px/image; "
          "{} runs timed.".format(
              measurement["images_profiled"], measurement["tiles_per_image"],
              measurement["pixels_per_image"],
              latency.get("per_tile", {}).get("timed_iterations", 0)))
    if measurement.get("average_bands_per_tile") is not None:
        print("Adaptive spectral budget: {:.2f} bands used per tile on average."
              .format(measurement["average_bands_per_tile"]))
    environment = report.get("environment", {})
    print("Device: {} | torch {} | CUDA {}".format(
        environment.get("gpu_name") or environment.get("device"),
        environment.get("torch_version"), environment.get("cuda_version")))
    print("Energy protocol: E_MAC={} pJ, E_SOP={} pJ, E_SIGN={} pJ (45 nm; "
          "Chen et al., AAAI 2026).".format(
              ENERGY_PER_MAC_PJ, ENERGY_PER_SOP_PJ, ENERGY_PER_SIGN_PJ))
    print("=" * width)


def _json_default(value):
    """Make numpy / torch scalars JSON-serialisable."""
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    for attribute in ('item', 'tolist'):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return str(value)


def save_efficiency_report(report, destination):
    """Write ``report`` to ``destination`` as JSON and return the path."""
    destination = os.fspath(destination)
    folder = os.path.dirname(os.path.abspath(destination))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(destination, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, default=_json_default)
    return destination


# --------------------------------------------------------------------------
# Command line interface
# --------------------------------------------------------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Measure parameters, FLOPs, MACs, SOPs, energy and latency "
                    "for a SpikeRain / SpectraSpike checkpoint.")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Folder of real test images used for the measurement')
    parser.add_argument('--model_version', type=str, default='M',
                        choices=['S', 'M', 'L', 's', 'm', 'l'])
    parser.add_argument('--weights', type=str, default=None,
                        help='Model checkpoint (.pth). Omit to profile an '
                             'untrained model of the same configuration.')
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--temporal_mode', type=str, default='replicate',
                        choices=['replicate', 'octave'])
    parser.add_argument('--octave_learnable', action='store_true')
    parser.add_argument('--attention_mode', type=str, default='mdsa',
                        choices=['mdsa', 'sdcba', 'none'])
    parser.add_argument('--head_mode', type=str, default='binary',
                        choices=['binary', 'integer'])
    parser.add_argument('--head_D', type=int, default=4)
    parser.add_argument('--adaptive_budget', action='store_true')
    parser.add_argument('--budget_threshold', type=float, default=0.9)
    parser.add_argument('--budget_min_bands', type=int, default=2)
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='CUDA device ids; -1 for CPU')
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--overlap_size', type=int, default=8)
    parser.add_argument('--num_images', type=int, default=3,
                        help='Test images to average the operation counts over')
    parser.add_argument('--profile_size', type=int, default=128,
                        help='Fixed resolution for the paper-table forward pass '
                             '(0 disables it)')
    parser.add_argument('--latency_warmup', type=int, default=5)
    parser.add_argument('--latency_runs', type=int, default=20)
    parser.add_argument('--sign_op_mode', type=str, default='spike',
                        choices=['spike', 'neuron'])
    parser.add_argument('--json_out', type=str, default=None,
                        help='Where to write the JSON report. Defaults to '
                             '<checkpoint>.complexity.json, or '
                             '<data_path>/complexity.json without a checkpoint.')
    return parser


def load_checkpoint_into(model, weights):
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith('module.') else key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print('Missing keys:', missing)
    if unexpected:
        print('Unexpected keys:', unexpected)


def _supported_factory_kwargs(factory_class, requested, defaults):
    """Keep only the kwargs ``factory_class.__init__`` actually accepts.

    The released SpikeRainFactory takes ``(T, device)``. The SpectraSpike
    variants add temporal/attention/budget switches. Passing the full set to
    either one would be a TypeError, so the signature decides, and any
    non-default flag that the factory cannot honour is reported instead of
    being silently ignored.
    """
    try:
        signature = inspect.signature(factory_class.__init__)
    except (TypeError, ValueError):  # pragma: no cover - exotic factories
        return dict(requested), []
    accepts_everything = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values())
    if accepts_everything:
        return dict(requested), []
    accepted = set(signature.parameters)
    supported = {key: value for key, value in requested.items() if key in accepted}
    ignored = [key for key, value in requested.items()
               if key not in accepted and value != defaults.get(key)]
    return supported, sorted(ignored)


def main():
    options = build_arg_parser().parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if options.gpu_ids.strip() != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_ids

    from model import SpikeRainFactory
    from spikingjelly.activation_based import functional

    wants_cuda = options.gpu_ids.strip() != '-1'
    use_cuda = wants_cuda and torch.cuda.is_available()
    if wants_cuda and not use_cuda:
        print('CUDA was requested but is not available; falling back to CPU. '
              'Latency numbers from a CPU run are not comparable with the '
              'GPU numbers reported in the paper.')
    device = torch.device('cuda' if use_cuda else 'cpu')

    model_kwargs = dict(temporal_mode=options.temporal_mode,
                        octave_learnable=options.octave_learnable,
                        attention_mode=options.attention_mode,
                        head_mode=options.head_mode,
                        head_D=options.head_D,
                        adaptive_budget=options.adaptive_budget,
                        budget_threshold=options.budget_threshold,
                        budget_min_bands=options.budget_min_bands)
    factory_defaults = dict(temporal_mode='replicate', octave_learnable=False,
                            attention_mode='mdsa', head_mode='binary', head_D=4,
                            adaptive_budget=False, budget_threshold=0.9,
                            budget_min_bands=2)
    factory_kwargs, ignored = _supported_factory_kwargs(
        SpikeRainFactory, model_kwargs, factory_defaults)
    if ignored:
        print('This SpikeRainFactory does not accept {}; those options were '
              'ignored.'.format(', '.join('--' + key for key in ignored)))
    model_kwargs = factory_kwargs

    factory = SpikeRainFactory(options.T, device=str(device), **factory_kwargs)
    model = factory.get_model(options.model_version).to(device)
    functional.set_step_mode(model, step_mode='m')
    functional.set_backend(model, backend='torch')
    if options.weights:
        print('Loading checkpoint:', options.weights)
        load_checkpoint_into(model, options.weights)
    else:
        print('No --weights given: profiling an untrained model of this '
              'configuration (parameter/operation counts are unaffected).')
    model.eval()

    image_paths = list_images(options.data_path)
    print('Found {} images in {}'.format(len(image_paths), options.data_path))

    configuration = OrderedDict(model_kwargs)
    configuration['model_version'] = options.model_version.upper()
    configuration['T'] = options.T

    report = profile_efficiency(
        model, image_paths,
        crop_size=options.crop_size, overlap_size=options.overlap_size,
        device=device, num_images=options.num_images,
        profile_size=options.profile_size or None,
        latency_warmup=options.latency_warmup, latency_runs=options.latency_runs,
        sign_op_mode=options.sign_op_mode,
        reset_fn=functional.reset_net,
        model_config=configuration, checkpoint=options.weights,
        max_spike_levels=max(8, min(64, options.head_D * 4)),
    )
    print_efficiency_summary(report)

    destination = options.json_out
    if destination is None:
        if options.weights:
            destination = os.path.splitext(options.weights)[0] + '.complexity.json'
        else:
            destination = os.path.join(options.data_path, 'complexity.json')
    written = save_efficiency_report(report, destination)
    print('===> Saved complexity report:', written)


if __name__ == '__main__':
    main()
