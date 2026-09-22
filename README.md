<h4 align="center"><strong><a href="https://wacv.thecvf.com/">Accepted at WACV 2026, Tucson, Arizona, USA</a></strong></h4>
<h2 align="center"><strong>SpikeRain: Towards Energy-Efficient Single Image Deraining with Spiking Neural Networks <a href="https://openaccess.thecvf.com/content/WACV2026/html/Islam_SpikeRain_Towards_Energy-Efficient_Single_Image_Deraining_with_Spiking_Neural_Networks_WACV_2026_paper.html" target="_blank">[Paper]</a></strong></h2>
<h6 align="center">Md Tanvir Islam<sup> 1</sup>, Inzamamul Alam<sup> 2</sup>, Sambit Bakshi<sup> 3</sup>, Khan Muhammad<sup> 2, *</sup>, Javier Del Ser<sup> 4</sup>, Sangtae Ahn<sup> 1, *</sup></h6>
<h6 align="center">| 1. Kyungpook National University, South Korea | 2. Sungkyunkwan University, South Korea | 3. National Institute of Technology, India | 4. University of the Basque Country, Spain || *Corresponding Authors |</h6> 
<hr>


## SpikeRain Architecture
![](./assets/figures/SpikeRain.jpg)


## Repository Structure
```
assets/
  figures/                 # Figures used in the paper/README
  paper/                   # Accepted paper PDF
model/
  modules.py               # Core blocks: DSRB, MDSA, Temporal Fusion, ARFE
  spikerain.py             # SpikeRain model definition and factory
utils/                     # Utility helpers (metrics, model utils, etc.)
dataset_loader.py          # Training dataset loader
train.py                   # Training script
test.py                    # Inference/testing script
evaluation.py              # PSNR/SSIM/LPIPS evaluation
inference_utils.py         # Tiled inference (split / merge) shared by test.py and the profiler
model_complexity.py        # Params / FLOPs / MACs / SOPs / energy / latency profiler
requirements.txt           # Python dependencies
```

## Environment Setup

### Conda
```bash
conda create -n spikerain python=3.8 -y
conda activate spikerain
pip install -r requirements.txt
```

## Dataset Preparation
The repository expects paired rainy/clean images stored in per-dataset folders. A recommended layout is:
```
data/
  Rain200H/
    train/
      input/
      gt/
    test/
      input/
      gt/
```

**Naming conventions:**
- Input images are the rainy observations in `input/`.
- Ground-truth images are the clean targets in `gt/`.
- Supported image formats include `.png` and `.jpg`.

**Important:** `dataset_loader.py` expects the clean folder to be named `target/` for training and validation. You can either:
- Rename `gt/` → `target/`, or
- Create a symlink `target` pointing to `gt`.

**RW-Data:** This dataset has no ground truth. The provided `evaluation.py` requires paired targets, so for RW-Data use qualitative inspection or external no-reference metrics.

## Training
The training script already exposes arguments via `argparse`. Example:
```bash
python train.py \
  --train_dir ./data/Rain200H/train \
  --val_dir ./data/Rain200H/test \
  --model_save_dir ./checkpoints \
  --version M \
  --T 4
```

Common arguments:
- `--train_dir`: path to training set root (contains `input/` and `target/`).
- `--val_dir`: path to validation/test set root (contains `input/` and `target/`).
- `--model_save_dir`: output checkpoint directory.
- `--version`: model size variant (`S`, `M`, `L`).
- `--T`: number of spiking timesteps.

## Testing / Inference
```bash
python test.py \
  --weights ./checkpoints/SpikeRain_M/models/<session>/model_best.pth \
  --data_path ./data/Rain200H/test/input \
  --save_path ./results/Rain200H
```

Key arguments:
- `--data_path`: directory containing input rainy images.
- `--save_path`: output directory for restored images.
- `--model_version`: model size variant (`S`, `M`, `L`).
- `--T`: number of spiking timesteps.

## Complexity, Energy and Latency

`model_complexity.py` measures the efficiency numbers reported in the paper on a
real checkpoint with real test images: parameter counts from the instantiated
model, operation counts from forward hooks that see the actual tensor shapes,
and wall-clock latency with CUDA synchronisation and warm-up.

Whether a convolution is spike-driven (accumulate-only, SOPs) or a dense ANN
convolution (multiply-accumulate, MACs) is decided by inspecting the values of
its input tensor at run time, so no hand-maintained layer list can go stale.

```bash
python model_complexity.py \
  --data_path ./data/Rain200H/test/input \
  --model_version M \
  --weights ./checkpoints/SpikeRain_M/models/<session>/model_best.pth
```

Or fold it into a normal test run with `--profile`:

```bash
python test.py \
  --weights ./checkpoints/SpikeRain_M/models/<session>/model_best.pth \
  --data_path ./data/Rain200H/test/input \
  --save_path ./results/Rain200H \
  --profile
```

Key arguments (`model_complexity.py`; under `test.py --profile` the first one is
`--profile_images` and the last is `--complexity_json`):
- `--num_images`: test images the operation counts are averaged over (default 3).
- `--profile_size`: fixed resolution for the paper-table forward pass (default 128, `0` disables).
- `--sign_op_mode`: charge the Sign energy per emitted spike (`spike`, default) or per neuron update (`neuron`).
- `--json_out`: where to write the JSON report (defaults to `<checkpoint>.complexity.json`).

Energy follows the 45 nm model used by the SNN deraining literature
(E_MAC = 12.5 pJ, E_SOP = 77 fJ, E_SIGN = 3.7 pJ), reported both under `full`
accounting (every measured operation) and `conv_only` accounting (convolution /
linear plus the Sign term), which is the convention of the published comparison
tables. Any leaf module without an operation counter is listed under
`unhandled_modules` in the JSON report, so nothing is silently uncounted.

## Evaluation
```bash
python evaluation.py \
  --generated_images_path ./results/Rain200H \
  --target_path ./data/Rain200H/test/gt
```

`evaluation.py` computes PSNR, SSIM, and LPIPS. For datasets without ground truth (e.g., RW-Data), skip this script or use no-reference metrics.

## Reproducibility
The training script sets random seeds for Python, NumPy, and PyTorch. It also enables `torch.backends.cudnn.benchmark = True`, which favors performance over strict determinism.

## Citation
If you find our work useful in your research, please consider citing our paper and star ✨✨ this repository. Thank you!

```bibtex
@InProceedings{Islam_2026_WACV,
    author    = {Islam, Md Tanvir and Alam, Inzamamul and Bakshi, Sambit and Muhammad, Khan and Del Ser, Javier and Ahn, Sangtae},
    title     = {SpikeRain: Towards Energy-Efficient Single Image Deraining with Spiking Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {1094-1105}
}
```

## Acknowledgement 
We would like to thank the authors of [ESDNet](https://github.com/MingTian99/ESDNet) for introducing SNN-based deraining.

