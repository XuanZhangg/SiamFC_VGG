Contributions by the project team:
vggadam.py: based on siamfc.py, tweaks on the parts related to the model architecture to try different ones
siamfc.py: implemented mean shift (functions mean_shift, center_of_mass, mean_shift_kernel, and kernel_CoM);
implemented non-parametric density estimation as an alternative way of mode seeking;
loc is the result by the baseline approach, loc1 is the result by mean shift, loc2 is the result by non-parametric density estimation.

Original readme:
## Dependencies

Install PyTorch, opencv-python and GOT-10k toolkit:

```bash
pip install torch opencv-python got10k
```

## Running the tracker

In the root directory of `siamfc`:

1. Download pretrained `model.pth` from [Baidu Yun](https://pan.baidu.com/s/1TT7ebFho63Lw2D7CXLqwjQ) or [Google Drive](https://drive.google.com/open?id=1Qu5K8bQhRAiexKdnwzs39lOko3uWxEKm), and put the file under `pretrained/siamfc`.

2. Create a symbolic link `data` to your datasets folder (e.g., `data/OTB`, `data/UAV123`, `data/GOT-10k`):

```
ln -s ./data /path/to/your/data/folder
```

3. Run:

```
python test.py
```

By default, the tracking experiments will be executed and evaluated over all 7 datasets. Comment lines in `run_tracker.py` as you wish if you need to skip some experiments.

## Training the tracker

1. Assume the GOT-10k dataset is located at `data/GOT-10K`.

2. Run:

```
python train.py
```
