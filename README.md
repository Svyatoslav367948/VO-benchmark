# VO-benchmark

A small Visual Odometry (VO) benchmarking toolkit for evaluating and comparing VO methods on standard datasets.

## Original methods used in this benchmark

- LEAP-VO https://github.com/wrchen530/leapvo.git
- CoProU-VO https://github.com/Jchao-Xie/CoProU.git
- DytanVO https://github.com/castacks/DytanVO?ysclid=mto4znjnjf1540868

## Overview

VO-benchmark provides utilities to run VO methods on datasets, compute standard evaluation metrics (ATE, RPE, translation and rotation errors), and visualize results.

## Features

- Run benchmarks for multiple VO methods.
- Compute ATE (Absolute Trajectory Error) and RPE (Relative Pose Error).
- Produce plots and result tables for easy comparison.
- Support for common datasets (KITTI, TUM RGB-D, EuRoC) — add dataset-specific adapters as needed.

## Typical repository layout

This is a suggested layout. Adjust to match the real repo structure.

- datasets/        # scripts or instructions to download & prepare datasets
- src/             # core implementation and benchmark runner
- methods/         # wrappers for VO/SLAM methods (e.g. ORB-SLAM2, DSO)
- experiments/     # experiment configs and run scripts
- results/         # generated results and logs
- notebooks/       # optional Jupyter notebooks for visualization
- requirements.txt # Python dependencies (if present)
- README.md        # this file

## Requirements

- Python 3.8+ (3.10 recommended)
- pip

Common Python packages used in VO benchmarking (add exact versions to requirements.txt):

- numpy
- scipy
- opencv-python
- matplotlib
- pandas
- transforms3d or pyquaternion
- tqdm
- yaml

Optional tools:

- evo (for trajectory evaluation and visualization): `pip install evo`

## Quickstart

Adjust the commands below to the actual script names in this repository.

1. Prepare dataset(s). Example for KITTI:

```bash
# download and unpack KITTI sequences into datasets/kitti/
# run any dataset preprocessing scripts if available
```

2. Run a benchmark (example placeholder):

```bash
python src/run_benchmark.py --method ORB-SLAM2 --dataset datasets/kitti --sequences 00 02 --output results/ORB-ORB
```

3. Evaluate results and visualize:

```bash
python src/evaluate.py --pred results/ORB-ORB/trajectory.txt --gt datasets/kitti/poses/00.txt --metrics ATE RPE
python src/plot_results.py --input results/ --out figures/
```

If your repository uses a different CLI or config format (YAML/JSON), update these examples accordingly.

## Configuration

Use a config file (YAML/JSON) to define experiments. Example YAML structure:

```yaml
method: dytanvo
dataset: datasets/kitti
sequences: [00, 02]
params:
  orb:
    n_features: 2000
  camera:
    fps: 10
```

## Supported datasets (suggested)

- KITTI odometry
- TUM RGB-D
- EuRoC MAV

Provide dataset adapters in `src/datasets/` that convert dataset ground-truth and poses into the repository's canonical format.

## Evaluation metrics

Common metrics implemented or recommended:

- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE) — translational and rotational components
- Translation / rotation RMSE
- Scale drift and mean endpoint error

The `evo` toolkit is commonly used for these metrics: https://github.com/MichaelGrupp/evo

## Results & Visualization

Save per-run outputs under `results/<method>/<dataset>/<sequence>/`. Include:

- trajectory files (timestamped poses)
- per-frame metrics/log
- summary.json or summary.csv with aggregated metrics
- plots and figures

Provide helper scripts to aggregate result tables for multiple methods.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository
2. Create a feature branch
3. Add tests and documentation for your changes
4. Open a pull request describing the change

Please add issue templates and a CONTRIBUTING.md if you want more formal contribution guidelines.

## License

No license file is present in the repository metadata. If you want this project to be open source, add a LICENSE (for example, MIT) and update this section.

## Contact

If you need help tailoring this README to the actual repository contents, tell me which files or folders exist and I will update the instructions and quickstart commands to match.

## Acknowledgement
 
We appreciate the contributions of the following projects, which have greatly supported our work:

* [SfMLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch) - A pioneering framework for end-to-end monocular visual odometry.

* [SC-Depth](https://github.com/JiawangBian/sc_depth_pl) - Our baseline.
 
* [Kitti-Odom-Eval-Python](https://github.com/Huangying-Zhan/kitti-odom-eval) - Python implementation for KITTI odometry evaluation.

* [RoGS](https://github.com/fzhiheng/RoGS) - Preprocessing code for the nuScenes dataset.

* [DepthAnything-v2](https://github.com/DepthAnything/Depth-Anything-V2) and [DINOv2](https://github.com/facebookresearch/dinov2) – Providing Vision Transformer backbone features.
