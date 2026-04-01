# ViPE Batch Inference

> For the original project README (paper, dataset, acknowledgments, citation), see [README_ORIGINAL.md](README_ORIGINAL.md).

This document describes the **code framework** — how to configure inputs, select pipelines, run inference, and process videos at scale.

---

## Installation

```bash
conda env create -f envs/base.yml
conda activate vipe
pip install -r envs/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation -e .
```

---

## Quick Start

```bash
# Single video or directory of mp4 files
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR

# Disable depth estimation (pose only)
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR \
    pipeline.post.depth_align_model=null

# Multi-GPU batch processing
python run_batch.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR
```

Results are written to `vipe_results/` by default (configurable via `pipeline.output.path`).

---

## Input Streams

Configured via `streams=<name>` on the command line. Config files live in `configs/streams/`.

### `raw_mp4_stream` — MP4 video input

```yaml
# configs/streams/raw_mp4_stream.yaml
instance: vipe.streams.raw_mp4_stream.RawMP4StreamList

base_path: ??        # required: path to an mp4 file, a directory of mp4s, or a .txt file
frame_start: 0
frame_end: 1000
frame_rate: 20       # target FPS; the stream is downsampled to this rate
cached: false
```

`base_path` accepts three formats:

| Format | Behaviour |
|--------|-----------|
| `/path/to/video.mp4` | Single video |
| `/path/to/dir/` | All `*.mp4` files in the directory (sorted) |
| `/path/to/list.txt` | One mp4 path per line |

### `frame_dir_stream` — Image directory input

```yaml
# configs/streams/frame_dir_stream.yaml
instance: vipe.streams.frame_dir_stream.FrameDirStreamList

base_path: ??
frame_start: 0
frame_end: -1
frame_skip: 1
cached: false
```

---

## Pipelines

Configured via `pipeline=<name>`. Config files live in `configs/pipeline/`.

| Name | Description |
|------|-------------|
| `default` | Standard pipeline for pinhole cameras. Depth via UniDepth + VideoDepthAnything (SVDA). |
| `no_vda` | Skips VideoDepthAnything; lower memory usage, less temporally stable depth. |
| `dav3` | Uses Depth-Anything-V3 for both keyframe depth and post-processing. |
| `lyra` | Configuration used in the [Lyra](https://github.com/nv-tlabs/lyra) paper (MoGe + VDA). |
| `wide_angle` | For wide-angle / fisheye cameras (Mei camera model, no depth post-processing). |
| `static_vda` | Disables instance segmentation; uses full VideoDepthAnything (VDA). |

### Key pipeline options (overridable on the command line)

```bash
# Change output directory
pipeline.output.path=my_results/

# Skip videos that already have results
pipeline.output.skip_exists=true   # default: true

# Save raw artifacts (poses, depth, intrinsics, masks, …)
pipeline.output.save_artifacts=true

# Save SLAM sparse map
pipeline.output.save_slam_map=true

# Disable depth post-processing
pipeline.post.depth_align_model=null

# Change camera model (pinhole / mei)
pipeline.init.camera_type=mei
```

### Output layout

```
vipe_results/
├── intrinsics/
│   ├── <name>.npz              # fx, fy, cx, cy per frame
│   └── <name>_camera.txt       # camera model identifier
├── pose/
│   └── <name>.npz              # c2w poses (OpenCV convention)
├── depth/
│   └── <name>.zip              # per-frame depth maps
├── mask/
│   ├── <name>.zip              # instance segmentation masks
│   └── <name>.txt              # instance id → phrase mapping
├── rgb/
│   └── <name>.mp4              # downsampled RGB video
└── vipe/
    ├── <name>_info.pkl         # bundle-adjustment residual
    ├── <name>_vis.mp4          # visualisation video
    └── <name>_slam_map.pt      # SLAM sparse map (if enabled)
```

---

## Batch Processing (`run_batch.py`)

`run_batch.py` distributes work across all available GPUs using Python `multiprocessing`.

```bash
python run_batch.py pipeline=default streams=raw_mp4_stream \
    streams.base_path=YOUR_VIDEO_OR_DIR \
    workers_per_gpu=1
```

**How it works:**

1. The main process scans the input path once and builds the full video list.
2. Videos whose `intrinsics/<name>_camera.txt` already exists are skipped.
3. Remaining videos are pushed into a shared task queue.
4. One or more worker processes are spawned per GPU (`workers_per_gpu`, default `1`). Each worker builds the pipeline once and processes tasks until the queue is empty.

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `workers_per_gpu` | `1` | Worker processes per GPU |
| `streams.base_path` | required | Video source (mp4 / directory / txt) |
| `pipeline.output.path` | `vipe_results/` | Output directory |

---

## Converting to COLMAP Format

```bash
python scripts/vipe_to_colmap.py vipe_results/ --sequence <name>

# For a sparser but more 3D-consistent point cloud (requires save_slam_map=true):
python scripts/vipe_to_colmap.py vipe_results/ --sequence <name> --use_slam_map
```

---

## Visualisation

```bash
vipe visualize vipe_results/
```

---

## Configuration System

ViPE uses [Hydra](https://hydra.cc/) for configuration. Any config field can be overridden directly on the command line:

```bash
python run.py pipeline=default \
    streams=raw_mp4_stream \
    streams.base_path=/data/videos/ \
    streams.frame_rate=10 \
    streams.frame_end=500 \
    pipeline.output.path=my_results/ \
    pipeline.output.save_artifacts=true \
    pipeline.post.depth_align_model=null
```

The config search path is `configs/`. The top-level config is `configs/default.yaml`.

---

## Acknowledgments

This project builds on the official ViPE repository. We thank the original authors for their open-source contribution:  
https://github.com/nv-tlabs/vipe
