import multiprocessing as mp
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def worker_process(
    gpu_id: int,
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    mp4_paths: list,
    streams_cfg: DictConfig,
    pipeline_cfg: DictConfig,
) -> None:
    """
    Worker process bound to a single GPU. Pulls tasks from the queue and runs the pipeline.

    Args:
        gpu_id:       GPU index to bind this process to
        worker_id:    worker index (for logging)
        task_queue:   queue of stream indices to process
        result_queue: queue for (stream_idx, success, error_msg) results
        mp4_paths:    pre-scanned list of mp4 paths from the main process
        streams_cfg:  streams config (frame_start / frame_end / frame_rate / cached)
        pipeline_cfg: pipeline config
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from pathlib import Path

    from vipe.pipeline import make_pipeline
    from vipe.streams.raw_mp4_stream import RawMP4StreamList
    from vipe.utils.logging import configure_logging

    logger = configure_logging()
    logger.info(f"[GPU {gpu_id} | Worker {worker_id}] started")

    # Build stream list directly from the pre-scanned paths to avoid redundant filesystem scans.
    stream_list = RawMP4StreamList.from_paths(
        mp4_paths=[Path(p) for p in mp4_paths],
        frame_start=streams_cfg.frame_start,
        frame_end=streams_cfg.frame_end,
        frame_rate=streams_cfg.frame_rate,
        cached=streams_cfg.get("cached", False),
    )

    # Build pipeline once so model weights are reused across all tasks in this worker.
    pipeline = make_pipeline(pipeline_cfg)

    while not task_queue.empty():
        try:
            stream_idx = task_queue.get(timeout=5)
        except Exception:
            break

        video_stream = stream_list[stream_idx]
        logger.info(
            f"[GPU {gpu_id} | Worker {worker_id}] "
            f"processing {video_stream.name()} (idx={stream_idx}), "
            f"{task_queue.qsize()} remaining"
        )

        try:
            pipeline.run(video_stream)
            logger.info(f"[GPU {gpu_id} | Worker {worker_id}] finished {video_stream.name()}")
            result_queue.put((stream_idx, True, None))
        except Exception as e:
            logger.error(
                f"[GPU {gpu_id} | Worker {worker_id}] "
                f"failed on {video_stream.name()}: {e}"
            )
            result_queue.put((stream_idx, False, str(e)))


def get_available_gpus() -> list[int]:
    """Detect and return all available GPU indices."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs available.")

    gpu_count = torch.cuda.device_count()
    gpus = list(range(gpu_count))

    print(f"Found {gpu_count} GPU(s):")
    for gpu_id in gpus:
        name = torch.cuda.get_device_name(gpu_id)
        memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        print(f"  GPU {gpu_id}: {name} ({memory:.1f} GB)")

    return gpus


def run_parallel(args: DictConfig, workers_per_gpu: int = 1) -> None:
    """
    Main entry point for multi-GPU parallel processing.

    Args:
        args:            Hydra config
        workers_per_gpu: number of worker processes per GPU
    """
    from vipe.streams.base import StreamList
    from vipe.utils.logging import configure_logging

    logger = configure_logging()

    gpu_ids = get_available_gpus()
    total_workers = len(gpu_ids) * workers_per_gpu
    logger.info(f"{len(gpu_ids)} GPU(s), {workers_per_gpu} worker(s) each, {total_workers} total workers")

    # Scan paths once in the main process; workers reuse the result via from_paths.
    stream_list = StreamList.make(args.streams)
    total_streams = len(stream_list)
    mp4_paths = [str(stream_list.mp4_sequences[i]) for i in range(total_streams)]
    logger.info(f"Found {total_streams} video(s)")

    task_queue = mp.Queue()
    skipped = 0
    for stream_idx in tqdm(range(total_streams), desc="Filtering processed videos"):
        stream_name = stream_list.stream_name(stream_idx).split(".")[0]
        if os.path.exists(f"vipe_results/intrinsics/{stream_name}_camera.txt"):
            skipped += 1
        else:
            task_queue.put(stream_idx)

    pending = task_queue.qsize()
    logger.info(f"Skipped {skipped}, pending {pending}")

    if pending == 0:
        logger.info("All videos already processed, exiting.")
        return

    result_queue = mp.Queue()

    processes = []
    for gpu_id in gpu_ids:
        for local_worker_id in range(workers_per_gpu):
            worker_id = gpu_id * workers_per_gpu + local_worker_id
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, worker_id, task_queue, result_queue,
                      mp4_paths, args.streams, args.pipeline),
                name=f"GPU{gpu_id}-Worker{local_worker_id}",
            )
            p.start()
            processes.append(p)
            logger.info(f"Started {p.name} (PID={p.pid})")

    for p in processes:
        p.join()

    success_count = 0
    failed_streams = []

    while not result_queue.empty():
        stream_idx, success, error_msg = result_queue.get()
        if success:
            success_count += 1
        else:
            failed_streams.append((stream_idx, error_msg))

    logger.info("=" * 60)
    logger.info(f"Succeeded: {success_count} / {pending}")

    if failed_streams:
        logger.warning(f"Failed: {len(failed_streams)}")
        for stream_idx, error_msg in failed_streams:
            logger.warning(f"  stream_idx={stream_idx} ({mp4_paths[stream_idx]}): {error_msg}")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    workers_per_gpu = OmegaConf.select(args, "workers_per_gpu", default=1)
    mp.set_start_method("spawn", force=True)
    run_parallel(args, workers_per_gpu=workers_per_gpu)


if __name__ == "__main__":
    run()
