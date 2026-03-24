import os
from pathlib import Path
from einops import rearrange

import torch
import torchvision
import numpy as np
import imageio
import shutil
import tarfile
import logging

CODE_SUFFIXES = {
    ".py",  # Python codes
    ".sh",  # Shell scripts
    ".yaml",
    ".yml",  # Configuration files
}


def safe_dir(path):
    """
    Create a directory (or the parent directory of a file) if it does not exist.

    Args:
        path (str or Path): Path to the directory.

    Returns:
        path (Path): Path object of the directory.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def safe_file(path):
    """
    Create the parent directory of a file if it does not exist.

    Args:
        path (str or Path): Path to the file.

    Returns:
        path (Path): Path object of the file.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path

def safe_save_file(save_path, *args, save_fn=None, **kwargs):
    save_path = Path(save_path)
    tmp_save_path = save_path.parent / f'temp_{save_path.name}'
    save_to = None
    try:
        save_to = save_fn(tmp_save_path, *args, **kwargs)
        shutil.copyfile(tmp_save_path, save_path)
        save_to = save_path
        tmp_save_path.unlink()
    except Exception as e:
        print(f'Failed to save to {save_path}. {type(e)}: {e}')
    return save_to


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def dump_codes(save_path, root, sub_dirs=None, valid_suffixes=None, save_prefix='./'):
    """
    Dump codes to the experiment directory.

    Args:
        save_path (str): Path to the experiment directory.
        root (Path): Path to the root directory of the codes.
        sub_dirs (list): List of subdirectories to be dumped. If None, all files in the root directory will
            be dumped. (default: None)
        valid_suffixes (tuple, optional): Valid suffixes of the files to be dumped. If None, CODE_SUFFIXES will be used.
            (default: None)
        save_prefix (str, optional): Prefix to be added to the files in the tarball. (default: './')
    """
    if valid_suffixes is None:
        valid_suffixes = CODE_SUFFIXES

    # Force to use tar.gz suffix
    save_path = safe_file(save_path)
    assert save_path.name.endswith('.tar.gz'), f"save_path should end with .tar.gz, got {save_path.name}."
    # Make root absolute
    root = Path(root).absolute()
    # Make a tarball of the codes
    with tarfile.open(save_path, "w:gz") as tar:
        # Recursively add all files in the root directory
        if sub_dirs is None:
            sub_dirs = list(root.iterdir())
        for sub_dir in sub_dirs:
            for file in Path(sub_dir).rglob('*'):
                if file.is_file() and file.suffix in valid_suffixes:
                    # make file absolute
                    file = file.absolute()
                    arcname = Path(save_prefix) / file.relative_to(root)
                    tar.add(file, arcname=arcname)
    return root

def get_next_available_save_id(src_dir):
    """
    Get the next available save ID in the specified directory.
    This function iterates through all files in the specified directory, extracts the numeric part at the beginning of the file name as an ID,
    and then returns the value of the maximum ID plus 1. If there are no valid IDs in the directory, it returns 0.

    Args:
        src_dir (str or Path): The path of the source directory.

    Returns:
        int: The next available save ID.
    """
    src_dir = Path(src_dir)
    existed_files = list(src_dir.glob("*"))
    valid_ids = []
    for existed_files in existed_files:
        head = existed_files.name.split('_')[0]
        if head.isdigit():
            valid_ids.append(int(head))
    if not valid_ids:
        return 0
    return max(valid_ids) + 1

def empty_logger():
    logger = logging.getLogger("hymm_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger

def rank0_logger(rank):
    if rank == 0:
        from loguru import logger
    else:
        logger = empty_logger()
    return logger

def validate_video_csv_files(args, logger):
    """
    Validate that all video_csv files exist before starting training.
    
    Args:
        args: Training arguments object containing video_csv attribute
        logger: Logger instance for outputting validation results
    
    Raises:
        FileNotFoundError: If any video_csv files are missing
    """
    if not hasattr(args, 'video_csv') or args.video_csv is None:
        logger.info("No video_csv files specified, skipping validation.")
        return
    
    logger.info("=" * 80)
    logger.info("VALIDATING VIDEO_CSV FILES")
    logger.info("=" * 80)
    
    missing_files = []
    valid_files = []
    
    for i, csv_path in enumerate(args.video_csv):
        if os.path.exists(csv_path):
            valid_files.append(csv_path)
            logger.info(f"✓ [{i+1:2d}/{len(args.video_csv)}] Found: {csv_path}")
        else:
            missing_files.append(csv_path)
            logger.error(f"✗ [{i+1:2d}/{len(args.video_csv)}] Missing: {csv_path}")
    
    logger.info("-" * 80)
    logger.info(f"Total video_csv files: {len(args.video_csv)}")
    logger.info(f"Valid files: {len(valid_files)}")
    logger.info(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        logger.error("=" * 80)
        logger.error("VALIDATION FAILED - MISSING FILES:")
        for missing_file in missing_files:
            logger.error(f"  - {missing_file}")
        logger.error("=" * 80)
        logger.error("Please check the file paths in your configuration and ensure all files exist.")
        logger.error("Training cannot continue with missing video_csv files.")
        raise FileNotFoundError(f"Missing {len(missing_files)} video_csv files. See log for details.")
    else:
        logger.info("✓ All video_csv files validated successfully!")
        logger.info("=" * 80)

def convert_to_json_serializable(obj):
    """Convert NumPy and other non-JSON-serializable types to native Python types."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    else:
        return obj