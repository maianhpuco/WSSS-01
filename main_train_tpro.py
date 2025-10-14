import argparse
import copy
import datetime
import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
TPRO_SRC = (REPO_ROOT / "src" / "externals" / "TPRO").resolve()
if str(TPRO_SRC) not in sys.path:
    sys.path.insert(0, str(TPRO_SRC))


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def dump_yaml(content: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(content, handle, sort_keys=False)


def ensure_symlink(src: Path, dst: Path, allow_existing: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if allow_existing:
            return
        raise FileExistsError(f"Destination {dst} already exists and does not point to {src}")
    if not src.exists():
        raise FileNotFoundError(f"Source for symlink not found: {src}")
    os.symlink(src, dst)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_dataset_structure(cfg: Dict[str, Any], tpro_root: Path) -> Path:
    dataset_cfg = cfg["dataset"]
    source_root = Path(dataset_cfg["source_root"]).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset source root not found: {source_root}")

    dataset_dir_name = source_root.name
    data_root = tpro_root / "data" / dataset_dir_name
    ensure_directory(data_root)

    splits = dataset_cfg["splits"]
    train_img_src = (source_root / splits["train"]).resolve()
    if not train_img_src.exists():
        raise FileNotFoundError(f"Training images directory not found: {train_img_src}")
    ensure_symlink(train_img_src, data_root / "train" / "img", allow_existing=True)

    for split_key in ("valid", "test"):
        split_src = (source_root / splits[split_key]).resolve()
        img_src = split_src / "img"
        mask_src = split_src / "mask"
        if not img_src.exists() or not mask_src.exists():
            raise FileNotFoundError(f"Expected img/mask under {split_src}")
        ensure_symlink(img_src, data_root / split_key / "img", allow_existing=True)
        ensure_symlink(mask_src, data_root / split_key / "mask", allow_existing=True)

    pseudo_dir = data_root / "train" / dataset_cfg["pseudo_label_dir_name"]
    ensure_directory(pseudo_dir)

    return data_root


def create_stage_config(base_cfg_source: Union[Path, Dict[str, Any]],
                        overrides: Dict[str, Any],
                        output_path: Path) -> Dict[str, Any]:
    if isinstance(base_cfg_source, Path):
        base_cfg = load_yaml(base_cfg_source)
    elif isinstance(base_cfg_source, dict):
        base_cfg = copy.deepcopy(base_cfg_source)
    else:
        raise TypeError(
            f"Unsupported base config source type: {type(base_cfg_source)!r}"
        )

    def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in src.items():
            if isinstance(value, dict) and key in dst and isinstance(dst[key], dict):
                deep_merge(dst[key], value)
            else:
                dst[key] = value
        return dst

    merged = deep_merge(base_cfg, overrides)
    dump_yaml(merged, output_path)
    return merged


def run_distributed(stage_name: str,
                    script: str,
                    config_path: Path,
                    runtime_cfg: Dict[str, Any],
                    cwd: Path) -> None:
    env = os.environ.copy()
    cuda_devices = runtime_cfg.get("cuda_visible_devices")
    if cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        f"--nproc_per_node={runtime_cfg['nproc_per_node']}",
        f"--master_port={runtime_cfg['master_port']}",
        script,
        "--config",
        str(config_path),
    ]

    if runtime_cfg.get("wandb_log"):
        cmd.extend(["--wandb_log", str(runtime_cfg["wandb_log"]).lower()])

    print(f"[TPRO][{stage_name}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def run_classification_inprocess(config_path: Path,
                                 runtime_cfg: Dict[str, Any]) -> None:
    from src.externals.TPRO import train_cls as train_cls_module

    cfg = OmegaConf.load(str(config_path))
    cfg.work_dir.dir = os.path.dirname(str(config_path))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.train_log_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.train_log_dir)

    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.train_log_dir, exist_ok=True)

    cuda_devices = runtime_cfg.get("cuda_visible_devices")
    if cuda_devices:
        if isinstance(cuda_devices, str):
            primary_device = cuda_devices.split(",")[0].strip()
            os.environ["CUDA_VISIBLE_DEVICES"] = primary_device or cuda_devices
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    backend = runtime_cfg.get("backend", "nccl")
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(runtime_cfg.get("master_port", 16372))
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

    args_namespace = SimpleNamespace(
        config=str(config_path),
        local_rank=0,
        backend=backend,
        wandb_log=bool(runtime_cfg.get("wandb_log", False)),
    )
    train_cls_module.args = args_namespace

    if args_namespace.local_rank == 0:
        if args_namespace.wandb_log:
            train_cls_module.wandb.init(project=f"TPRO-{cfg.dataset.name}-cls")
        train_cls_module.setup_logger(
            filename=os.path.join(cfg.work_dir.train_log_dir, timestamp + ".log")
        )
        train_cls_module.logging.info("\nargs: %s", args_namespace)
        train_cls_module.logging.info("\nconfigs: %s", cfg)

    train_cls_module.set_seed(0)
    train_cls_module.train(cfg=cfg)


def find_checkpoint(ckpt_root: Path, filename: str) -> Path:
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint root {ckpt_root} does not exist")
    candidates = sorted(ckpt_root.glob(f"*/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No {filename} found under {ckpt_root}")
    return candidates[0]


def run_command(stage: str,
                cmd: List[str],
                cwd: Path,
                env: Optional[Dict[str, str]] = None) -> None:
    print(f"[TPRO][{stage}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="TPRO end-to-end training helper")
    parser.add_argument("--config", required=True, help="Path to configs_maui/tpro_*.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_yaml(config_path)

    tpro_root = Path(cfg["tpro"]["root"]).resolve()
    if not tpro_root.exists():
        raise FileNotFoundError(f"TPRO root not found: {tpro_root}")

    modify_root = Path(cfg["tpro"]["modify_root"]).resolve()
    dataset_name = cfg["dataset"]["name"]
    stage_config_root = modify_root / dataset_name
    ensure_directory(stage_config_root)

    data_root = prepare_dataset_structure(cfg, tpro_root)

    # Stage-specific directories
    work_dir_cfg = cfg["work_dir"]
    cls_ckpt_root = Path(work_dir_cfg["ckpt_dir"]) / "classification"
    seg_ckpt_root = Path(work_dir_cfg["ckpt_dir"]) / "segmentation"
    cls_pred_root = Path(work_dir_cfg["pred_dir"]) / "classification"
    seg_pred_root = Path(work_dir_cfg["pred_dir"]) / "segmentation"
    cls_log_root = Path(work_dir_cfg["train_log_dir"]) / "classification"
    seg_log_root = Path(work_dir_cfg["train_log_dir"]) / "segmentation"

    for path in (cls_ckpt_root, seg_ckpt_root, cls_pred_root, seg_pred_root, cls_log_root, seg_log_root):
        ensure_directory(path)

    pseudo_cfg = cfg["model"]["pseudo_label"]
    pseudo_label_dir = data_root / "train" / cfg["dataset"]["pseudo_label_dir_name"]
    ensure_directory(pseudo_label_dir)

    # Prepare classification config
    cls_model_cfg = cfg["model"]["classification"]

    def resolve_base_config(stage_cfg: Dict[str, Any]) -> Union[Path, Dict[str, Any]]:
        if "config_file" in stage_cfg:
            return stage_cfg["config_file"]
        if "base_config" in stage_cfg:
            return Path(stage_cfg["base_config"]).resolve()
        raise KeyError("Either 'config_file' or 'base_config' must be provided in stage configuration.")

    feature_base_dir = (tpro_root / "text&features/text_features").resolve()
    label_feature_file = Path(cls_model_cfg["label_feature_file"]).resolve()
    knowledge_feature_file = Path(cls_model_cfg["knowledge_feature_file"]).resolve()
    try:
        label_feature_rel_path = label_feature_file.relative_to(feature_base_dir)
        knowledge_feature_rel_path = knowledge_feature_file.relative_to(feature_base_dir)
    except ValueError as exc:
        raise ValueError(
            "Label/knowledge feature files must be located under "
            f"{feature_base_dir}."
        ) from exc
    label_feature_rel = str(label_feature_rel_path.with_suffix("")).replace("\\", "/")
    knowledge_feature_rel = str(knowledge_feature_rel_path.with_suffix("")).replace("\\", "/")

    cls_cfg_overrides = {
        "model": {
            "backbone": {
                "config": cls_model_cfg["backbone"],
                "stride": cls_model_cfg["stride"],
            },
            "label_feature_path": label_feature_rel,
            "knowledge_feature_path": knowledge_feature_rel,
            "n_ratio": cls_model_cfg["n_ratio"],
        },
        "dataset": {
            "name": cfg["dataset"]["name"],
            "train_root": str((data_root / "train" / "img").resolve()),
            "val_root": str(data_root.resolve()),
        },
        "work_dir": {
            "ckpt_dir": str(cls_ckpt_root.resolve()),
            "pred_dir": str(cls_pred_root.resolve()),
            "train_log_dir": str(cls_log_root.resolve()),
        },
    }

    cls_config_output = stage_config_root / "classification" / "config.generated.yaml"
    cls_base_config_source = resolve_base_config(cls_model_cfg)
    create_stage_config(cls_base_config_source, cls_cfg_overrides, cls_config_output)

    pretrained_dir = tpro_root / "pretrained"
    ensure_directory(pretrained_dir)

    def link_pretrained(backbone: str, weight_path: Optional[str]) -> None:
        if weight_path is None:
            return
        src = Path(weight_path).resolve()
        dst = pretrained_dir / f"{backbone}.pth"
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and dst.resolve() == src:
                return
            raise FileExistsError(f"Pretrained file already exists at {dst}")
        ensure_symlink(src, dst)

    link_pretrained(cls_model_cfg["backbone"], cls_model_cfg.get("pretrained"))
    seg_model_cfg = cfg["model"]["segmentation"]
    link_pretrained(seg_model_cfg["backbone"], seg_model_cfg.get("pretrained"))

    # Run classification training directly within this process
    os.chdir(tpro_root)
    run_classification_inprocess(cls_config_output, cfg["runtime"]["classification"])

    # Subsequent stages (pseudo-label extraction and segmentation) will be added later
    return


if __name__ == "__main__":
    main()
