import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from main_train_tpro import (
    create_stage_config,
    ensure_directory,
    ensure_symlink,
    find_checkpoint,
    load_yaml,
    prepare_dataset_structure,
)


def run_segmentation_inprocess(config_path: Path,
                               runtime_cfg: dict,
                               pseudo_mask_root: Path) -> None:
    from src.externals.TPRO import train_seg as train_seg_module

    cfg = train_seg_module.OmegaConf.load(str(config_path))
    cfg.work_dir.dir = os.path.dirname(str(config_path))

    timestamp = train_seg_module.datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
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
            primary = cuda_devices.split(",")[0].strip()
            os.environ["CUDA_VISIBLE_DEVICES"] = primary or cuda_devices
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    backend = runtime_cfg.get("backend", "nccl")
    if backend == "nccl" and not train_seg_module.torch.cuda.is_available():
        backend = "gloo"

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(runtime_cfg.get("master_port", 17362))
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

    args_namespace = SimpleNamespace(
        config=str(config_path),
        local_rank=0,
        backend=backend,
        wandb_log=bool(runtime_cfg.get("wandb_log", False)),
    )
    train_seg_module.args = args_namespace

    if args_namespace.local_rank == 0:
        if args_namespace.wandb_log:
            train_seg_module.wandb.init(project=f"TPRO-{cfg.dataset.name}-seg")
        train_seg_module.setup_logger(
            filename=os.path.join(cfg.work_dir.train_log_dir, timestamp + ".log")
        )
        train_seg_module.logging.info("\nargs: %s", args_namespace)
        train_seg_module.logging.info("\nconfigs: %s", cfg)

    train_seg_module.set_seed(0)
    cfg.dataset.mask_root = str(pseudo_mask_root)
    train_seg_module.train(cfg=cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TPRO segmentation stage")
    parser.add_argument("--config", required=True, help="Path to configs_maui/tpro_*.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_yaml(config_path)

    repo_root = Path(__file__).resolve().parent
    tpro_module_path = (repo_root / "src" / "externals" / "TPRO").resolve()
    if str(tpro_module_path) not in sys.path:
        sys.path.insert(0, str(tpro_module_path))

    tpro_root = Path(cfg["tpro"]["root"]).resolve()
    if not tpro_root.exists():
        raise FileNotFoundError(f"TPRO root not found: {tpro_root}")

    modify_root = Path(cfg["tpro"]["modify_root"]).resolve()
    dataset_name = cfg["dataset"]["name"]
    stage_config_root = modify_root / dataset_name
    ensure_directory(stage_config_root)

    data_root = prepare_dataset_structure(cfg, tpro_root)

    work_dir_cfg = cfg["work_dir"]
    seg_ckpt_root = Path(work_dir_cfg["ckpt_dir"]) / "segmentation"
    ensure_directory(seg_ckpt_root)

    pseudo_dir = data_root / "train" / cfg["dataset"]["pseudo_label_dir_name"]
    if not pseudo_dir.exists():
        raise FileNotFoundError(
            f"Pseudo-label directory not found: {pseudo_dir}. "
            "Run main_train_tpro_pseudo.py first."
        )

    seg_model_cfg = cfg["model"]["segmentation"]

    seg_overrides = {
        "model": {
            "backbone": {
                "config": seg_model_cfg["backbone"],
                "stride": seg_model_cfg["stride"],
            },
        },
        "dataset": {
            "name": cfg["dataset"]["name"],
            "train_root": str((data_root / "train").resolve()),
            "val_root": str(data_root.resolve()),
            "mask_root": cfg["dataset"]["pseudo_label_dir_name"],
            "seg_num_classes": seg_model_cfg["seg_num_classes"],
        },
        "work_dir": {
            "ckpt_dir": str(seg_ckpt_root.resolve()),
            "pred_dir": str((Path(work_dir_cfg["pred_dir"]) / "segmentation").resolve()),
            "train_log_dir": str((Path(work_dir_cfg["train_log_dir"]) / "segmentation").resolve()),
        },
    }

    seg_config_output = stage_config_root / "segmentation" / "config.generated.yaml"
    base_cfg_source: Optional[object] = seg_model_cfg.get("config_file")
    if base_cfg_source is None:
        if "base_config" not in seg_model_cfg:
            raise KeyError("Segmentation model config must define either 'config_file' or 'base_config'.")
        base_cfg_source = Path(seg_model_cfg["base_config"]).resolve()
    create_stage_config(base_cfg_source, seg_overrides, seg_config_output)

    pretrained_dir = tpro_root / "pretrained"
    ensure_directory(pretrained_dir)

    def link_pretrained(backbone: str, weight_path: Optional[str]) -> None:
        if not weight_path:
            return
        src = Path(weight_path).resolve()
        dst = pretrained_dir / f"{backbone}.pth"
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and dst.resolve() == src:
                return
            return
        ensure_symlink(src, dst)

    link_pretrained(seg_model_cfg["backbone"], seg_model_cfg.get("pretrained"))

    print(f"[TPRO][seg] TPRO root: {tpro_root}")
    print(f"[TPRO][seg] Dataset workspace: {data_root}")
    print(f"[TPRO][seg] Pseudo-label directory: {pseudo_dir}")
    print(f"[TPRO][seg] Segmentation config: {seg_config_output}")

    os.chdir(tpro_root)
    run_segmentation_inprocess(seg_config_output, cfg["runtime"]["segmentation"], pseudo_dir)

    best_seg_ckpt = find_checkpoint(seg_ckpt_root, "best_seg.pth")
    print(f"[TPRO][seg] Segmentation training complete. Best checkpoint: {best_seg_ckpt}")


if __name__ == "__main__":
    main()
