import argparse
import os
from pathlib import Path
from typing import Optional

from main_train_tpro import (
    create_stage_config,
    ensure_directory,
    ensure_symlink,
    load_yaml,
    prepare_dataset_structure,
    run_classification_inprocess,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TPRO classification stage without torch.distributed.launch")
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

    work_dir_cfg = cfg["work_dir"]
    cls_ckpt_root = Path(work_dir_cfg["ckpt_dir"]) / "classification"
    cls_pred_root = Path(work_dir_cfg["pred_dir"]) / "classification"
    cls_log_root = Path(work_dir_cfg["train_log_dir"]) / "classification"
    for path in (cls_ckpt_root, cls_pred_root, cls_log_root):
        ensure_directory(path)

    pseudo_label_dir = data_root / "train" / cfg["dataset"]["pseudo_label_dir_name"]
    ensure_directory(pseudo_label_dir)

    cls_model_cfg = cfg["model"]["classification"]

    feature_base_dir = (tpro_root / "text&features/text_features").resolve()
    label_feature_file = Path(cls_model_cfg["label_feature_file"]).resolve()
    knowledge_feature_file = Path(cls_model_cfg["knowledge_feature_file"]).resolve()
    try:
        label_feature_rel = str(label_feature_file.relative_to(feature_base_dir).with_suffix("")).replace("\\", "/")
        knowledge_feature_rel = str(knowledge_feature_file.relative_to(feature_base_dir).with_suffix("")).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(
            "Label/knowledge feature files must be located under "
            f"{feature_base_dir}."
        ) from exc

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
    base_cfg_source = cls_model_cfg.get("config_file")
    if base_cfg_source is None:
        if "base_config" not in cls_model_cfg:
            raise KeyError("Classification model config must define either 'config_file' or 'base_config'.")
        base_cfg_source = Path(cls_model_cfg["base_config"]).resolve()
    create_stage_config(base_cfg_source, cls_cfg_overrides, cls_config_output)

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

    link_pretrained(cls_model_cfg["backbone"], cls_model_cfg.get("pretrained"))

    os.chdir(tpro_root)
    run_classification_inprocess(cls_config_output, cfg["runtime"]["classification"])

    print("[TPRO] Classification training finished.")


if __name__ == "__main__":
    main()
