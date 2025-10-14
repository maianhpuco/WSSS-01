import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from main_train_tpro import (
    ensure_directory,
    ensure_symlink,
    find_checkpoint,
    load_yaml,
    prepare_dataset_structure,
)


def run_evaluate_inprocess(
    *,
    tpro_root: Path,
    dataset: str,
    split: str,
    checkpoint: Path,
    save_dir: Path,
    gpu_id: int,
    backbone: str,
    label_feature_rel: str,
    knowledge_feature_rel: str,
    n_ratio: float,
    palette_path: Optional[Path],
) -> None:
    from src.externals.TPRO import evaluate_cls as evaluate_module

    def build_args() -> SimpleNamespace:
        return SimpleNamespace(
            gpu=gpu_id,
            dataset=dataset,
            cls_num_classes=4,
            model_path=str(checkpoint),
            img_root="",
            palette_path=str(palette_path) if palette_path else "./datasets/luad_palette.npy",
            seed=0,
            backbone=backbone,
            split=split,
            label_feature_path=label_feature_rel,
            knowledge_feature_path=knowledge_feature_rel,
            n_ratio=n_ratio,
            l1=0.0,
            l2=0.0,
            l3=0.0,
            save_dir=str(save_dir),
        )

    original_get_args = evaluate_module.get_args
    evaluate_module.get_args = build_args
    try:
        evaluate_module.main()
    finally:
        evaluate_module.get_args = original_get_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pseudo labels using the latest classification checkpoint"
    )
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
    cls_ckpt_root = Path(work_dir_cfg["ckpt_dir"]) / "classification"
    ensure_directory(cls_ckpt_root)

    pseudo_cfg = cfg["model"]["pseudo_label"]
    pseudo_label_dir = data_root / "train" / cfg["dataset"]["pseudo_label_dir_name"]
    ensure_directory(pseudo_label_dir)

    cls_model_cfg = cfg["model"]["classification"]
    feature_base_dir = (tpro_root / "text&features/text_features").resolve()
    label_feature_file = Path(cls_model_cfg["label_feature_file"]).resolve()
    knowledge_feature_file = Path(cls_model_cfg["knowledge_feature_file"]).resolve()
    try:
        label_feature_rel = str(
            label_feature_file.relative_to(feature_base_dir).with_suffix("")
        ).replace("\\", "/")
        knowledge_feature_rel = str(
            knowledge_feature_file.relative_to(feature_base_dir).with_suffix("")
        ).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(
            "Label/knowledge feature files must be located under "
            f"{feature_base_dir}."
        ) from exc

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

    best_cam_ckpt = find_checkpoint(cls_ckpt_root, "best_cam.pth")

    def prepare_checkpoint_for_eval(original_ckpt: Path) -> Path:
        if "/checkpoints/" in str(original_ckpt):
            return original_ckpt
        alias_dir = original_ckpt.parent.parent / "checkpoints"
        ensure_directory(alias_dir)
        alias_path = alias_dir / original_ckpt.name
        if alias_path.exists() or alias_path.is_symlink():
            if alias_path.is_symlink() and alias_path.resolve() == original_ckpt:
                return alias_path
            alias_path.unlink()
        ensure_symlink(original_ckpt, alias_path)
        return alias_path

    eval_checkpoint = prepare_checkpoint_for_eval(best_cam_ckpt.resolve())

    print(f"[TPRO][pseudo] TPRO root: {tpro_root}")
    print(f"[TPRO][pseudo] Dataset workspace: {data_root}")
    print(f"[TPRO][pseudo] Classification checkpoint: {best_cam_ckpt}")
    print(f"[TPRO][pseudo] Evaluation checkpoint path: {eval_checkpoint}")
    print(f"[TPRO][pseudo] Pseudo-label output directory: {pseudo_label_dir}")

    os.chdir(tpro_root)

    if pseudo_cfg.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(pseudo_cfg["gpu"])

    palette_path = (
        Path(pseudo_cfg["palette_path"]).resolve() if pseudo_cfg.get("palette_path") else None
    )

    run_evaluate_inprocess(
        tpro_root=tpro_root,
        dataset=cfg["dataset"]["name"],
        split=pseudo_cfg["split"],
        checkpoint=eval_checkpoint,
        save_dir=pseudo_label_dir,
        gpu_id=0,
        backbone=cls_model_cfg["backbone"],
        label_feature_rel=label_feature_rel,
        knowledge_feature_rel=knowledge_feature_rel,
        n_ratio=cls_model_cfg["n_ratio"],
        palette_path=palette_path,
    )

    print("[TPRO][pseudo] Pseudo-label generation complete.")


if __name__ == "__main__":
    main()
