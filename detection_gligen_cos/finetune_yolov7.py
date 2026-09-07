from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .common import CocoMini, append_yolov7_device_arg, parse_class_subset, run_logged_subprocess, setup_logger, subset_label, write_yolo_yaml


def patched_yolov7_train_script(yolov7_dir: Path, output_dir: Path, save_every_epochs: int | None) -> Path:
    if save_every_epochs is None or save_every_epochs <= 0:
        return yolov7_dir / "train.py"

    source = yolov7_dir / "train.py"
    target = output_dir / "_yolov7_train_save_every.py"
    text = source.read_text(encoding="utf-8")
    marker = """                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
"""
    replacement = """                if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                    logger.info('Saved periodic checkpoint: %s' % (wdir / 'epoch_{:03d}.pt'.format(epoch)))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
"""
    if marker not in text:
        raise RuntimeError("Unable to patch YOLOv7 train.py for save_every_epochs; expected save_period block not found.")
    text = text.replace(marker, replacement)
    prefix = f"import sys\nsys.path.insert(0, {str(yolov7_dir.resolve())!r})\n"
    target.write_text(prefix + text, encoding="utf-8")
    shutil.copymode(source, target)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--real-train-list", default=None, help="Pre-exported YOLO train image list. Skips COCO->YOLO export when set.")
    parser.add_argument("--real-val-list", default=None, help="Pre-exported YOLO val image list. Skips COCO->YOLO export when set.")
    parser.add_argument("--yolov7", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save-every-epochs", type=int, default=None,
                        help="Save epoch_XXX.pt every N epochs in addition to YOLOv7 last.pt/best.pt behavior.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-name", default="finetune_depth_adapt")
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    args = parser.parse_args()

    output = Path(args.output_dir)
    (output / "datasets").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output)
    class_subset = parse_class_subset(args.classes_subset is not None or args.class_ids is not None, args.classes_subset, args.class_ids)
    train = CocoMini(args.train_annotations, args.train_images, class_subset=class_subset)
    val = CocoMini(args.val_annotations, args.val_images, class_subset=class_subset)
    logger.info("class subset: %s", subset_label(class_subset))
    logger.info("train dataset loaded: images=%d annotations=%d classes=%d",
                len(train.images), sum(len(v) for v in train.annotations.values()), len(train.class_names))
    logger.info("val dataset loaded: images=%d annotations=%d classes=%d",
                len(val.images), sum(len(v) for v in val.annotations.values()), len(val.class_names))
    real_train_ids = [x[0] for x in train.iter_images(None)]
    real_val_ids = [x[0] for x in val.iter_images(None)]
    if args.real_train_list and args.real_val_list:
        real_train = Path(args.real_train_list).resolve()
        real_val = Path(args.real_val_list).resolve()
        logger.info("using shared YOLO real datasets: real_train=%s real_val=%s", real_train, real_val)
    else:
        real_train = train.export_yolo(output / "datasets" / "real_train", real_train_ids)
        real_val = val.export_yolo(output / "datasets" / "real_val", real_val_ids)
        logger.info("exported local YOLO real datasets: real_train=%s real_val=%s", real_train, real_val)
    synthetic = sorted(str(x.resolve()) for x in (Path(args.synthetic_dir) / "images").glob("*.jpg"))
    merged = output / "datasets" / "train_merged.txt"
    merged.write_text(real_train.read_text() + "\n".join(synthetic) + ("\n" if synthetic else ""), encoding="utf-8")
    data_yaml = output / "datasets" / "data.yaml"
    write_yolo_yaml(data_yaml, merged, real_val, train.class_names)
    for cache in (merged.with_suffix(".cache"), real_val.with_suffix(".cache")):
        cache.unlink(missing_ok=True)
        logger.info("cache cleared: %s", cache)
    yolov7_dir = Path(args.yolov7)
    train_script = patched_yolov7_train_script(yolov7_dir, output, args.save_every_epochs)
    command = [
        sys.executable, str(train_script.resolve()),
        "--workers", str(args.workers),
        "--batch-size", str(args.batch_size),
        "--data", str(data_yaml.resolve()),
        "--img", str(args.image_size),
        "--cfg", str(yolov7_dir / "cfg/training/yolov7.yaml"),
        "--weights", str(Path(args.weights).resolve()),
        "--name", args.run_name,
        "--hyp", str(yolov7_dir / "data/hyp.scratch.p5.yaml"),
        "--epochs", str(args.epochs),
        "--project", str((output / "runs").resolve()),
        "--exist-ok",
    ]
    if args.save_every_epochs is not None and args.save_every_epochs > 0:
        command += ["--save_period", str(args.save_every_epochs)]
        logger.info("periodic checkpoint saving enabled: save_every_epochs=%d script=%s",
                    args.save_every_epochs, train_script)
    append_yolov7_device_arg(command, args.device)
    logger.info("finetune start: real_train=%d synthetic=%d val=%d data_yaml=%s epochs=%d batch_size=%d save_every_epochs=%s",
                len(real_train_ids), len(synthetic), len(real_val_ids), data_yaml, args.epochs,
                args.batch_size, args.save_every_epochs)
    run_logged_subprocess(command, yolov7_dir, logger, output / "yolov7_train.log", prefix="yolov7-train")
    weights_dir = output / "runs" / args.run_name / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    results = output / "runs" / args.run_name / "results.txt"
    periodic = sorted(weights_dir.glob("epoch_*.pt"))
    logger.info("finetune complete: weights_dir=%s best_exists=%s last_exists=%s periodic_checkpoints=%d results=%s",
                weights_dir, best.exists(), last.exists(), len(periodic), results)
    if args.save_every_epochs is not None and args.save_every_epochs > 0:
        logger.info("periodic checkpoint files: %s", [str(x) for x in periodic])


if __name__ == "__main__":
    main()
