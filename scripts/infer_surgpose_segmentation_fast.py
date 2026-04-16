"""
Fast batched inference script for toolpose segmentation models.
Writes mask locations and logs evaluation metrics.
"""

import json
import logging
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import configargparse
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(".")
sys.path.append("./models/")

from configs.config_toolposeseg import test_config_parser as config_parser
from models import get_tooltip_segmentation_model as get_model
from src.dataloader import (
    RoboticSurgeryFramesDataset,
    RoboticSurgeryFramesDataset_withoptflow,
    get_transform,
)
from src.metrics import get_metrics
from utils.dataloader_utils import (
    get_JIGSAWS_dataset_filenames,
    get_MICCAI2015_dataset_filenames,
    get_MICCAI2017_dataset_filenames,
    get_SurgPose_dataset_filenames,
    load_attmap,
    load_image,
)
from utils.localization_utils_v2 import centroid_error
from utils.log_utils import AverageMeter, ProgressMeter
from utils.model_utils import load_model_weights
from utils.train_utils import add_metrics_meters
from utils.vis_utils import draw_plus, mask_overlay


surgpose_palette = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (0, 255, 255),
}


def _get_mask_locations(mask_array, num_classes):
    mask_locations: dict[str, object] = {}
    for cls in range(1, num_classes):
        class_mask = (mask_array == cls).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            class_mask, connectivity=8
        )
        if num_labels <= 1:
            mask_locations[str(cls)] = None
            continue
        components = []
        for label_idx in range(1, num_labels):
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            num_pixels = int(stats[label_idx, cv2.CC_STAT_AREA])
            components.append(
                {
                    "bbox_xyxy": [x, y, x + w - 1, y + h - 1],
                    "num_pixels": num_pixels,
                }
            )

        # Keep backward-compatible top-level keys by selecting the largest component.
        largest_component = max(components, key=lambda c: c["num_pixels"])
        mask_locations[str(cls)] = {
            "bbox_xyxy": largest_component["bbox_xyxy"],
            "num_pixels": largest_component["num_pixels"],
            "components": components,
        }
    return mask_locations


def _get_component_center(component_info):
    if component_info is None:
        return None
    x1, y1, x2, y2 = component_info["bbox_xyxy"]
    center_x = int(round((x1 + x2) / 2.0))
    center_y = int(round((y1 + y2) / 2.0))
    return [center_x, center_y]


def _safe_mean(values):
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _safe_std(values):
    if len(values) == 0:
        return None
    return float(np.std(values))


def _save_mask_overlay(file_path, output_path, mask_array, c_gt, c_pred, args):
    disp_image = cv2.imread(str(file_path))
    if disp_image is None:
        return
    disp_image = cv2.resize(disp_image, (args.input_width, args.input_height))
    for i in range(1, args.num_classes):
        disp_image = mask_overlay(
            disp_image, (mask_array == i).astype(np.uint8), color=surgpose_palette[i]
        )
    # if c_gt is not None and c_pred is not None:
    #     disp_image = draw_plus(disp_image, [c_gt[0][0], c_gt[1][0]], color=(0, 255, 0))
    #     disp_image = draw_plus(disp_image, [c_gt[0][1], c_gt[1][1]], color=(0, 255, 0))
    #     disp_image = draw_plus(disp_image, [c_gt[2][0], c_gt[3][0]], color=(0, 255, 0))
    #     disp_image = draw_plus(disp_image, [c_gt[4][0], c_gt[5][0]], color=(0, 255, 0))
    #     disp_image = draw_plus(disp_image, [c_gt[4][1], c_gt[5][1]], color=(0, 255, 0))
    #     disp_image = draw_plus(disp_image, [c_gt[6][0], c_gt[7][0]], color=(0, 255, 0))
    #     disp_image = draw_plus(
    #         disp_image, [c_pred[0][0], c_pred[1][0]], color=(255, 255, 255)
    #     )
    #     disp_image = draw_plus(
    #         disp_image, [c_pred[0][1], c_pred[1][1]], color=(255, 255, 255)
    #     )
    #     disp_image = draw_plus(
    #         disp_image, [c_pred[2][0], c_pred[3][0]], color=(255, 255, 255)
    #     )
    #     disp_image = draw_plus(
    #         disp_image, [c_pred[4][0], c_pred[5][0]], color=(255, 255, 255)
    #     )
    #     disp_image = draw_plus(
    #         disp_image, [c_pred[4][1], c_pred[5][1]], color=(255, 255, 255)
    #     )
    #     disp_image = draw_plus(
    #         disp_image, [c_pred[6][0], c_pred[7][0]], color=(255, 255, 255)
    #     )
    cv2.imwrite(str(output_path), disp_image)


def _save_keypoint_overlay(file_path, output_path, mask_locations, args):
    disp_image = cv2.imread(str(file_path))
    if disp_image is None:
        return
    disp_image = cv2.resize(disp_image, (args.input_width, args.input_height))
    for cls_str, component_info in mask_locations.items():
        center = _get_component_center(component_info)
        if center is None:
            continue
        disp_image = draw_plus(
            disp_image,
            center,
            color=surgpose_palette.get(int(cls_str), (255, 255, 255)),
            size=8,
            thickness=2,
        )
    cv2.imwrite(str(output_path), disp_image)


def _add_fast_inference_args(parser):
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=16,
        help="Batch size used by the fast inference dataloader.",
    )
    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision during inference.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Number of prefetched batches per dataloader worker.",
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip evaluation metric computation and only export predictions.",
    )
    return parser


def _save_output_freq_was_provided(argv):
    return any(
        arg == "--save_output_freq" or arg.startswith("--save_output_freq=")
        for arg in argv
    )


def main():
    parser = configargparse.ArgumentParser()
    parser = config_parser(parser)
    parser = _add_fast_inference_args(parser)
    args = parser.parse_args()
    if not _save_output_freq_was_provided(sys.argv[1:]):
        args.skip_output_images = True
    main_worker(args)


def save_attention_maps(model, args):
    model.eval()
    if args.dataset == "MICCAI2017":
        file_names, _ = get_MICCAI2017_dataset_filenames(args)
    elif args.dataset == "JIGSAWS":
        file_names, _ = get_JIGSAWS_dataset_filenames(args)
    elif args.dataset == "MICCAI2015":
        file_names, _ = get_MICCAI2015_dataset_filenames(args)
    elif args.dataset == "SurgPose":
        file_names, _ = get_SurgPose_dataset_filenames(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    with torch.inference_mode():
        for idx, file_name in enumerate(file_names):
            input = torch.from_numpy(load_image(file_name).transpose(2, 0, 1))
            input = input.type(torch.float32) / 255.0
            input = transforms.Resize((args.input_height, args.input_width))(input)
            input = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )(input)
            input = input.unsqueeze(0)
            input = (
                input.cuda(non_blocking=True) if torch.cuda.is_available() else input
            )
            input = input.contiguous(memory_format=torch.channels_last)
            attmap = load_attmap(file_names, idx, args.num_frames_per_video)
            attmap = torch.from_numpy(attmap).unsqueeze(0).unsqueeze(0)
            attmap = transforms.Resize((args.input_height, args.input_width))(attmap)
            attmap = (
                attmap.cuda(non_blocking=True) if torch.cuda.is_available() else attmap
            )
            with _get_autocast_context(args):
                output = model(input, attmap)
            output = torch.exp(output)
            output = torch.sum(output[:, 1:, :, :], dim=1, keepdim=False)
            for i in range(input.size(0)):
                cv2.imwrite(
                    str(file_name).replace("images", "attmaps").replace("jpg", "png"),
                    (255 * output[i].detach().cpu().squeeze().numpy()).astype(np.uint8),
                )
    return


def _get_dataset_filenames(args):
    if args.dataset == "MICCAI2017":
        return get_MICCAI2017_dataset_filenames(args)[0]
    if args.dataset == "JIGSAWS":
        return get_JIGSAWS_dataset_filenames(args)[0]
    if args.dataset == "MICCAI2015":
        return get_MICCAI2015_dataset_filenames(args)[0]
    if args.dataset == "SurgPose":
        return get_SurgPose_dataset_filenames(args)[0]
    raise ValueError(f"Unknown dataset: {args.dataset}")


def _build_test_dataloader(args, test_file_names):
    test_transform = get_transform("test", args)
    if not args.add_optflow_inputs:
        test_dataset = RoboticSurgeryFramesDataset(
            test_file_names,
            transform=test_transform,
            mode=args.mode,
            prediction_task=args.prediction_task,
        )
    else:
        assert "TAPNet" in args.model_type
        test_dataset = RoboticSurgeryFramesDataset_withoptflow(
            test_file_names,
            args.optflow_dir,
            transform=test_transform,
            mode=args.mode,
            prediction_task=args.prediction_task,
            num_frames=args.num_frames_per_video,
        )

    dataloader_kwargs = {
        "batch_size": args.test_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(test_dataset, **dataloader_kwargs)


def _get_autocast_context(args):
    if torch.cuda.is_available() and not args.disable_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _forward_logits(model, inputs, args):
    if "TAPNet" in args.model_type:
        attmaps = inputs[:, 3:, :, :]
        inputs = inputs[:, :3, :, :]
        return model(inputs, attmaps)
    if "DeepLab_v3" in args.model_type or "FCN" in args.model_type:
        return model(inputs)["out"]
    return model(inputs)


def _prepare_outputs_for_centroid(outputs, args):
    if "DeepLab_v3" in args.model_type or "FCN" in args.model_type:
        return torch.exp(F.log_softmax(outputs, dim=1))
    if "HRNet" in args.model_type:
        return F.log_softmax(outputs, dim=1)
    return outputs


def _normalize_centroid_error_result(centroid_result, args):
    if args.num_classes == 6:
        err_list, pres_gt, pres, c_gt, c_pred = centroid_result
        return list(err_list), list(pres_gt), list(pres), list(c_gt), list(c_pred)
    if args.num_classes == 5:
        err_rc, err_rb, err_lc, err_lb, pres_gt, pres, c_gt, c_pred = centroid_result
        return [err_rc, err_rb, err_lc, err_lb], list(pres_gt), list(pres), c_gt, c_pred
    if args.num_classes == 3:
        err_rc, err_lc, pres_gt, pres, c_gt, c_pred = centroid_result
        return [err_rc, err_lc], list(pres_gt), list(pres), c_gt, c_pred
    raise ValueError(
        f"Centroid metrics are only implemented for 3, 5, or 6 classes, got {args.num_classes}."
    )


def _update_pck_counts(err_list, pck_2p5, pck_5, pck_10):
    for err in err_list:
        if math.isnan(err):
            pck_2p5.append(0)
            pck_5.append(0)
            pck_10.append(0)
        elif err <= 2.5:
            pck_2p5.append(1)
            pck_5.append(1)
            pck_10.append(1)
        elif err <= 5:
            pck_2p5.append(0)
            pck_5.append(1)
            pck_10.append(1)
        elif err <= 10:
            pck_2p5.append(0)
            pck_5.append(0)
            pck_10.append(1)
        else:
            pck_2p5.append(0)
            pck_5.append(0)
            pck_10.append(0)


def infer_mask_locations(test_dataloader, model, args, test_file_names, logger):
    logger.info(
        "Running fast inference%s",
        (
            " without metric evaluation"
            if args.skip_metrics
            else " with metric evaluation"
        ),
    )
    progress = None
    summary_metric_meters = {}
    num_kpt = args.num_classes - 1
    centroid_pred_errs = {}
    centroid_pres_errs = {}
    pck_2p5 = []
    pck_5 = []
    pck_10 = []
    if not args.skip_metrics:
        batch_time = AverageMeter("Forward Time", ":2.2f")
        data_time = AverageMeter("Data Time", ":2.2f")
        progress_meter_list = [batch_time, data_time]
        progress_meter_list = add_metrics_meters(
            progress_meter_list, args.metric_fns, args.num_classes
        )
        progress = ProgressMeter(len(test_dataloader), progress_meter_list)
        metric_meter_start_idx = 2
        centroid_pred_errs = {class_idx: [] for class_idx in range(1, args.num_classes)}
        centroid_pres_errs = {class_idx: [] for class_idx in range(1, args.num_classes)}

    model.eval()
    mask_locations_path = args.output_dir / "mask_locations.jsonl"
    save_keypoint_images = not args.skip_output_images
    frame_idx = 0
    start_time = time.time()
    data_time_start = start_time if not args.skip_metrics else None
    with torch.inference_mode(), open(mask_locations_path, "w") as mask_locations_file:
        for step, (inputs, targets) in enumerate(
            tqdm(
                test_dataloader,
                desc="Inference",
                total=len(test_dataloader),
            )
        ):
            if not args.skip_metrics:
                data_time.update(time.time() - data_time_start)
                batch_time_start = time.time()
            c_gt = None
            c_pred = None
            inputs = inputs.cuda(non_blocking=True)
            inputs = inputs.contiguous(memory_format=torch.channels_last)
            if not args.skip_metrics:
                targets = targets.long().cuda(non_blocking=True).squeeze(1)
            with _get_autocast_context(args):
                outputs = _forward_logits(model, inputs, args)
            if not args.skip_metrics:
                metrics, metric_dict = get_metrics(
                    outputs, targets, args.metric_fns, args
                )
                for metric_name, metric_value in metric_dict.items():
                    if metric_name not in summary_metric_meters:
                        summary_metric_meters[metric_name] = AverageMeter(
                            metric_name, ":.4f"
                        )
                    summary_metric_meters[metric_name].update(
                        metric_value, inputs.size(0)
                    )

                metric_meter_idx = metric_meter_start_idx
                for metric_values in metrics:
                    for cls in range(1, args.num_classes):
                        progress_meter_list[metric_meter_idx].update(
                            metric_values[cls - 1], inputs.size(0)
                        )
                        metric_meter_idx += 1

                centroid_outputs = _prepare_outputs_for_centroid(outputs, args)
                for batch_offset in range(outputs.shape[0]):
                    err_list, pres_gt, pres, c_gt, c_pred = (
                        _normalize_centroid_error_result(
                            centroid_error(
                                centroid_outputs[batch_offset : batch_offset + 1],
                                targets[batch_offset : batch_offset + 1],
                                args,
                            ),
                            args,
                        )
                    )
                    if len(err_list) != num_kpt:
                        raise ValueError(
                            "Centroid metric output size does not match the configured "
                            f"foreground classes ({len(err_list)} vs {num_kpt})."
                        )
                    for class_idx, err in enumerate(err_list, start=1):
                        centroid_pred_errs[class_idx].append(err)
                        centroid_pres_errs[class_idx].append(
                            bool(pres_gt[class_idx - 1]) ^ bool(pres[class_idx - 1])
                        )
                    _update_pck_counts(err_list, pck_2p5, pck_5, pck_10)

                batch_time.update(time.time() - batch_time_start)
            output_classes = outputs.argmax(dim=1).detach().cpu().numpy()
            batch_size = output_classes.shape[0]
            batch_file_names = test_file_names[frame_idx : frame_idx + batch_size]

            for batch_offset, (file_name, mask_array) in enumerate(
                zip(batch_file_names, output_classes)
            ):
                current_frame_idx = frame_idx + batch_offset
                mask_locations = _get_mask_locations(mask_array, args.num_classes)
                frame_record = {
                    "frame_idx": int(current_frame_idx),
                    "frame_path": str(file_name),
                    "mask_locations": mask_locations,
                }
                mask_locations_file.write(json.dumps(frame_record) + "\n")
                if (
                    save_keypoint_images
                    and current_frame_idx % args.save_output_freq == 0
                ):
                    for i in range(1, args.num_classes):
                        _save_mask_overlay(
                            file_name,
                            args.output_dir / f"{current_frame_idx}.png",
                            mask_array,
                            c_gt,
                            c_pred,
                            args,
                        )
                    # _save_keypoint_overlay(
                    #     file_name,
                    #     args.output_dir / f"{current_frame_idx}.png",
                    #     mask_locations,
                    #     args,
                    # )
            frame_idx += batch_size
            if progress is not None and step % args.print_freq == 0:
                progress.display(step, logger=logger)
            if not args.skip_metrics:
                data_time_start = time.time()

    elapsed = max(time.time() - start_time, 1e-6)
    if not args.skip_metrics:
        assert len(pck_2p5) == len(pck_5) == len(pck_10)

        pck_2p5_mean = _safe_mean(pck_2p5)
        pck_5_mean = _safe_mean(pck_5)
        pck_10_mean = _safe_mean(pck_10)
        if pck_2p5_mean is None:
            logger.warning(
                "PCK at 2.5 pixels is undefined (no valid keypoint samples)."
            )
        else:
            logger.info(
                "Percentage of Correct Keypoints (PCK) at 2.5 pixels: %.2f%%",
                pck_2p5_mean * 100,
            )
        if pck_5_mean is None:
            logger.warning("PCK at 5 pixels is undefined (no valid keypoint samples).")
        else:
            logger.info(
                "Percentage of Correct Keypoints (PCK) at 5 pixels: %.2f%%",
                pck_5_mean * 100,
            )
        if pck_10_mean is None:
            logger.warning("PCK at 10 pixels is undefined (no valid keypoint samples).")
        else:
            logger.info(
                "Percentage of Correct Keypoints (PCK) at 10 pixels: %.2f%%",
                pck_10_mean * 100,
            )

        for class_idx in range(1, args.num_classes):
            det_mean = _safe_mean(centroid_pres_errs[class_idx])
            det_std = _safe_std(centroid_pres_errs[class_idx])
            if det_mean is None:
                logger.warning(
                    "Centroid detection error for class %d is undefined (no samples).",
                    class_idx,
                )
            else:
                logger.info(
                    "Avg. Centroid Detection Error %d: %s",
                    class_idx,
                    (1.0 - det_mean) * 100,
                )
            if det_std is None:
                logger.warning(
                    "Centroid detection std for class %d is undefined (no samples).",
                    class_idx,
                )
            else:
                logger.info(
                    "Std. Centroid Detection Error %d: %s",
                    class_idx,
                    det_std * 100,
                )

            valid_pred_errs = [
                err for err in centroid_pred_errs[class_idx] if not math.isnan(err)
            ]
            pred_mean = _safe_mean(valid_pred_errs)
            pred_std = _safe_std(valid_pred_errs)
            if pred_mean is None or pred_std is None:
                logger.warning(
                    "Centroid prediction error for class %d is undefined (no valid samples).",
                    class_idx,
                )
            else:
                logger.info(
                    "Avg. Centroid Prediction Error Class %d: %s +/- %s",
                    class_idx,
                    pred_mean,
                    pred_std,
                )

        metric_meter_idx = metric_meter_start_idx
        for metric_fn in args.metric_fns:
            for cls in range(1, args.num_classes):
                logger.info(
                    "Avg. %s for class %d: %s",
                    metric_fn,
                    cls,
                    progress_meter_list[metric_meter_idx].avg,
                )
                metric_meter_idx += 1

        avg_metric_dict = {
            metric_name: meter.avg
            for metric_name, meter in summary_metric_meters.items()
        }
        if avg_metric_dict:
            logger.info("Avg. Metrics: %s", avg_metric_dict)

    logger.info("Saved %d frame records to %s", frame_idx, mask_locations_path)
    if save_keypoint_images:
        logger.info(
            "Saved keypoint overlay images every %d frames to %s",
            args.save_output_freq,
            args.output_dir,
        )
    logger.info(
        "Fast inference throughput: %.2f frames/sec (batch_size=%d, amp=%s)",
        frame_idx / elapsed,
        args.test_batch_size,
        "off" if args.disable_amp else "on",
    )


def main_worker(args):
    args.mode = "testing"
    args.data_dir = Path(args.data_dir)
    args.log_dir = Path(os.path.join(args.expt_savedir, args.expt_name, "logs"))
    args.output_dir = Path(os.path.join(args.expt_savedir, args.expt_name, "outputs"))
    for dir in [args.log_dir, args.output_dir]:
        if not dir.is_dir():
            print(f"Creating {dir.resolve()} if non-existent")
            dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(os.path.join(args.log_dir, "log.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if not torch.cuda.is_available():
        raise SystemError("GPU device not found! Fast inference expects CUDA.")

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    test_file_names = _get_dataset_filenames(args)
    test_dataloader = _build_test_dataloader(args, test_file_names)

    model = get_model(args)
    model = model.cuda()
    model = model.to(memory_format=torch.channels_last)
    cudnn.benchmark = True

    if args.expt_savedir is not None and args.expt_name is not None:
        if args.pth_file_name is not None:
            args.load_wts_model = os.path.join(
                args.expt_savedir, args.expt_name, "ckpts", args.pth_file_name
            )
        else:
            args.load_wts_model = os.path.join(
                args.expt_savedir, args.expt_name, "ckpts", "model_050.pth"
            )
    model, _, load_flag = load_model_weights(
        model, args.load_wts_model, args.model_type
    )
    if load_flag:
        logger.info("Model weights loaded from {}".format(args.load_wts_model))
    else:
        logger.info("No model weights loaded")

    if "TAPNet" in args.model_type:
        logger.info("Preparing TAPNet attention maps")
        save_attention_maps(model, args)

    infer_mask_locations(test_dataloader, model, args, test_file_names, logger)
    return


if __name__ == "__main__":
    main()
