from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .swap_coco_metric import SwapCocoMetric

__all__ = [
    "SwapCocoMetric",
    "calculate_confusion_matrix_from_arrays",
    "get_dice",
    "get_distance_rmse",
    "get_jaccard",
    "get_metrics",
]


def __getattr__(name):
    if name != "SwapCocoMetric":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from .swap_coco_metric import SwapCocoMetric as _SwapCocoMetric
    except ModuleNotFoundError as exc:
        if exc.name not in {"mmengine", "mmpose", "xtcocotools"}:
            raise
        raise ModuleNotFoundError(
            "SwapCocoMetric requires mmengine, mmpose, and xtcocotools."
        ) from exc
    return _SwapCocoMetric


def _num_foreground_classes(args):
    return max(int(getattr(args, "num_classes", 1)) - 1, 1)


def _normalize_pose_metric_inputs(outputs, targets):
    if isinstance(outputs, Mapping) and isinstance(targets, Mapping):
        return [outputs], [targets]

    if isinstance(outputs, Sequence) and isinstance(targets, Sequence):
        if len(outputs) != len(targets):
            raise ValueError(
                "COCO metric requires outputs and targets to have the same length."
            )
        return list(outputs), list(targets)

    raise ValueError(
        "COCO metric requires COCO-style prediction/target dicts or sequences of dicts."
    )


def _get_tensor_class_masks(outputs, targets):
    output_classes = outputs.detach().cpu().numpy().argmax(axis=1)
    target_classes = targets.detach().cpu().numpy()
    if target_classes.ndim == 4 and target_classes.shape[1] == 1:
        target_classes = target_classes[:, 0]
    return output_classes, target_classes


def _mask_to_keypoints(mask, num_keypoints):
    keypoints = np.zeros((num_keypoints, 3), dtype=np.float64)
    for keypoint_idx in range(num_keypoints):
        coords = np.argwhere(mask == (keypoint_idx + 1))
        if coords.size == 0:
            continue
        keypoints[keypoint_idx, 0] = float(np.mean(coords[:, 1]))
        keypoints[keypoint_idx, 1] = float(np.mean(coords[:, 0]))
        keypoints[keypoint_idx, 2] = 2.0
    return keypoints


def _swap_keypoints(keypoints, swap_indices):
    swapped = keypoints.copy()
    first_idx, second_idx = swap_indices
    if max(first_idx, second_idx) >= keypoints.shape[0]:
        return swapped
    swapped[[first_idx, second_idx]] = swapped[[second_idx, first_idx]]
    return swapped


def _mask_bbox(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None

    x_min = float(np.min(coords[:, 1]))
    x_max = float(np.max(coords[:, 1]))
    y_min = float(np.min(coords[:, 0]))
    y_max = float(np.max(coords[:, 0]))
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    return (x_min, y_min, width, height)


def _compute_oks_components(pred_keypoints, gt_keypoints, bbox, sigma):
    visible = gt_keypoints[:, 2] > 0
    if not np.any(visible):
        pred_visible = pred_keypoints[:, 2] > 0
        empty_match = float(not np.any(pred_visible))
        return empty_match, np.zeros(gt_keypoints.shape[0], dtype=np.float64), visible

    vars_ = np.full(gt_keypoints.shape[0], (2.0 * sigma) ** 2, dtype=np.float64)
    scale = (bbox[2] ** 2 + bbox[3] ** 2) / 2.0 + np.spacing(1)
    dx = pred_keypoints[:, 0] - gt_keypoints[:, 0]
    dy = pred_keypoints[:, 1] - gt_keypoints[:, 1]
    keypoint_scores = np.exp(-((dx**2 + dy**2) / vars_ / scale / 2.0))
    oks = float(np.mean(keypoint_scores[visible]))
    return oks, keypoint_scores, visible


def _get_default_tip_swap_indices(num_keypoints):
    if num_keypoints < 2:
        return (0, 0)
    return (num_keypoints - 2, num_keypoints - 1)


def _get_coco_metrics_from_tensors(outputs, targets, args):
    output_classes, target_classes = _get_tensor_class_masks(outputs, targets)
    num_keypoints = _num_foreground_classes(args)
    sigma = float(getattr(args, "coco_sigma", 0.107))
    swap_indices = tuple(
        getattr(args, "coco_tip_swap_indices", _get_default_tip_swap_indices(num_keypoints))
    )

    per_keypoint_sum = np.zeros(num_keypoints, dtype=np.float64)
    per_keypoint_count = np.zeros(num_keypoints, dtype=np.float64)
    sample_oks_scores = []

    for pred_mask, gt_mask in zip(output_classes, target_classes):
        gt_bbox = _mask_bbox(gt_mask)
        if gt_bbox is None:
            sample_oks_scores.append(float(not np.any(pred_mask > 0)))
            continue

        pred_keypoints = _mask_to_keypoints(pred_mask, num_keypoints)
        gt_keypoints = _mask_to_keypoints(gt_mask, num_keypoints)

        base_oks, base_scores, base_visible = _compute_oks_components(
            pred_keypoints, gt_keypoints, gt_bbox, sigma
        )
        swapped_gt_keypoints = _swap_keypoints(gt_keypoints, swap_indices)
        swapped_oks, swapped_scores, swapped_visible = _compute_oks_components(
            pred_keypoints, swapped_gt_keypoints, gt_bbox, sigma
        )

        if swapped_oks > base_oks:
            sample_oks_scores.append(swapped_oks)
            per_keypoint_sum[swapped_visible] += swapped_scores[swapped_visible]
            per_keypoint_count[swapped_visible] += 1
        else:
            sample_oks_scores.append(base_oks)
            per_keypoint_sum[base_visible] += base_scores[base_visible]
            per_keypoint_count[base_visible] += 1

    per_keypoint_metrics = np.divide(
        per_keypoint_sum,
        per_keypoint_count,
        out=np.zeros_like(per_keypoint_sum),
        where=per_keypoint_count > 0,
    )
    mean_oks = float(np.mean(sample_oks_scores)) if sample_oks_scores else 0.0
    return per_keypoint_metrics.tolist(), mean_oks, {"OKS": mean_oks}


def get_coco_metrics(outputs, targets, args):
    if torch.is_tensor(outputs) and torch.is_tensor(targets):
        return _get_coco_metrics_from_tensors(outputs, targets, args)

    pred_list, target_list = _normalize_pose_metric_inputs(outputs, targets)
    coco_metric = getattr(args, "coco_metric", None)

    if coco_metric is None:
        coco_metric_cls = __getattr__("SwapCocoMetric")
        coco_metric_kwargs = getattr(args, "coco_metric_kwargs", {})
        coco_metric = coco_metric_cls(**coco_metric_kwargs)

    dataset_meta = getattr(args, "dataset_meta", None)
    if dataset_meta is not None:
        coco_metric.dataset_meta = dataset_meta
    elif getattr(coco_metric, "dataset_meta", None) is None:
        raise ValueError(
            "COCO metric requires dataset metadata with keypoint sigmas on "
            "`args.dataset_meta` or `args.coco_metric.dataset_meta`."
        )

    eval_results = dict(coco_metric.compute_metrics(list(zip(pred_list, target_list))))
    primary_metric = float(eval_results.get("AP", next(iter(eval_results.values()))))
    per_class_metrics = [primary_metric] * _num_foreground_classes(args)
    return per_class_metrics, primary_metric, eval_results

def get_metrics(outputs, targets, metric_fns, args):
    metric_dict = {}
    output_classes = None
    target_classes = None
    metric_vals_per_class = []
    for metric_fn in metric_fns:
        metric = None
        if metric_fn == 'jaccard':
            raise NotImplementedError
        elif metric_fn == 'iou':
            if output_classes is None or target_classes is None:
                output_classes = outputs.data.cpu().numpy().argmax(axis=1)
                target_classes = targets.data.cpu().numpy()
            ious = {}
            iou_list = []
            for cls in range(args.num_classes):
                if cls==0:
                    continue # exclude background
                iou = get_jaccard(target_classes==cls, output_classes==cls)
                ious['iou_{}'.format(cls)] = iou
                iou_list.append(iou)
            metric = np.mean(list(ious.values()))
            metric_vals_per_class.append(iou_list)
        elif metric_fn == 'dice':
            if output_classes is None or target_classes is None:
                output_classes, target_classes = _get_tensor_class_masks(outputs, targets)
            dices = {}
            dice_list = []
            for cls in range(args.num_classes):
                if cls==0:
                    continue # exclude background
                dice = get_dice(target_classes==cls, output_classes==cls)
                dices['dice_{}'.format(cls)] = dice
                dice_list.append(dice)
            metric = np.mean(list(dices.values()))
            metric_vals_per_class.append(dice_list)
        elif metric_fn == 'rmse':
            if output_classes is None or target_classes is None:
                output_classes, target_classes = _get_tensor_class_masks(outputs, targets)
            dists = {}
            dist_list = []
            for cls in range(args.num_classes):
                if cls==0:
                    continue
                dist = get_distance_rmse(target_classes==cls, output_classes==cls)
                if np.isnan(dist):
                    print(f'Warning: RMSE distance for class {cls} is NaN, skipping...')
                    continue  # skip if distance is NaN
                dists['rmse_{}'.format(cls)] = dist
                dist_list.append(dist)
            metric = np.mean(list(dists.values()))
            metric_vals_per_class.append(dist_list)
        elif metric_fn == "coco":
            coco_vals, metric, coco_stats = get_coco_metrics(outputs, targets, args)
            metric_vals_per_class.append(coco_vals)
            for stat_name, stat_value in coco_stats.items():
                metric_dict[f"metric_coco_{stat_name}"] = float(stat_value)
        else:
            raise ValueError(f'Metric function {metric_fn} not implemented')
        metric_dict['metric_' + metric_fn] = float(metric)
    return metric_vals_per_class, metric_dict

def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(axis=-2).sum(axis=-1)
    union = y_true.sum(axis=-2).sum(axis=-1) + y_pred.sum(axis=-2).sum(axis=-1)
    return ((intersection + epsilon) / (union - intersection + epsilon))[0]

def get_dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def get_distance_rmse(y_true, y_pred):
    """
    Compute RMSE of centroid distances between predicted and true masks.
    
    Parameters:
        y_true: np.ndarray of shape (B, C, H, W) - ground truth masks
        y_pred: np.ndarray of shape (B, C, H, W) - predicted masks

    Returns:
        rmse: scalar value representing RMSE of centroid distances
    """
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    
    batch_size = y_true.shape[0]
    distances = []

    for i in range(batch_size):
        # For each item in batch, compute centroid across all channels
        true_mask = y_true[i]
        pred_mask = y_pred[i]
        
        true_points = np.argwhere(true_mask > 0)
        pred_points = np.argwhere(pred_mask > 0)

        if true_points.size == 0 or pred_points.size == 0:
            # If one of the masks is empty, skip or append NaN
            distances.append(np.nan)
            continue

        true_midpoint = np.mean(true_points, axis=0)
        pred_midpoint = np.mean(pred_points, axis=0)
        dist = float(np.linalg.norm(true_midpoint - pred_midpoint))
        distances.append(dist)

    distances = np.array(distances)
    distances = distances[~np.isnan(distances)]  # remove invalid ones if any

    rmse = np.sqrt(np.mean(distances ** 2)) if distances.size > 0 else np.nan
    return rmse
def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((ground_truth.flatten(), prediction.flatten())).T
    confusion_matrix, _ = np.histogramdd(replace_indices,
                                        bins=(nr_labels, nr_labels),
                                        range=[(0, nr_labels), (0, nr_labels)])
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix

