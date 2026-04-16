from collections import OrderedDict
from typing import Sequence, Tuple

import numpy as np
from xtcocotools.cocoeval import COCOeval

from mmpose.evaluation.metrics import CocoMetric
from mmpose.registry import METRICS


class RecommendedOksCocoEval(COCOeval):
    """COCOeval variant matching the ROBUST-MIPS recommended OKS metric.

    The paper treats Tip1/Tip2 as an unordered pair, so evaluation should
    consider both the original and tip-swapped ground truth pose and keep the
    better OKS. It also replaces the standard area-based scale with the scaled
    bbox diagonal: s^2 = (w^2 + h^2) / 2.
    """

    def __init__(
        self,
        *args,
        swap_indices: Tuple[int, int] = (2, 3),
        oks_sigma: float = 0.107,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.swap_indices = tuple(int(idx) for idx in swap_indices)
        self.oks_sigma = float(oks_sigma)
        sigmas = np.asarray(self.sigmas, dtype=np.float64)
        self.sigmas = np.full(sigmas.shape, self.oks_sigma, dtype=np.float64)

    @staticmethod
    def _scaled_diagonal_area(bbox: Sequence[float]) -> float:
        width = float(bbox[2])
        height = float(bbox[3])
        return (width**2 + height**2) / 2.0

    def _swap_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        swapped = keypoints.copy()
        if keypoints.size % 3 != 0:
            return swapped

        first_idx, second_idx = self.swap_indices
        num_keypoints = keypoints.size // 3
        if max(first_idx, second_idx) >= num_keypoints:
            return swapped

        first_slice = slice(first_idx * 3, first_idx * 3 + 3)
        second_slice = slice(second_idx * 3, second_idx * 3 + 3)
        swapped[first_slice], swapped[second_slice] = (
            swapped[second_slice].copy(),
            swapped[first_slice].copy(),
        )
        return swapped

    def _compute_oks_against_gt(
        self, detection: np.ndarray, gt_keypoints: np.ndarray, bbox: Sequence[float]
    ) -> float:
        vars = (self.sigmas * 2.0) ** 2
        scale = self._scaled_diagonal_area(bbox) + np.spacing(1)

        xd = detection[0::3]
        yd = detection[1::3]
        xg = gt_keypoints[0::3]
        yg = gt_keypoints[1::3]
        vg = gt_keypoints[2::3]

        visible = vg > 0
        if np.any(visible):
            dx = xd - xg
            dy = yd - yg
            errors = (dx**2 + dy**2) / vars / scale / 2.0
            errors = errors[visible]
            return float(np.sum(np.exp(-errors)) / errors.shape[0])

        x0 = bbox[0] - bbox[2]
        x1 = bbox[0] + bbox[2] * 2
        y0 = bbox[1] - bbox[3]
        y1 = bbox[1] + bbox[3] * 2
        zeros = np.zeros_like(self.sigmas, dtype=np.float64)
        dx = np.maximum(zeros, x0 - xd) + np.maximum(zeros, xd - x1)
        dy = np.maximum(zeros, y0 - yd) + np.maximum(zeros, yd - y1)
        errors = (dx**2 + dy**2) / vars / scale / 2.0
        return float(np.sum(np.exp(-errors)) / errors.shape[0])

    def computeOks(self, imgId, catId):
        if self.params.iouType != "keypoints":
            return super().computeOks(imgId, catId)

        params = self.params
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-det[self.score_key] for det in dts], kind="mergesort")
        dts = [dts[idx] for idx in inds]
        if len(dts) > params.maxDets[-1]:
            dts = dts[: params.maxDets[-1]]
        if len(gts) == 0 or len(dts) == 0:
            return []

        ious = np.zeros((len(dts), len(gts)), dtype=np.float64)
        for gt_idx, gt in enumerate(gts):
            gt_keypoints = np.asarray(gt["keypoints"], dtype=np.float64)
            swapped_gt_keypoints = self._swap_keypoints(gt_keypoints)
            bbox = gt["bbox"]

            for det_idx, det in enumerate(dts):
                det_keypoints = np.asarray(det["keypoints"], dtype=np.float64)
                base_oks = self._compute_oks_against_gt(
                    det_keypoints, gt_keypoints, bbox
                )
                swapped_oks = self._compute_oks_against_gt(
                    det_keypoints, swapped_gt_keypoints, bbox
                )
                ious[det_idx, gt_idx] = max(base_oks, swapped_oks)

        return ious


@METRICS.register_module()
class SwapCocoMetric(CocoMetric):
    """COCO metric with tip-swap-invariant OKS for surgical tools."""

    default_prefix = "swap_coco"

    def __init__(
        self,
        *args,
        tip_swap_indices: Tuple[int, int] = (2, 3),
        oks_sigma: float = 0.107,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if len(tip_swap_indices) != 2:
            raise ValueError("tip_swap_indices must contain exactly two keypoint ids.")
        self.tip_swap_indices = (
            int(tip_swap_indices[0]),
            int(tip_swap_indices[1]),
        )
        self.oks_sigma = float(oks_sigma)

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        res_file = f"{outfile_prefix}.keypoints.json"
        coco_det = self.coco.loadRes(res_file)
        dataset_meta = self.dataset_meta
        if dataset_meta is None:
            raise ValueError(
                "SwapCocoMetric requires dataset_meta with keypoint sigmas."
            )
        coco_eval = RecommendedOksCocoEval(
            self.coco,
            coco_det,
            self.iou_type,
            dataset_meta["sigmas"],
            self.use_area,
            swap_indices=self.tip_swap_indices,
            oks_sigma=self.oks_sigma,
        )
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if self.iou_type == "keypoints_crowd":
            stats_names = [
                "AP",
                "AP .5",
                "AP .75",
                "AR",
                "AR .5",
                "AR .75",
                "AP(E)",
                "AP(M)",
                "AP(H)",
            ]
        else:
            stats_names = [
                "AP",
                "AP .5",
                "AP .75",
                "AP (M)",
                "AP (L)",
                "AR",
                "AR .5",
                "AR .75",
                "AR (M)",
                "AR (L)",
            ]

        return list(zip(stats_names, coco_eval.stats))
