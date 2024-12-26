import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from PIL import Image
from torchvision import transforms
import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from detectron2.evaluation import DatasetEvaluator
from . import utils
import pycocotools.mask as mask_utils

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


class UNIFIEDEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        allow_cached_coco=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
            allow_cached_coco (bool): Whether to use cached coco json from previous validation
                runs. You should set this to False if you need to use different validation data.
                Defaults to True.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        if use_fast_impl and (COCOeval_opt is COCOeval):
            self._logger.info("Fast COCO eval is not built. Falling back to official COCO eval.")
            use_fast_impl = False
        self._use_fast_impl = use_fast_impl

        # COCOeval requires the limit on the number of detections per image (maxDets) to be a list
        # with at least 3 elements. The default maxDets in COCOeval is [1, 10, 100], in which the
        # 3rd element (100) is used as the limit on the number of detections per image when
        # evaluating AP. COCOEvaluator expects an integer for max_dets_per_image, so for COCOeval,
        # we reformat max_dets_per_image into [1, 10, max_dets_per_image], based on the defaults.
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]
        else:
            max_dets_per_image = [1, 10, max_dets_per_image]
        self._max_dets_per_image = max_dets_per_image

        if tasks is not None and isinstance(tasks, CfgNode):
            kpt_oks_sigmas = (
                tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
            )
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")
        self.dataset_name = dataset_name

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator "
                    "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path, allow_cached=allow_cached_coco)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
            

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        if self._do_evaluation:
            self._kpt_oks_sigmas = kpt_oks_sigmas



    def reset(self):
        self._predictions = []



    def process(self, inputs, outputs):

        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            mask = output["instances"].pred_masks.detach().cpu().numpy()
            prediction["instances"] = [{'segmentation': mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]}]            
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset.
        """
        if self._distributed:
            # Assuming comm is a communication utility for distributed computing
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[CustomEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            # Assuming PathManager is a utility for file system operations
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)
        
        self._results = OrderedDict()
        self._eval_unified(predictions)

        return copy.deepcopy(self._results)


    def _eval_unified(self, predictions):
        """
        Args:
            pred_masks: a list of binary masks of shape (H, W) or (H, W, 1)
            gt_masks: a list of binary masks of shape (H, W) or (H, W, 1)
        """
        mae_scores=[]
        s_measure_scores=[]
        e_measure_scores=[]
        wfm_scores=[]
        ber_scores=[]
        f1_score=[]
        auc_score=[]
        f_max_scores=[]
        fm_scores=[]
        for prediction in predictions:
            instances=[instance for instance in prediction['instances']]

            pred_masks = [x["segmentation"] for x in instances]
            gt_masks = self._coco_api.loadAnns(self._coco_api.getAnnIds(imgIds=prediction["image_id"]))
            metrics = ["MAE","SM","EM","WFM","BER","F1","AUC","FM"]

            for pred_mask, gt_mask in zip(pred_masks, gt_masks):

                pred_binary_mask = np.array(mask_util.decode(pred_mask), dtype=np.float32)
                gt_binary_mask = np.array(mask_util.decode(gt_mask["segmentation"]), dtype=np.float32)

                result1, result2, result3, result4, result9 =utils.calc_metrics(pred_binary_mask,gt_binary_mask)
                if self.dataset_name != "pascal":
                    result5 = utils.calc_ber(pred_binary_mask,gt_binary_mask)
                    result6,result7,_,_ = utils.calc_f1(pred_binary_mask,gt_binary_mask)
                mae_scores.append(result4)
                s_measure_scores.append(result1)
                e_measure_scores.append(result2)
                wfm_scores.append(result3)
                if self.dataset_name != "pascal":
                    ber_scores.append(result5)
                    f1_score.append(result6)
                    auc_score.append(result7)
                fm_scores.append(result9)
        mean_mae = np.mean(mae_scores)
        mean_s_measure = np.mean(s_measure_scores)
        mean_e_measure = np.mean(e_measure_scores)
        mean_wfm = np.mean(wfm_scores)
        if self.dataset_name != "pascal":
            mean_ber = np.mean(ber_scores)
            mean_f1 = np.mean(f1_score)
            mean_auc = np.mean(auc_score)
        else:
            mean_ber = np.nan
            mean_f1 = np.nan
            mean_auc = np.nan

        mean_fm = np.mean(fm_scores)
        results = [mean_mae,mean_s_measure,mean_e_measure,mean_wfm,mean_ber,mean_f1,mean_auc,mean_fm]
        res = {
            metric: float(results[idx] if results[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }  
        formatted_results = [("MAE", {"Value": mean_mae}),
                            ("S-Measure", {"Value": mean_s_measure}),
                            ("E-Measure", {"Value": mean_e_measure}),
                            ("WFM", {"Value": mean_wfm}),
                            ("BER", {"Value": mean_ber}),
                            ("F1", {"Value": mean_f1}),
                            ("AUC", {"Value": mean_auc}),
                            ("FM", {"Value": mean_fm})]

        # Logging the results for visibility
        self._logger.info("Evaluation Results: ")
        for metric, result in formatted_results:
            self._logger.info(f"{metric}: {result['Value']}")
        self._results['unified_evaluation'] = res  
