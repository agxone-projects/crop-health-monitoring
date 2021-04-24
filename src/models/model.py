import os

import detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


class CornNetv2():
    def __init__(self, weights, n_classes, train_sets=()):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.DATASETS.TRAIN = train_sets
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 4
        self.cfg.MODEL.WEIGHTS = weights
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes

    def compile(self, n_iter, output_folder, resume=False):
        self.cfg.SOLVER.IMS_PER_BATCH = 5
        self.cfg.SOLVER.BASE_LR = 0.0015
        self.cfg.SOLVER.MAX_ITER = n_iter
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

        self.cfg.OUTPUT_DIR = output_folder
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        self.trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=resume)

    def fit(self):
        self.trainer.train()

    def evaluate(self, test_set, output_folder):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        predictor = DefaultPredictor(self.cfg)
        evaluator = COCOEvaluator(
            test_set, self.cfg, False, output_dir=output_folder)
        val_loader = build_detection_test_loader(self.cfg, test_set)
        inference_on_dataset(self.trainer.model, val_loader, evaluator)

    def get_predictor(self, score_thresh=0.7):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        predictor = DefaultPredictor(self.cfg)
        return predictor
