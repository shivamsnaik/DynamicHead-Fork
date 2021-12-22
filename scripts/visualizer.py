from detectron2.engine import DefaultPredictor, default_setup, default_argument_parser
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from dyhead import add_dyhead_config
from extra import add_extra_config
import cv2
import argparse
import os
from itertools import chain
import tqdm
from detectron2.data.datasets import register_coco_instances
 
register_coco_instances("coco_docbank_train", {}, "/netscratch/naik/DynamicHead/output.json", "/ds-av/public_datasets/docbank_500k/raw/DocBank_500K_ori_img/")
register_coco_instances("coco_docbank_test", {}, "/netscratch/naik/DynamicHead/output.json", "/ds-av/public_datasets/docbank_500k/raw/DocBank_500K_ori_img/")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_dyhead_config(cfg)
    add_extra_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ('coco_docbank_train',)
    cfg.DATASETS.TEST = ('coco_docbank_test',)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)

if __name__ == "__main__":
#    args = default_argument_parser().parse_args()
    args = parse_args()
    print("Command Line Args", args)
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    scale=1
    dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TEST]))

    if cfg.MODEL.KEYPOINT_ON:
        dicts = filter_images_with_few_keypoints(dicts, 1)
    for dic in tqdm.tqdm(dicts):
        img = utils.read_image(dic["file_name"], "RGB")
        result = predictor(img)
        visualizer = Visualizer(img, metadata=metadata, scale=scale)
        vis = visualizer.draw_instance_predictions(result["instances"].to("cpu"))
        output(vis, os.path.basename(dic["file_name"]))
