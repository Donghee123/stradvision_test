import fiftyone as fo
import fiftyone.zoo as foz
import os
import torch
import torch.utils.data
import torchvision
import torchvision.ops 
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from PIL import Image
from torchvision.transforms import functional as func
import cv2
import numpy as np
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
import pickle

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

inferencemode = 'COCO-Detection'
expendName = '.yaml'

load_model_names = ['retinanet_R_50_FPN_1x', 'retinanet_R_50_FPN_3x', 'retinanet_R_101_FPN_3x']

# make config
cfg = get_cfg()

for load_model_name in load_model_names:
    
    # load model
    cfg.merge_from_file(model_zoo.get_config_file(os.path.join(inferencemode, load_model_name + expendName)))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(os.path.join(inferencemode, load_model_name + expendName))
    predictor = DefaultPredictor(cfg)
    predictor.model.eval()
    print("Model ready")

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        dataset_name="evaluate-detections-tutorial",
    )

    dataset.persistent = True

    # Print some information about the dataset
    print(dataset)

    # Print a ground truth detection
    sample = dataset.first()
    print(sample.ground_truth.detections[0])

    # Choose a random subset of 100 samples to add predictions to
    predictions_view = dataset.take(5000,seed=51) # limit size

    #Get class list
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    classes = metadata.get("thing_classes", None)

    predictions = {}

    # Add predictions to samples
    with fo.ProgressBar() as pb:
        for sample in pb(predictions_view):

            # Load image
            image = cv2.imread(sample.filepath)
            h, w, c = image.shape

            # Perform inference
            preds = predictor(image)
            instances = preds["instances"].to('cpu')

            boxes = instances.pred_boxes.tensor.numpy()
            boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            boxes = np.array(boxes)# .tolist()
            scores = np.array(instances.scores)# .tolist()
            labels = np.array(instances.pred_classes) # .tolist()

            # Save predictions to dataset
            filename = os.path.basename(sample.filepath)
            predictions[filename] = [boxes, labels, scores]


    # save
    with open(os.path.join(ROOT_PATH, 'predictions', f'{load_model_name}_predictions.pickle'), 'wb') as f:
        pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)
