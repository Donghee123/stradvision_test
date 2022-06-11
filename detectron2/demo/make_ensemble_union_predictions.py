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

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
PREDICTION_PATH = os.path.join(ROOT_PATH, 'predictions')

load_model_names = ['retinanet_R_50_FPN_1x', 'retinanet_R_50_FPN_3x', 'retinanet_R_101_FPN_3x']

# load
with open(os.path.join(PREDICTION_PATH, f'{load_model_names[0]}_predictions.pickle'), 'rb') as f:
    model1_predictions = pickle.load(f)

# load
with open(os.path.join(PREDICTION_PATH, f'{load_model_names[1]}_predictions.pickle'), 'rb') as f:
    model2_predictions = pickle.load(f)

# load
with open(os.path.join(PREDICTION_PATH, f'{load_model_names[2]}_predictions.pickle'), 'rb') as f:
    model3_predictions = pickle.load(f)

print('predictions ready')

dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        dataset_name="evaluate-detections-tutorial",
    )

dataset.persistent = True

# Print some information about the dataset
print(dataset)

sample = dataset.first()
# Print a ground truth detection
print(sample.ground_truth.detections[0])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(os.path.join(inferencemode, load_model_names[0] + expendName)))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(os.path.join(inferencemode, load_model_names[0] + expendName))

#Get class list
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
classes = metadata.get("thing_classes", None)

predictions = {}
overlabthreshold = 0.15

# Add predictions to samples
with fo.ProgressBar() as pb:
    for sample in pb(dataset):

        # Load image
        image = cv2.imread(sample.filepath)
        h, w, c = image.shape

        filename = os.path.basename(sample.filepath)

        # read by prediction dict
        prediction1 = model1_predictions[filename]
        prediction2 = model2_predictions[filename]
        prediction3 = model3_predictions[filename]

        boxes =  np.concatenate((prediction1[0], prediction2[0], prediction3[0]))
        labels = np.concatenate((prediction1[1], prediction2[1], prediction3[1]))
        scores = np.concatenate((prediction1[2], prediction2[2], prediction3[2]))
        
        tensor_boxes =  torch.tensor(boxes)
        tensor_scores = torch.tensor(scores)
        results = torchvision.ops.nms(tensor_boxes, tensor_scores, overlabthreshold)

        detections = []
        # Convert detections to FiftyOne format
        for nIndex, (label, score, box) in enumerate(zip(labels, scores, boxes)):
            # only use best F1 Score model on class
            if (nIndex in results) is False:
                continue

            #print(f'class {classes[label]} : {modelname}')
            
            x, y, width, height = box

            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height
            
            rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

            detections.append(
                fo.Detection(
                    label=classes[label],
                    bounding_box=rel_box,
                    confidence=score
                )
            )

        predictions[filename] = detections

            
# save
with open(os.path.join(ROOT_PATH, 'predictions', f'ensemble_union_predictions.pickle'), 'wb') as f:
    pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)
