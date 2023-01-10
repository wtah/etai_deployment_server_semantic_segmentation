import json

import torch
from PIL import Image
from rest_framework import generics
import sys
sys.path.insert(0, "ext_repo/Mask2Former")
sys.path.insert(0, "ext_repo/detectron2")
import tempfile
from pathlib import Path
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from mask2former import add_maskformer2_config


class Predictor():
    def setup(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file("ext_repo/Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
        cfg.MODEL.WEIGHTS = 'models/model_final_f07440.pkl'
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        self.predictor = DefaultPredictor(cfg)
        self.coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

    def make_colour_mask(sefl, segment_mask):
        colour_table = np.zeros((256, 1, 3), dtype=np.uint8)
        colour_table[0] = [0, 0, 0]
        colour_table[1] = [255, 255, 255]
        colour_table[2] = [255, 0, 0]
        colour_table[3] = [0, 255, 0]
        colour_table[4] = [0, 0, 255]
        colour_table[5] = [255 ,0 ,255]
        colour_table[6] = [0, 255 ,255]

        colour_mask = cv2.applyColorMap(segment_mask, colour_table)
        return colour_mask

    def predict(self, im):
        im_ =np.array(im)
        outputs = self.predictor(im_)
        v = Visualizer(im_[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        panoptic_result = Image.fromarray(v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"),
                                              outputs["panoptic_seg"][1]).get_image())
        panoptic_result.thumbnail((100,100))
        v = Visualizer(im_[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        instance_result = Image.fromarray(v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image())
        instance_result.thumbnail((100,100))
        v = Visualizer(im_[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        semantic_result = Image.fromarray(v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image())
        instance_result.thumbnail((100,100))
        result = {'panoptic_result': np.array(panoptic_result).tolist(),
                  'instance_result': np.array(instance_result).tolist(),
                  'semantic_result': np.array(semantic_result).tolist()}

        return result

predictor = Predictor()
predictor.setup()

from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from imantics import Polygons, Mask

from model_deployment.serializers import TextPredictionSerializer, ImagePredictionSerializer
from model_deployment.models import TextPrediction, ImagePrediction

from etai_deployment_server import settings

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DetrImageProcessor, DetrForObjectDetection
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_text_model():
    if settings.INFERENCE_MODE=='text':
        return AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment"), \
               AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    else:
        return None ,None

def get_image_model():
    if settings.INFERENCE_MODE=='image':
        return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50"), \
            DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    else:
        return None ,None

### Example for Text based deployment
class TextPredictionListCreate(generics.ListCreateAPIView):
    queryset = TextPrediction.objects.all()
    serializer_class = TextPredictionSerializer
    permission_classes = []
    #tokenizer, model = get_text_model() commented to save memeory

    ### ENTRYPOINT FOR INFERENCE
    def perform_create(self, serializer):
        # Here you get the text string submitted for inference
        prediction = self.infer(serializer.validated_data['sample'])
        serializer.validated_data['prediction'] = prediction
        if settings.DO_SAVE_PREDICTIONS:
            serializer.save()

    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def infer(self, text):
        encoded_input = self.tokenizer(self.preprocess(text), return_tensors='pt')
        with torch.no_grad():
            output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy().tolist()
        return {'labels':scores}








### Example for Image based deployment
class ImagePredictionListCreate(generics.ListCreateAPIView):
    queryset = ImagePrediction.objects.all()
    serializer_class = ImagePredictionSerializer
    permission_classes = []

    # extractor, model = get_image_model() again commented to save memory

    ### ENTRYPOINT FOR INFERENCE
    def perform_create(self, serializer):
        prediction = self.infer(serializer.validated_data['sample'])
        serializer.validated_data['prediction'] = prediction
        if settings.DO_SAVE_PREDICTIONS:
            serializer.save()

    def preprocess(self, image):
        # Here you load the submitted image
        img = Image.open(self.request.FILES['sample'])
        # Resize image to know dimensions
        img.thumbnail((400,400))
        return img

    def process_logits(self, outputs, image):
        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        predictions=[]
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            predictions.append({'label': self.model.config.id2label[label.item()], 'score':round(score.item(), 3), 'box': box })
            print(
                f"Detected {self.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
        return predictions

    def infer(self, image):
        im = self.preprocess(image)
        preds = predictor.predict(im)
        return {'labels': json.dumps(preds)}
