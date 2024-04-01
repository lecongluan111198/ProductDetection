import torch.nn as nn
import numpy as np
import torch
# from optimum.onnxruntime import ORTModelForImageClassification
# from transformers import ViTForImageClassification, AutoImageProcessor
import transformers
# import optimum.onnxruntime as onxx

class ViewMode(nn.Module):
    def __init__(self, model_name, weight_path, labels, is_oxx = False):
        super().__init__()
        self.model_name = model_name
        self.weight_path = weight_path
        # if is_oxx:
        #     self.vit  = onxx.ORTQuantizer.from_pretrained(
        #         self.model_name,
        #         num_labels=len(labels),
        #         id2label={str(i): c for i, c in enumerate(labels)},
        #         label2id={c: str(i) for i, c in enumerate(labels)},
        #         from_transformers=True
        #     )
        # else:
        self.vit  = transformers.ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        outputs = self.vit(**x)
        k = outputs.logits
        k = self.sm(k)
        return k

class ViewClassifier:
    def __init__(self, weight_path, labels, is_oxx = False):
        super().__init__()
        self.model_name = 'google/vit-base-patch16-224-in21k'
        self.weight_path = weight_path
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(self.model_name)
        self.vit = ViewMode(self.model_name, self.weight_path, labels, is_oxx)
        self.vit.load_state_dict(torch.load(weight_path))
        if torch.cuda.is_available():
            self.vit.cuda()

    def predict(self, img):
        with torch.no_grad():
            inputs = self.image_processor(img, return_tensors="pt")
            outputs = self.vit({'pixel_values': inputs['pixel_values'].cuda()})
            return outputs