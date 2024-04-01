import torch
from ..utils import pytorch_loader, box_utils
from .. import config

class BottleDetector:
    def __init__(self, path):
        self.model = pytorch_loader.load_yolo_model(path)
        if torch.cuda.is_available():
            self.model.cuda()

    def predict(self, img):
        group_cnt = {}
        labels = []
        with torch.no_grad():
            results = self.model(img)
        df = results.pandas().xyxy[0]
        for i in range(len(df['name'])):
            cls = df['class'][i]
            name = df['name'][i]
            conf = df['confidence'][i]
            xmin = df['xmin'][i]
            ymin = df['ymin'][i]
            xmax = df['xmax'][i]
            ymax = df['ymax'][i]
            if conf >= config.THRESH_HOLD:
                group = name
                if group in config.CLS_2_GROUP:
                    group = config.CLS_2_GROUP[name]
                labels.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'cls': group, 'conf': conf})
        if len(labels) > 0:
            labels = box_utils.remove_overlap(labels)
        for l in labels:
            group = l['cls']
            if group not in group_cnt:
                group_cnt[group] = 1
            else:
                group_cnt[group] += 1
        return group_cnt, labels