import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import functools
from . import profiler
from .. import config

def is_cluster(cluster1, cluster2, type, eps, min_samples, i, j):
    if type == 0:
        points = [ymin for xmin, ymin, xmax, ymax, cls in cluster1] + [ymin for  xmin, ymin, xmax, ymax, cls in cluster2]
    elif type == 1:
        points = [ymax for  xmin, ymin, xmax, ymax, cls in cluster1] + [ymax for  xmin, ymin, xmax, ymax, cls in cluster2]
    else:
        points = [ymin + (ymax - ymin) / 2 for  xmin, ymin, xmax, ymax, cls in cluster1] + [ymin + (ymax - ymin) / 2 for  xmin, ymin, xmax, ymax, cls in cluster2]
    points = np.array(points).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    if config.ENV == 'local' and len(np.unique(clustering.labels_)) == 1:
        print('is_cluster', len(np.unique(clustering.labels_)), i, j)
    return len(np.unique(clustering.labels_)) == 1

def is_overlap(cluster1, cluster2, i, j):
    box1 = cluster1[len(cluster1) - 1]
    box2 = cluster2[0]
    box1 = [box1[0], box1[1], box1[2], box1[3]]
    box2 = [box2[0], box2[1], box2[2], box2[3]]
    xcenter1 = box1[0] + (box1[2] - box1[0]) / 2
    xcenter2 = box2[0] + (box2[2] - box2[0]) / 2
    gap = (box2[0] - box1[2]) / abs(xcenter1 - xcenter2)
    iou = cal_iou(box1, box2)
    if config.ENV == 'local':
        print(box2[0], box1[2], xcenter1, xcenter2)
        print('is_overlap', iou , gap, i , j, len(cluster1), len(cluster2))
    return iou > 0.0 or (gap >= 0.0 and gap <= 0.1)

def is_centers_not_too_far(cluster1, cluster2, i, j):
    box1 = cluster1[len(cluster1) - 1]
    box2 = cluster2[0]
    box1 = [box1[0], box1[1], box1[2], box1[3]]
    box2 = [box2[0], box2[1], box2[2], box2[3]]
    ycenter1 = box1[1] + (box1[3] - box1[1]) / 2
    ycenter2 = box2[1] + (box2[3] - box2[1]) / 2
    return abs(ycenter1 - ycenter2) < max(abs(box1[3] - ycenter1), abs(box2[3] - ycenter2))

def merge_cluster(clusters, eps=20, min_samples=1, request_id = None):
    try:
        profiler.push("merge_cluster", request_id)
        n_cluster = len(clusters.values())
        merged = {}
        for i in range(n_cluster):
            if len(clusters[i]) == 0:
                continue
            merged[i] = clusters[i]
            for j in range(i + 1, n_cluster):
                if len(clusters[j]) == 0:
                    continue
                if is_cluster(clusters[i], clusters[j], 0, eps, min_samples, i, j):
                    merged[i] += clusters[j]
                    clusters[j] = []
                elif is_cluster(clusters[i], clusters[j], 1, eps, min_samples, i, j):
                    merged[i] += clusters[j]
                    clusters[j] = []
                elif is_overlap(clusters[i], clusters[j], i, j) and is_centers_not_too_far(clusters[i], clusters[j], i, j):
                    merged[i] += clusters[j]
                    clusters[j] = []
        return merged
    finally:
        profiler.pop("merge_cluster", request_id)

def build_layout(labels, eps=20, min_samples=1, request_id = None):
    try:
        profiler.push("build_layout", request_id)
        if len(labels) == 0:
            return []
        bboxes = [(l['xmin'], l['ymin'], l['xmax'], l['ymax'], l['cls']) for l in labels]
        def compare(v1, v2):
            return v1[0] - v2[0]
        bboxes = sorted(bboxes, key=functools.cmp_to_key(compare))
        y_centers = [ymin + (ymax - ymin) / 2 for xmin, ymin, xmax, ymax, cls in bboxes]
        y_centers = np.array(y_centers).reshape(-1, 1)
        points = y_centers
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)  # Perform DBSCAN on y-coordinates
        clusters = {}
        for label, bbox in zip(clustering.labels_, bboxes):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(bbox)

        if config.ENV == 'local':
            print(clusters)
            print(len(clusters))
        clusters = merge_cluster(clusters, eps, min_samples, request_id = request_id)
        if config.ENV == 'local':
            print(clusters)
            print(len(clusters))
        def compare_cluster(v1, v2):
            return v1[1] - v2[1]
        clusters_sorted = sorted([(label, np.min([bb[1] for bb in bbs]), bbs) for label, bbs in clusters.items()], key=functools.cmp_to_key(compare_cluster))

        line_bboxes = []
        for label, min_y, bbs in clusters_sorted:
            line_bboxes.append([bb[4] for bb in bbs])
        return line_bboxes
    finally:
        profiler.pop("build_layout", request_id)



def remove_overlap(labels, max_iou=0.8):
    ret = []
    bboxes = [(l['xmin'], l['ymin'], l['xmax'], l['ymax']) for l in labels]
    for i in range(len(bboxes)):
        is_choice = True
        for j in range(len(bboxes)):
            if i == j:
                continue
            iou = cal_iou(bboxes[i], bboxes[j])
            if iou >= max_iou:
                if labels[i]['conf'] < labels[j]['conf']:
                    is_choice = False
                    break
        if is_choice:
            ret.append(labels[i])
    return ret

def cal_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    w = xB - xA
    h = yB - yA
    if w < 0 or h < 0:
        return 0.0
    # compute the area of intersection rectangle
    interArea = w * h
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
