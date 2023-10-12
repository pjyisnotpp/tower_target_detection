import os
import numpy as np
import cv2
import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from tower_dataset import Tower_Dataset

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('train_model2.pth'))
iou_threshold = 0.3
model.eval()
test_dict_path = 'data/JPEGImages/test'


def image_pre_treat(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=200)
    height, width = edges.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    color_image[:, :, 0] = edges  # 复制灰度通道到三个通道
    color_image[:, :, 1] = edges
    color_image[:, :, 2] = edges
    return color_image


def non_maximum_suppression(box_list, score_list, iou_threshold):
    selected_boxes_index = []
    sorted_index = sorted(range(len(score_list)), key=lambda i: score_list[i], reverse=True)
    while len(sorted_index) > 0:
        current_index = sorted_index[0]
        if score_list[current_index] >= 0.3:
            selected_boxes_index.append(current_index)
        current_box = box_list[current_index]
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        del sorted_index[0]
        for index in sorted_index[:]:
            box = box_list[index]
            area = (box[2] - box[0]) * (box[3] - box[1])
            x1 = max(current_box[0], box[0])
            y1 = max(current_box[1], box[1])
            x2 = min(current_box[2], box[2])
            y2 = min(current_box[3], box[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            iou = intersection / (current_area + area - intersection)
            if iou >= 0.15:
                sorted_index.remove(index)
    return selected_boxes_index


def nms(bounding_boxes, confidence_score, threshold):
    if len(bounding_boxes) == 0:
        return [], []
    bounding_boxes = [tensor.detach().numpy() for tensor in bounding_boxes]
    confidence_score = [tensor.detach().numpy() for tensor in confidence_score]
    boxes = np.array(bounding_boxes)
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    score = np.array(confidence_score)
    picked_boxes = []
    picked_score = []
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    order = np.argsort(score)
    while order.size > 0:
        index = order[-1]
        if len(picked_boxes) < 8 and confidence_score[index] > 0.4:
            picked_boxes.append(bounding_boxes[index])
            picked_score.append(confidence_score[index])
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ratio < threshold)
        order = order[left]
    return picked_boxes, picked_score


for filename in os.listdir(test_dict_path):
    if filename.endswith('.jpg'):
        file_path = os.path.join(test_dict_path, filename)
        image = cv2.imread(file_path, 1)
        # image = image_pre_treat(image)
        image = torch.tensor(image)
        image = image / 255.0
        image = [image.permute(2, 0, 1)]
        output = model(image)
        detection = output[0]
        boxes = detection['boxes']
        labels = detection['labels']
        scores = detection['scores']
        selected_boxes, selected_scores = nms(boxes, scores, iou_threshold)
        print(selected_boxes)
        save_path = 'test/' + filename[0:4] + '.txt'
        with open(save_path, 'w') as f:
            for box, score in zip(selected_boxes, selected_scores):
                f.write('tower')
                f.write(' ')
                f.write(str(score.item()))
                f.write(' ')
                for m in box:
                    f.write(str(int(m.item())))
                    f.write(' ')
                f.write('\n')
            f.close()
