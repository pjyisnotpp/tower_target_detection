import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from tower_dataset import Tower_Dataset
import numpy as np


def calculate_iou(box1, box2):
    # 计算两个框的IoU
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x_intersection = max(0, min(x2, x4) - max(x1, x3))
    y_intersection = max(0, min(y2, y4) - max(y1, y3))
    intersection = x_intersection * y_intersection
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)
    union = area_box1 + area_box2 - intersection
    return intersection / union


def calculate_ap(precision, recall):
    # 计算AP（Average Precision）
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_precision_recall(predictions, ground_truth, confidence_threshold=0.5):
    # 计算精确度和召回率
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    num_ground_truth = len(ground_truth)
    precision = []
    recall = []
    for prediction in predictions:
        pred_box = prediction[:4]
        confidence = prediction[0]
        if confidence < confidence_threshold:
            false_positives += 1
        else:
            best_iou = 0
            best_gt_index = -1
            for i, gt_box in enumerate(ground_truth):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = i
            if best_iou >= 0.4:
                true_positives += 1
                ground_truth.pop(best_gt_index)
            else:
                false_positives += 1
        # 计算精确度和召回率
        if true_positives + false_positives == 0:
            precision.append(0)
        else:
            precision.append(true_positives / (true_positives + false_positives))
        recall.append(true_positives / num_ground_truth)
    return precision, recall


def is_box_A_contains_box_B(box_A, box_B):
    x1_A, y1_A, x2_A, y2_A = box_A
    x1_B, y1_B, x2_B, y2_B = box_B

    if x1_A <= x1_B and y1_A <= y1_B and x2_A >= x2_B and y2_A >= y2_B:
        return True
    else:
        return False


def is_almost_containment(box1, box2, tolerance=0.1):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算box1在x轴上的宽度
    width_1 = x2_1 - x1_1
    # 计算box2在x轴上的宽度
    width_2 = x2_2 - x1_2

    # 计算box1在y轴上的高度
    height_1 = y2_1 - y1_1
    # 计算box2在y轴上的高度
    height_2 = y2_2 - y1_2

    # 判断box1是否在x轴上包含box2，并且在y轴上的高度比例小于等于tolerance
    x_containment = x1_2 >= x1_1 and x2_2 <= x2_1
    y_ratio = height_2 / height_1 <= tolerance

    # 判断box1是否在y轴上包含box2，并且在x轴上的宽度比例小于等于tolerance
    y_containment = y1_2 >= y1_1 and y2_2 <= y2_1
    x_ratio = width_2 / width_1 <= tolerance

    # 如果在一个维度上包含且另一个维度上稍大，则返回True
    if (x_containment and y_ratio) or (y_containment and x_ratio):
        return True
    else:
        return False


# def non_maximum_suppression(box_list, score_list, iou_threshold):
#     selected_boxes_index = []
#     sorted_index = sorted(range(len(score_list)), key=lambda i: score_list[i], reverse=True)
#     while len(sorted_index) > 0:
#         current_index = sorted_index[0]
#         if score_list[current_index] > 0.4 and len(selected_boxes_index) < 6:
#             selected_boxes_index.append(current_index)
#         current_box = box_list[current_index]
#         current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
#         del sorted_index[0]
#         for index in sorted_index[:]:
#             box = box_list[index]
#             if box[2]-box[0] > box[3]-box[1]:
#                 sorted_index.remove(index)
#                 continue
#             if is_box_A_contains_box_B(box, current_box) or is_box_A_contains_box_B(current_box, box):
#                     # or is_almost_containment(box, current_box):
#             # if is_almost_containment(box, current_box):
#                 sorted_index.remove(index)
#                 continue
#             area = (box[2] - box[0]) * (box[3] - box[1])
#             x1 = max(current_box[0], box[0])
#             y1 = max(current_box[1], box[1])
#             x2 = min(current_box[2], box[2])
#             y2 = min(current_box[3], box[3])
#             intersection = max(0, x2 - x1) * max(0, y2 - y1)
#             iou = intersection / (current_area + area - intersection)
#             if iou >= 0.5:
#                 sorted_index.remove(index)
#     return selected_boxes_index

def nms(bounding_boxes, confidence_score, threshold):
    if len(bounding_boxes) == 0:
        return [], []
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
        if confidence_score[index] > 0.5 and len(picked_score) < 8:
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


model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('train_model2.pth'))
iou_threshold = 0.5
total_objects = 0
correct_detections = 0
model.eval()
root_dir = 'data'
validation_dataset = Tower_Dataset(root_dir, 'val')
label_list = ['tower']
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
all_ap = 0
precisions = []
recalls = []
no_target = 0
for i, data in enumerate(validation_loader):
    model.eval()
    with torch.no_grad():
        x = data['image']
        y = data['labels']
        for j in y:
            j['bbox'] = torch.cat(j['bbox'], dim=0)
        images = list(image / 255.0 for image in x)
        images = [image.permute(2, 0, 1) for image in images]
        outputs = model(images)
        detections = outputs[0]
        true_boxes = [y[p]['bbox'] for p in range(len(y))]
        true_labels = []
        for one_item in y:
            real_label = one_item['label'][0]
            true_labels.append(real_label)
        true_labels = [label_list.index(i) + 1 for i in true_labels]
        boxes = detections['boxes']
        # print(boxes)
        labels = detections['labels']
        scores = detections['scores']
        # print(scores)
        pred_boxes, pred_scores = nms(boxes, scores, iou_threshold)
        print(pred_boxes)
        precision, recall = calculate_precision_recall(pred_boxes, true_boxes)
        precisions.append(precision)
        recalls.append(recall)
        ap = calculate_ap(precision, recall)
        all_ap += ap
        if ap < 0.001:
            no_target += 1
        print(ap)
print(no_target)
print(float(all_ap / (len(validation_dataset) + 0.0)))
# average_precision = np.trapz(precisions, recalls)
# print(average_precision)
