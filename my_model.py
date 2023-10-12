import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from tower_dataset import Tower_Dataset


def collate_fn(data):
    image_list = []
    label_list = []
    for unit in data:
        image_list.append(unit['image'])
        label_list.append(unit['labels'])
    new_data = {}
    new_data['image'] = image_list
    new_data['labels'] = label_list
    return new_data


root_dir = 'data'
train_dataset = Tower_Dataset(root_dir, 'train')

label_list = ['tower']
# print(train_dataset[0])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
model = fasterrcnn_resnet50_fpn(weights=None)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
for name, param in model.named_parameters():
    if "box_predictor" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-5, momentum=0.9, weight_decay=5e-6)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# model.to('cuda:0')
epochs = 1

for epoch in range(epochs):
    model.train()
    # breakpoint()
    for i, data in enumerate(train_loader):
        # breakpoint()
        x = data['image']
        y = data['labels']
        images = list(image / 255.0 for image in x)
        images = [image.permute(2, 0, 1) for image in images]
        original_targets = []
        for target_dict in y:
            original_targets.append([{'boxes': box['bbox'], 'labels': box['label']} for box in target_dict])
        # print(original_targets)
        for target_set in original_targets:
            for target in target_set:
                target['boxes'] = torch.tensor(target['boxes'])
                target['labels'] = torch.tensor(label_list.index(target['labels']))
        # print(original_targets)
        for k in range(len(images)):
            target = {'boxes': [], 'labels': []}
            for item in original_targets[k]:
                target['boxes'].append(item['boxes'])
                target['labels'].append(item['labels'])
            target['boxes'] = torch.stack(target['boxes'])
            target['labels'] = torch.stack(target['labels'])
            target = [target]
            image = [images[k]]
            # fixed_boxes = [box.tolist() for box in target['boxes']]
            # fixed_labels = target['labels'].tolist()
            # fixed_target = {'boxes': fixed_boxes, 'labels': fixed_labels}
            print(target)
            # print(images[k])
            for item in target:
                new_item = item['boxes']
                # print(type(new_item))
                item['labels'] = item['labels']
            for item in image:
                item = item
            loss_dict = model(image, target)
            # print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            print(losses)
            optimizer.zero_grad()
            losses.backward()
        optimizer.step()
    lr_scheduler.step()
torch.save(model.state_dict(), 'model.pth')
