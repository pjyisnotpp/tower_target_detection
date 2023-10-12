import cv2
import numpy as np
import torch
img = cv2.imread('data/JPEGImages/val/0010.jpg', 1)


def image_pre_treat(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=300)
    cv2.imwrite('demo1.jpg', edges)
    height, width = edges.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    color_image[:, :, 0] = edges  # 复制灰度通道到三个通道
    color_image[:, :, 1] = edges
    color_image[:, :, 2] = edges
    # alpha = 0.5  # 透明度因子，可根据需要调整
    # result_image = cv2.addWeighted(image, 1, color_image, alpha, 0)
    cv2.imwrite('demo.jpg', color_image)


def draw_boxes_on_image(image, boxes):
    output_image = image.copy()
    for box in boxes:
        box = box.numpy()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite('demo_rec.jpg', output_image)
# image_pre_treat(img)

# image_pre_treat(img)
boxes = []
draw_boxes_on_image(img, boxes)
