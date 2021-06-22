import numpy as np
import cv2

def resize_image(img: np.ndarray, out_img: np.ndarray) -> (float, int, int):
    assert img.dtype == out_img.dtype, "Input images must have same dtype"
    left_pad = 0
    top_pad = 0
    h, w, _ = out_img.shape
    if img.shape[0] / img.shape[1] > h / w:
        resize_scale = h / img.shape[0]
        tmp_img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
        left_pad = int((w - tmp_img.shape[1]) / 2)
        out_img[:, left_pad:left_pad + tmp_img.shape[1], :] = tmp_img
    else:
        resize_scale = w / img.shape[1]
        tmp_img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
        top_pad = int((h - tmp_img.shape[0]) / 2)
        out_img[top_pad:top_pad + tmp_img.shape[0], :, :] = tmp_img
    return resize_scale, top_pad, left_pad

def image_standardise(rgb_img: np.array): # input shape : NHWC
    # ((inp / 255) - mean) / std
    rgb_img = rgb_img.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406]) # RGB
    std = np.array([0.229, 0.224, 0.225])
    rgb_img = np.divide(rgb_img, 255)
    rgb_img = np.divide(np.subtract(rgb_img, mean, dtype=np.float32), std,dtype=np.float32)  # (inp-mean)/std
    rgb_img = rgb_img.transpose((0, 3, 1, 2)) # NHWC to NCHW

    return rgb_img # NCHW


def NMS(boxes, overlap_threshold):
    '''

    :param boxes: numpy nx5, n is the number of boxes, 0:4->x1, y1, x2, y2, 4->score
    :param overlap_threshold:
    :return:
    '''
    if boxes.shape[0] == 0:
        return boxes

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype != np.float32:
        boxes = boxes.astype(np.float32)

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    sc = boxes[:, 4]
    widths = x2 - x1
    heights = y2 - y1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = heights * widths
    idxs = np.argsort(sc)  # 从小到大排序

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compare secend highest score boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bo（ box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]

def cropping(image, bboxes: tuple): 
    # return one cropped image
    bbox = [int(coor) for coor in bboxes]
    crop = image[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    return crop