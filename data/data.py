import numpy as np
import random, math
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import imgaug.augmenters as iaa


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def normalize_image(image):
    image = image / 255.0

    return image


def letterbox_resize(image, target_size, return_padding_info=False):
    src_w, src_h = image.size
    target_w, target_h = target_size

    scale = min(target_w / src_w, target_h / src_h)
    padding_w = int(src_w * scale)
    padding_h = int(src_h * scale)
    padding_size = (padding_w, padding_h)

    dx = (target_w - padding_w) // 2
    dy = (target_h - padding_h) // 2
    offset = (dx, dy)

    image = image.resize(padding_size, Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128, 128, 128))
    new_image.paste(image, offset)

    if return_padding_info:
        return new_image, padding_size, offset
    else:
        return new_image


def random_resize_crop_pad(image, target_size, aspect_ratio_jitter=0.3, scale_jitter=0.5):
    target_w, target_h = target_size

    rand_aspect_ratio = target_w / target_h * rand(1 - aspect_ratio_jitter, 1 + aspect_ratio_jitter) / rand(
        1 - aspect_ratio_jitter, 1 + aspect_ratio_jitter)
    rand_scale = rand(scale_jitter, 1 / scale_jitter)

    if rand_aspect_ratio < 1:
        padding_h = int(rand_scale * target_h)
        padding_w = int(padding_h * rand_aspect_ratio)
    else:
        padding_w = int(rand_scale * target_w)
        padding_h = int(padding_w / rand_aspect_ratio)
    padding_size = (padding_w, padding_h)
    image = image.resize(padding_size, Image.BICUBIC)

    dx = int(rand(0, target_w - padding_w))
    dy = int(rand(0, target_h - padding_h))
    padding_offset = (dx, dy)

    new_image = Image.new('RGB', (target_w, target_h), (128, 128, 128))
    new_image.paste(image, padding_offset)

    return new_image, padding_size, padding_offset


def reshape_boxes(boxes, src_shape, target_shape, padding_shape, offset, horizontal_flip=False, vertical_flip=False):
    if len(boxes) > 0:
        src_w, src_h = src_shape
        target_w, target_h = target_shape
        padding_w, padding_h = padding_shape
        dx, dy = offset

        np.random.shuffle(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * padding_w / src_w + dx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * padding_h / src_h + dy

        if horizontal_flip:
            boxes[:, [0, 2]] = target_w - boxes[:, [2, 0]]

        if vertical_flip:
            boxes[:, [1, 3]] = target_h - boxes[:, [3, 1]]

        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > target_w] = target_w
        boxes[:, 3][boxes[:, 3] > target_h] = target_h

        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(boxes_w > 1, boxes_h > 1)]

    return boxes


def random_hsv_distort(image, hue=.1, sat=1.5, val=1.5):
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)

    x = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

    x = x.astype(np.float64)
    x[..., 0] = (x[..., 0] * (1 + hue)) % 180
    x[..., 1] = x[..., 1] * sat
    x[..., 2] = x[..., 2] * val
    x[..., 1:3][x[..., 1:3] > 255] = 255
    x[..., 1:3][x[..., 1:3] < 0] = 0
    x = x.astype(np.uint8)

    x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    new_image = Image.fromarray(x)

    return new_image


def random_brightness(image, jitter=.5):
    enh_bri = ImageEnhance.Brightness(image)
    brightness = rand(jitter, 1 / jitter)
    new_image = enh_bri.enhance(brightness)

    return new_image


def random_chroma(image, jitter=.5):
    enh_col = ImageEnhance.Color(image)
    color = rand(jitter, 1 / jitter)
    new_image = enh_col.enhance(color)

    return new_image


def random_contrast(image, jitter=.5):
    enh_con = ImageEnhance.Contrast(image)
    contrast = rand(jitter, 1 / jitter)
    new_image = enh_con.enhance(contrast)

    return new_image


def random_sharpness(image, jitter=.5):
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = rand(jitter, 1 / jitter)
    new_image = enh_sha.enhance(sharpness)

    return new_image


def random_horizontal_flip(image, prob=.5):
    flip = rand() < prob
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    return image, flip


def random_vertical_flip(image, prob=.2):
    flip = rand() < prob
    if flip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    return image, flip


def random_grayscale(image, prob=.2):
    convert = rand() < prob
    if convert:
        # convert to grayscale first, and then
        # back to 3 channels fake RGB
        image = image.convert('L')
        image = image.convert('RGB')

    return image


def random_blur(image, prob=.1):
    blur = rand() < prob
    if blur:
        image = image.filter(ImageFilter.BLUR)

    return image


def random_motion_blur(image, prob=.1):
    motion_blur = rand() < prob
    if motion_blur:
        img = np.array(image)
        # random blur severity from 1 to 5
        severity = np.random.randint(1, 6)

        seq = iaa.Sequential([iaa.imgcorruptlike.MotionBlur(severity=severity)])
        # seq = iaa.Sequential([iaa.MotionBlur(k=30)])

        img = seq(images=np.expand_dims(img, 0))
        image = Image.fromarray(img[0])

    return image


def get_ground_truth_data(annotation_line, input_shape, augment=True, max_boxes=100):
    line = annotation_line.split()
    image = Image.open(line[0])
    image_size = image.size
    model_input_size = tuple(reversed(input_shape))
    boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not augment:
        new_image, padding_size, offset = letterbox_resize(image, target_size=model_input_size,
                                                           return_padding_info=True)
        image_data = np.array(new_image)
        image_data = normalize_image(image_data)

        boxes = reshape_boxes(boxes, src_shape=image_size, target_shape=model_input_size, padding_shape=padding_size,
                              offset=offset)
        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]

        box_data = np.zeros((max_boxes, 5))
        if len(boxes) > 0:
            box_data[:len(boxes)] = boxes

        return image_data, box_data

    image, padding_size, padding_offset = random_resize_crop_pad(image, target_size=model_input_size)

    image, horizontal_flip = random_horizontal_flip(image)

    image = random_brightness(image)

    image = random_chroma(image)

    image = random_contrast(image)

    image = random_sharpness(image)

    image = random_grayscale(image)

    # image = random_blur(image)

    # image = random_motion_blur(image, prob=0.2)

    # random vertical flip image
    image, vertical_flip = random_vertical_flip(image)

    boxes = reshape_boxes(boxes, src_shape=image_size, target_shape=model_input_size, padding_shape=padding_size,
                          offset=padding_offset, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
    if len(boxes) > max_boxes:
        boxes = boxes[:max_boxes]

    # prepare image & box data
    image_data = np.array(image)
    image_data = normalize_image(image_data)
    box_data = np.zeros((max_boxes, 5))
    if len(boxes) > 0:
        box_data[:len(boxes)] = boxes

    return image_data, box_data


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes, multi_anchor_assign, iou_thresh=0.2):
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    batch_size = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(batch_size):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue

        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Sort anchors according to IoU score
        # to find out best assignment
        best_anchors = np.argsort(iou, axis=-1)[..., ::-1]

        if not multi_anchor_assign:
            best_anchors = best_anchors[..., 0]
            # keep index dim for the loop in following
            best_anchors = np.expand_dims(best_anchors, -1)

        for t, row in enumerate(best_anchors):
            for l in range(num_layers):
                for n in row:
                    # use different matching policy for single & multi anchor assign
                    if multi_anchor_assign:
                        matching_rule = (iou[t, n] > iou_thresh and n in anchor_mask[l])
                    else:
                        matching_rule = (n in anchor_mask[l])

                    if matching_rule:
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment, rescale_interval,
                   multi_anchor_assign):
    n = len(annotation_lines)
    i = 0
    while True:

        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_ground_truth_data(annotation_lines[i], input_shape, augment=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)

        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes, multi_anchor_assign)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment=None,
                           rescale_interval=-1, multi_anchor_assign=False, **kwargs):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment,
                          rescale_interval, multi_anchor_assign)
