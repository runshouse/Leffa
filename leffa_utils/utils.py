import os
import cv2
import torch
import numpy as np
from numpy.linalg import lstsq
from PIL import Image, ImageDraw


def resize_and_center(image, target_width, target_height):
    img = np.array(image)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    original_height, original_width = img.shape[:2]

    scale = min(target_height / original_height, target_width / original_width)
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)

    resized_img = cv2.resize(img, (new_width, new_height),
                            interpolation=cv2.INTER_CUBIC)

    padded_img = np.ones((target_height, target_width, 3),
                        dtype=np.uint8) * 255

    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2

    padded_img[top:top + new_height, left:left + new_width] = resized_img

    padding_info = {
        'top': top,
        'bottom': top + new_height,
        'left': left,
        'right': left + new_width
    }

    return Image.fromarray(padded_img), padding_info


def list_dir(folder_path):
    # Collect all file paths within the directory
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    file_paths = sorted(file_paths)
    return file_paths


label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}


def get_agnostic_mask(model_parse, keypoint, category, size=(384, 512)):
    parse_array = np.array(model_parse)
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    parse_shape = (parse_array > 0).astype(np.float32)

    parse_head = (parse_array == 1).astype(np.float32) + \
        (parse_array == 2).astype(np.float32) + \
        (parse_array == 3).astype(np.float32) + \
        (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                        (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["scarf"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (
        parse_array == label_map["background"]).astype(np.float32)

    arms = (parse_array == 14).astype(np.float32) + \
        (parse_array == 15).astype(np.float32)

    if category == 'dresses':
        label_cat = 7
        parse_mask = (parse_array == 7).astype(np.float32) + \
            (parse_array == 12).astype(np.float32) + \
            (parse_array == 13).astype(np.float32)
        parser_mask_changeable += np.logical_and(
            parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        label_cat = 4
        parse_mask = (parse_array == 4).astype(np.float32)

        parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
            (parse_array == label_map["pants"]).astype(np.float32)

        parser_mask_changeable += np.logical_and(
            parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':
        label_cat = 6
        parse_mask = (parse_array == 6).astype(np.float32) + \
            (parse_array == 12).astype(np.float32) + \
            (parse_array == 13).astype(np.float32)

        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
            (parse_array == 14).astype(np.float32) + \
            (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(
            parse_array, np.logical_not(parser_mask_fixed))

    parse_head = torch.from_numpy(parse_head)  # [0,1]
    parse_mask = torch.from_numpy(parse_mask)  # [0,1]
    parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
    parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

    # dilation
    parse_without_cloth = np.logical_and(
        parse_shape, np.logical_not(parse_mask))
    parse_mask = parse_mask.cpu().numpy()

    width = size[0]
    height = size[1]

    im_arms = Image.new('L', (width, height))
    arms_draw = ImageDraw.Draw(im_arms)
    if category == 'dresses' or category == 'upper_body':
        shoulder_right = tuple(np.multiply(pose_data[2, :2], height / 512.0))
        shoulder_left = tuple(np.multiply(pose_data[5, :2], height / 512.0))
        elbow_right = tuple(np.multiply(pose_data[3, :2], height / 512.0))
        elbow_left = tuple(np.multiply(pose_data[6, :2], height / 512.0))
        wrist_right = tuple(np.multiply(pose_data[4, :2], height / 512.0))
        wrist_left = tuple(np.multiply(pose_data[7, :2], height / 512.0))
        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                arms_draw.line(
                    [wrist_left, elbow_left, shoulder_left, shoulder_right], 'white', 30, 'curve')
            else:
                arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right], 'white', 30,
                               'curve')
        elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                arms_draw.line([shoulder_left, shoulder_right,
                               elbow_right, wrist_right], 'white', 30, 'curve')
            else:
                arms_draw.line([elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right], 'white', 30,
                               'curve')
        else:
            arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right], 'white',
                           30, 'curve')

        if height > 512:
            im_arms = cv2.dilate(np.float32(im_arms), np.ones(
                (10, 10), np.uint16), iterations=5)
        elif height > 256:
            im_arms = cv2.dilate(np.float32(im_arms), np.ones(
                (5, 5), np.uint16), iterations=5)
        hands = np.logical_and(np.logical_not(im_arms), arms)
        parse_mask += im_arms
        parser_mask_fixed += hands

    # delete neck
    parse_head_2 = torch.clone(parse_head)
    if category == 'dresses' or category == 'upper_body':
        points = []
        points.append(np.multiply(pose_data[2, :2], height / 512.0))
        points.append(np.multiply(pose_data[5, :2], height / 512.0))
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords, rcond=None)[0]
        for i in range(parse_array.shape[1]):
            y = i * m + c
            parse_head_2[int(y - 20 * (height / 512.0)):, i] = 0

    parser_mask_fixed = np.logical_or(
        parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
    parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                           np.logical_not(np.array(parse_head_2, dtype=np.uint16))))

    if height > 512:
        parse_mask = cv2.dilate(parse_mask, np.ones(
            (20, 20), np.uint16), iterations=5)
    elif height > 256:
        parse_mask = cv2.dilate(parse_mask, np.ones(
            (10, 10), np.uint16), iterations=5)
    else:
        parse_mask = cv2.dilate(parse_mask, np.ones(
            (5, 5), np.uint16), iterations=5)
    parse_mask = np.logical_and(
        parser_mask_changeable, np.logical_not(parse_mask))
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    return mask


def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width=1,
                 mode='constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst


def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask
