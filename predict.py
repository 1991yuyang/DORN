from network import Dorn
import os
import cv2
import torch as t
import numpy as np
from torch.nn import functional as F
from utils import *
from metric import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model_pth = r"/home/yuyang/python_project/DORN/model/epoch.pth"
image_pth = r"/home/yuyang/data/make3d/train/image/568.png"
resnet_type = "resnet34"
scale_max_factor = 1.5
alpha = 0
beta = 255
K = 120
us_use_interpolate = True
show_scale = 1
input_image_size = [512, 512]  # [w, h]
metric_depth_range = [0, 255]  # 制定测量哪个深度范围内的指标值
##########################################
alpha = alpha // scale_max_factor
epsilon = 1 - alpha  # shift value
alpha_star = 1
beta_star = beta + epsilon


def load_one_image(image_pth, alpha_star, beta_star, K):
    img = cv2.imread(image_pth)
    cv2.imshow("RGB", img)
    orig_image_size = img.shape[:2]  # [h, w]
    img = cv2.resize(img, input_image_size)
    img = t.from_numpy(np.transpose(img, axes=[2, 0, 1]) / 255).type(t.FloatTensor).unsqueeze(0)  # [1, 3, h, w]
    img = img.cuda(0)
    ti = SID(alpha_star, beta_star, K).expand((int(input_image_size[1]), int(input_image_size[0]), -1))  # [1, 1, K + 1]
    ti = ti.expand((int(input_image_size[1]), int(input_image_size[0]), -1))  # [image_size[1], image_size[0], K + 1]
    return img, orig_image_size, ti


def load_model(model_pth, resnet_type, K, us_use_interpolate):
    model = Dorn(resnet_type, K, us_use_interpolate)
    model.load_state_dict(t.load(model_pth))
    model = model.cuda(0)
    model.eval()
    return model


def inference(model, img, orig_image_size, K, ti, epsilon):
    with t.no_grad():
        prob = model(img)[0].cpu().detach()  # [K, input_image_size[1], input_image_size[0]]
        pos_prob = prob.permute(dims=[1, 2, 0])  # [input_image_size[1], input_image_size[0], K]
        pos_mask = t.sum(pos_prob > 0.5, dim=2)
        pos_mask[pos_mask == 0] = 1
        low_interval_label = F.one_hot(pos_mask - 1, K + 1).type(t.BoolTensor)  # [input_image_size[1], input_image_size[0], K]
        hight_intever_label = F.one_hot(pos_mask, K + 1).type(t.BoolTensor)  # [input_image_size[1], input_image_size[0], K]
        low_depth = ti[low_interval_label].view((input_image_size[1], input_image_size[0]))
        hight_depth = ti[hight_intever_label].view((input_image_size[1], input_image_size[0]))
        depth = (low_depth + hight_depth) / 2 - epsilon
        depth = depth.numpy()
    depth = cv2.resize(depth, orig_image_size[::-1])
    return depth


def show(depth, alpha, beta, show_scale, is_gt):
    depth = cv2.resize(depth, (int(depth.shape[1] * show_scale), int(depth.shape[0] * show_scale)))
    if is_gt:
        depth_window_name = "depth_gt"
        depth_color_name = "depth_colormap_gt"
    else:
        depth_window_name = "depth"
        depth_color_name = "depth_colormap"
    cv2.imshow(depth_window_name, depth.astype(np.uint8))
    depth_show = 255 * (depth - alpha) / (beta - alpha)
    depth_colormap = cv2.applyColorMap(depth_show.astype(np.uint8), 4)
    cv2.imshow(depth_color_name, depth_colormap)
    cv2.waitKey()


def get_metric_value(depth_pred, depth_gt, depth_range):
    depth_pred = depth_pred.copy()
    depth_gt = depth_gt.copy()
    mask = np.logical_and(depth_gt > depth_range[0], depth_gt <= depth_range[1])
    depth_pred = depth_pred[mask]
    depth_gt = depth_gt[mask]
    abs_rel_value = abs_rel(depth_pred, depth_gt)
    sq_rel_value = sq_rel(depth_pred, depth_gt)
    mae_value = mae(depth_pred, depth_gt)
    rmse_value = rmse(depth_pred, depth_gt)
    log_rmse_value = log_rmse(depth_pred, depth_gt)
    print("abs_rel:", abs_rel_value)
    print("sq_rel_value:", sq_rel_value)
    print("mae_value:", mae_value)
    print("rmse_value:", rmse_value)
    print("log_rmse_value:", log_rmse_value)
    return abs_rel_value, sq_rel_value, mae_value, rmse_value, log_rmse_value


if __name__ == "__main__":
    img, orig_image_size, ti = load_one_image(image_pth, alpha_star, beta_star, K)
    model = load_model(model_pth, resnet_type, K, us_use_interpolate)
    depth = inference(model, img, orig_image_size, K, ti, epsilon)
    show(depth, alpha, beta, show_scale, False)
    gt_depth = np.load(r"/home/yuyang/data/make3d/train/depth/568.npy")
    show(gt_depth, alpha, beta, 1, True)
    abs_rel_value, sq_rel_value, mae_value, rmse_value, log_rmse_value = get_metric_value(depth, gt_depth, metric_depth_range)
    print(depth.astype(int))
    print(gt_depth)