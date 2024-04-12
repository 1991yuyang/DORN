import numpy as np


def abs_rel(depth_pred, depth_gt):
    abs_rel = np.mean(np.abs(depth_pred - depth_gt) / depth_gt)
    return abs_rel


def mae(depth_pred, depth_gt):
    mae = np.mean(np.abs(depth_pred - depth_gt))
    return mae


def rmse(depth_pred, depth_gt):
    rmse = np.sqrt(np.mean(np.power(depth_pred - depth_gt, 2)))
    return rmse


def log_rmse(depth_pred, depth_gt):
    log_rmse = np.sqrt(np.mean(np.power(np.log10(depth_pred) - np.log10(depth_gt), 2)))
    return log_rmse


def sq_rel(depth_pred, depth_gt):
    sq_rel = np.mean(np.power((depth_pred - depth_gt), 2) / depth_gt)
    return sq_rel