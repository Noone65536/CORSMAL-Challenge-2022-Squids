import copy
import torch
import numpy as np

def computeScoreType1(gt, _est):
  gt = gt.cpu().data.numpy()
  _est = _est.cpu().data.numpy()
  est = copy.deepcopy(_est)
  assert (len(gt) == len(est))
  indicator_f = est > -1
  ec = np.exp(-(np.abs(gt - est) / gt)) * indicator_f
  score = np.sum(ec) / len(gt)
  return score

def myLoss(est, gt):
  ec = 1 - torch.exp(-(torch.abs(gt - est) / gt))
  score = torch.sum(ec) / len(gt)

  return score

