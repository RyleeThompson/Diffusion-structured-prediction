import torch as th
from utils.matcher import DiffusionHungarianMatcher


class RMSEEvaluation:
    def __init__(self, cfg):
        self.matcher = DiffusionHungarianMatcher()

    @th.no_grad()
    def evaluate(self, bb_preds, class_preds, x_start, x_start_classes):
        matched_preds, matched_x_start = self.matcher.match_preds(bb_preds, class_preds, x_start, x_start_classes)
        return self.compute_rmse(matched_preds, matched_x_start)

    def compute_rmse(self, x1, x2):
        try:
            x1 = th.cat(x1)
            x2 = th.cat(x2)
        except:
            keys = ['rmse', 'rmse_x', 'rmse_y', 'rmse_widths', 'rmse_heights']
            return {key: 1e3 for key in keys}

        mse_fn = th.nn.MSELoss()
        rmse_fn = lambda x, y: th.sqrt(mse_fn(x, y))

        rmse = {}
        rmse['rmse'] = rmse_fn(x1, x2).item()
        rmse['rmse_x'] = rmse_fn(x1[:, 0], x2[:, 0]).item()
        rmse['rmse_y'] = rmse_fn(x1[:, 1], x2[:, 1]).item()
        rmse['rmse_widths'] = rmse_fn(x1[:, 2], x2[:, 2]).item()
        rmse['rmse_heights'] = rmse_fn(x1[:, 3], x2[:, 3]).item()
        return rmse
