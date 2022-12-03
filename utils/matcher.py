import torch as th
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops.boxes import box_area
from utils.data.BBNormalizer import create_bbox_transformer


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = th.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = th.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# modified from thvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = th.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = th.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def box_xywh_to_x0y0x1y1(boxes):
    assert (boxes[:, ..., 2] > 0).all(), (boxes[:, ..., 2] > 0).type(th.float32).mean()
    assert (boxes[:, ..., 3] > 0).all(), (boxes[:, ..., 3] > 0).type(th.float32).mean()
    boxes[:, ..., 2] = boxes[:, ..., 2] + boxes[:, ..., 0]  # x + w
    boxes[:, ..., 3] = boxes[:, ..., 3] + boxes[:, ..., 1]  # y + h
    return boxes


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 2, cost_giou: float = 5):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @th.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = th.cat([v["labels"] for v in targets])
        tgt_bbox = th.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = th.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        # import ipdb; ipdb.set_trace()
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # except:
            # import ipdb; ipdb.set_trace()
            # pass
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(th.as_tensor(i, dtype=th.int64), th.as_tensor(j, dtype=th.int64)) for i, j in indices]


class DiffusionHungarianMatcher(HungarianMatcher):
    # def __init__(self, cfg, cost_class=0, cost_bbox=5, cost_giou=2):
    #     super().__init__(
    #         cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)

    @th.no_grad()
    def forward(self, preds, targets):
        preds = preds.clone().to(targets.device)
        preds['bbox'] = preds['bbox'].clamp(-1, 1)
        preds = preds.inverse_normalize().to_xyxy()
        #.unmask(use_cls=preds.unmask_with_cls)
        # pred_bbs = preds_unmask['bbox']
        # pred_cls = preds_unmask['classes_softmax']
        targets = targets.clone()
        targets['bbox'] = targets['bbox'].clamp(-1, 1)
        targets = targets.inverse_normalize().to_xyxy()#.unmask(use_cls=True)
        # target_bbs = targets_unmask['bbox']
        # target_cls = targets_unmask['classes']
        target_lst = []
        for ix in range(len(targets['bbox'])):
            dct = {'labels': targets['classes'][ix], 'boxes': targets['bbox'][ix]}
            target_lst.append(dct)
        pred_dct = {'pred_logits': preds['classes_softmax'].to(targets['bbox'].device), 'pred_boxes': preds['bbox'].to(targets['bbox'].device)}
        # import ipdb; ipdb.set_trace()
        ixs = super().forward(pred_dct, target_lst)

        pred_ixs = th.stack([ix[0] for ix in ixs]).to(targets['bbox'].device)
        tgt_ixs = th.stack([ix[1] for ix in ixs]).to(targets['bbox'].device)
        return pred_ixs, tgt_ixs

    def match_single(self, pred_bbs, pred_cls, target_bbs, target_cls, ix):
        target_cls = target_cls[ix]
        target_bbs = target_bbs[ix]
        pred_bbs = pred_bbs[ix]
        pred_cls = pred_cls[ix]
        targets = [{'labels': target_cls,
                   'boxes': target_bbs}]
        preds = {'pred_logits': pred_cls.unsqueeze(0),
                 'pred_boxes': pred_bbs.unsqueeze(0)}
        pred_ixs, tgt_ixs = super().forward(preds, targets)[0]
        return pred_ixs, tgt_ixs

    def match_preds(self, bb_preds, class_preds, x_start, classes):
        matched_gt = []
        matched_preds = []
        for pred_bb, pred_cls, gt_bb, gt_cls in zip(bb_preds, class_preds, x_start, classes):
            pred_bb_ = box_xywh_to_x0y0x1y1(pred_bb.clone())
            gt_bb_ = box_xywh_to_x0y0x1y1(gt_bb.clone())

            targets = {'labels': gt_cls.nonzero()[:, -1],
                       'boxes': gt_bb_}
            cls_shape = [pred_bb_.shape[0], pred_cls.shape[1]]
            outputs = {'pred_boxes': pred_bb_.unsqueeze(0),
                       'pred_logits': th.randn(cls_shape).unsqueeze(0).to(pred_bb_.device)}
            if pred_bb_.shape[0] > 0 and gt_bb_.shape[0] > 0:
                # try:
                matched_ixs = super().forward(outputs, [targets])[0]
                # except:
                    # import ipdb; ipdb.set_trace()
                    # pass
                matched_gt.append(gt_bb[matched_ixs[1]])
                matched_preds.append(pred_bb[matched_ixs[0]])
        return matched_preds, matched_gt

    def get_targets(self, batch):
        if self.match_x_start:
            gt_bboxes = batch['x_start'].clone().view(batch['x_t'].shape)
            gt_classes = batch['classes']
        else:
            raise RuntimeError('Not yet implemented')

        gt_bboxes = self.convert_bboxes(gt_bboxes)

        targets = []
        for bbs, classes, num_bbs in zip(gt_bboxes, gt_classes, batch['num_bbs']):
            target_dct = {'labels': classes[:num_bbs].nonzero()[:, -1],
                          'boxes': bbs[:num_bbs]}
            targets.append(target_dct)

        return targets

    def get_outputs(self, model_out, batch):
        if self.match_x_start:
            pred_bboxes = model_out['pred_xstart'].detach().clone().clamp(-1, 1)
        else:
            raise RuntimeError('Not yet implemented')

        pred_bboxes = self.convert_bboxes(pred_bboxes)
        pred_classes = batch['classes'].clone()

        pred_bboxes[batch['remove_masks']] = 1e12
        outputs = {'pred_boxes': pred_bboxes,
                   'pred_logits': pred_classes}
        return outputs

    def convert_bboxes(self, bboxes):
        raise Exception('should use bbox norm features')
        bboxes = self.bbox_transformer.inverse_bbox_normalization(bboxes)
        bboxes = box_xywh_to_x0y0x1y1(bboxes)
        return bboxes

    def reorder_batch(self, matched_ixs, batch):
        for sample_ix, (pred_ixs, tgt_ixs) in enumerate(matched_ixs):
            num_bbs = batch['num_bbs'][sample_ix]
            assert pred_ixs.max() <= num_bbs, '{} {}'.format(pred_ixs.max(), num_bbs)

            keys_to_match = ['x_t', 'classes', 'x_start', 'noise', 'x_start']
            keys_to_reshape = ['x_start', 'noise']
            for key in keys_to_reshape:
                batch[key] = batch[key].view(batch['x_t'].shape)

            for key in keys_to_match:
                batch[key][sample_ix, :num_bbs] = batch[key][sample_ix, tgt_ixs]

            for key in keys_to_reshape:
                batch[key] = batch[key].flatten(start_dim=0, end_dim=1)
        return batch
