import numpy as np


class ColMaker:
    def Configure(args):
        pass
    def LoadConfig(in):
        pass
    def SaveConfig(p_out):
        pass
    def FindSplit(depth, qexpand, gpair, p_fmat, p_tree):
        evaluator = tree_evaluator_.GetEvaluator()
        feat_set = tree_evaluator_.GetFeatureSet(depth)
        for batch in p_fmat.GetBatches():
            self.updateSolution(batch, feat_set.HostVector(), gpair, p_fmat)
        self.SyncBestSolution(qexpand)
        for nid in qexpand:
            e = snode_[nid]
            if e.best.loss_chg > kRtEps:
                left_leaf_weight = evaluator.CalcWeight(nid, param_, e.best.left_sum)*param_.lr
                right_leaf_weight = evaluator.CalcWeight(nid, param_, e.best.right_sum)*param_.lr
                p_tree.ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                                  e.best.DefaultLeft(), )
            else:
                (*p_tree)[nid].SetLeaf(e.weight * param_.lr)

    def SyncBestSolution(self):



class SplitEvaluation:
    def __init__(self, constraints, lower, upper, has_constraint=True):
        self.constraints = constraints
        self.has_constraint = has_constraint

    def CalcSplitGain(self, param, nidx, fidx, left, right):
        wleft = self.CalcWeight(nidx, param, left)
        wright = self.CalcWeight(nidx, param, right)
        gain = self.CalcGainGivenWeight(param, left, wleft) + \
               self.CalcGainGivenWeight(param, right, wright)
        return gain

    def CalcWeight(self, nidx, param, stats):
        sum_grad = stats.grad
        sum_hess = stats.hess
        dw = thresholdL1(sum_grad, param.reg_alpha)/ (sum_hess +
                                                      param.reg_lambda)
        if (param.max_delta_step != 0) and (np.abs(dw) > param.max_delta_step):
            dw = np.copysign(param.max_delta_step, dw)
        return dw

    def CalcGainGivenWeight(self, p, stats, w):
        sum_grad = stats.grad
        sum_hess = stats.hess
        if stats.hess <=0:
            return 0
        # avoiding - 2 (G*w + 0.5(H+lambda)*w^2 (using obj = G^2/(H+lambda))
        if not self.has_constraint:
            return (thresholdL1(sum_grad, p.reg_alpha)**2)/ (sum_hess +
                                                         p.reg_lambda)
        return -(2.0 * sum_grad * w + (sum_hess + p.reg_lambda) * (w*w));

    def CalcGain(self, p, stats):
        w = self.CalcWeight(p, stats)
        self.CalcGainGivenWeight(p, stats, w)

    def make_summary(self, data, weight):
        idx_sort = np.argsort(data)
        n = data.shape[0]
        data_s = data[idx_sort]
        weight_s = weight[idx_sort]
        size = 0
        # start update sketch
        wsum = 0
        # construct data_ with unique weights_
      #   for i in range(n):
      #       j = i + 1
      #       w = weight_s[i];
      #       while j < n and data_[j] == data_[i]:
      #           w += weight_s[j].weight
      #           j+=1
      #       .append()
      #       wsum += w
      #       i = j
      #
      #   out->data_[out->size++] = Entry(wsum, wsum + w, w, queue[i].value);
      #   wsum += w; i = j;
      # }


def thresholdL1(w, alpha):
    if w > + alpha:
        return w - alpha
    if w < - alpha:
        return w + alpha
    return 0.0



def Entry(rmin, rmax, wmin, value):
    pass

def CheckValid(eps=0):









