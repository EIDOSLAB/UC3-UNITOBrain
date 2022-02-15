import numpy as np
from scipy.stats import pearsonr
from scipy.spatial import distance


class Evaluator:
    def __init__(self):
        self.eps = 1e-06
        self.buf = []

    def ResetEval(self):
        self.buf = []

    def IoU(self, a, b, thresh=0.5):
        if thresh:
            a = Threshold(a, thresh)
            b = Threshold(b, thresh)
            intersection = np.count_nonzero(np.logical_and(a, b))#.sum()
            union = np.count_nonzero(np.logical_or(a, b))#.sum() 
            rval = intersection / (union + self.eps)
        else:
            intersection = (a * b).sum()
            #not boolean OR, so needs to subtract intersection
            union = (a.sum() + b.sum()) - intersection 
            rval = intersection / (union + self.eps)
        self.buf.append(rval)
        return rval


    def DiceCoefficient(self, a, b, thresh=0.5):
        a = Threshold(a, thresh)
        b = Threshold(b, thresh)
        intersection = np.count_nonzero(np.logical_and(a, b))#.sum()
        rval = (2 * intersection + self.eps) / (np.count_nonzero(a) + np.count_nonzero(b) + self.eps)

        self.buf.append(rval)
        return rval

    def PearsonCorrelation(self, a, b):

        rval,_ = pearsonr(a.flatten(),b.flatten())

        self.buf.append(rval)
        return rval

    def MIoU(self):
        if not self.buf:
            return 0
        return sum(self.buf) / len(self.buf)

    MeanMetric = MIoU


def Threshold(a, thresh=0.5):
    return (a >= thresh)
    #.astype(int)
    #a[a >= thresh] = 1
    #a[a < thresh] = 0
    #return a


def ImageSqueeze(img):
    k = img.dims_.index(1)
    img.dims_ = [_ for i, _ in enumerate(img.dims_) if i != k]
    img.strides_ = [_ for i, _ in enumerate(img.strides_) if i != k]
    k = img.channels_.find("z")
    img.channels_ = "".join([_ for i, _ in enumerate(img.channels_) if i != k])
