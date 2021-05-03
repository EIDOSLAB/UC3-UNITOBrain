# Copyright (c) 2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pyeddl.eddl as eddl


def LeNet(in_layer, num_classes):
    x = in_layer
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 20, [5, 5])), [2, 2], [2, 2])
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 50, [5, 5])), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.ReLu(eddl.Dense(x, 500))
    x = eddl.Softmax(eddl.Dense(x, num_classes))
    return x


def VGG16(in_layer, num_classes):
    x = in_layer
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 64, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 128, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 256, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.ReLu(eddl.Dense(x, 4096))
    x = eddl.ReLu(eddl.Dense(x, 4096))
    x = eddl.Softmax(eddl.Dense(x, num_classes))
    return x


def SegNet(in_layer, num_classes):
    x = in_layer
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.Conv(x, num_classes, [3, 3], [1, 1], "same")
    return x


def SegNetBN(x, num_classes):
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])

    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.Conv(x, num_classes, [3, 3], [1, 1], "same")

    return x


def UNet(x, num_classes):
    depth = 64

    # encoder
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.AveragePool(x, [2, 2], [2, 2])#eddl.MaxPool(x, [2, 2], [2, 2])#
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.AveragePool(x2, [2, 2], [2, 2])#eddl.MaxPool(x2, [2, 2], [2, 2])#
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.AveragePool(x3, [2, 2], [2, 2])#eddl.MaxPool(x3, [2, 2], [2, 2])#
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x5 = eddl.AveragePool(x4, [2, 2], [2, 2])#eddl.MaxPool(x4, [2, 2], [2, 2])#

    # middle conv
    x5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x5, 16*depth, [3, 3], [1, 1], "same"), True))
    x5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x5, 16*depth, [3, 3], [1, 1], "same"), True))

    # decoder
    x5 = eddl.Conv(
        eddl.UpSampling(x5, [2, 2]), 8*depth, [2, 2], [1, 1], "same"
    )
    x4 = eddl.Concat([x4, x5])
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.Conv(
        eddl.UpSampling(x4, [2, 2]), 4*depth, [2, 2], [1, 1], "same"
    )
    x3 = eddl.Concat([x3, x4])
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.Conv(
        eddl.UpSampling(x3, [2, 2]), 2*depth, [2, 2], [1, 1], "same"
    )
    x2 = eddl.Concat([x2, x3])
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.Conv(
        eddl.UpSampling(x2, [2, 2]), depth, [2, 2], [1, 1], "same"
    )
    x = eddl.Concat([x, x2])
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))

    # final conv
    x = eddl.Sigmoid(eddl.Conv(x, num_classes, [1, 1]))

    # Kaiming Init
    def he(p):
        for pp in p.parent:
            he(pp)
        aus = [ eddl.HeUniform(l,seed=42) for l in p.parent] 
        for a in aus:
            a.initialize()
        p.parent = aus

    he(x)

    return x
