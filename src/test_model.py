# Copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia
# (UNIMORE), AImageLab
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

"""
Brain uc3 inference with pyecvl/pyeddl.
https://github.com/deephealthproject/pyecvl
"""

import argparse
import os
import random
import time
import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import utils 
from models import UNet
from dataloader import UC3_Dataset, LoadBatch
import cv2

MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")

def main(args):
   
    np.random.seed(42)

    size = [ args.shape, args.shape]  # size of images

    thresh = 0.5
    miou_best = -1

    if args.runs_dir:
        os.makedirs(os.path.join(args.runs_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.runs_dir, "logs"), exist_ok=True)

    num_channels = 89
    num_channels_gt = 1

    in_ = eddl.Input([num_channels, size[0], size[1]])
    out = UNet(in_, num_channels_gt)
    net = eddl.Model([in_], [out])

    count = 0
    for layer in net.layers:
        for params in layer.params:
            count += params.size
    print("Number of trainable parameters: {}".format(count))


    eddl.build(
        net,
        eddl.adam(0.001),
        ["mean_squared_error"],
        ["mean_absolute_error"],
        eddl.CS_GPU(args.gpu, mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )
    eddl.summary(net)
    log_filepath = "test_uc_3"
    if args.runs_dir:
        log_filepath = os.path.join(args.runs_dir, "logs", "test_uc_3")
    eddl.setlogfile(net, log_filepath)

    if not os.path.exists(args.ckpts):
        raise RuntimeError('Checkpoint "{}" not found'.format(args.ckpts))
    eddl.load(net, args.ckpts, "bin")

    print("Reading dataset")

    dataset_test = UC3_Dataset(os.path.join(args.in_ds,'input_tensored'),os.path.join(args.in_ds,args.target), num_channels,num_channels_gt, size, is_test=True, gpu=args.gpu)

    batch_size_val = 1
    num_samples_test = len(dataset_test)
    num_batches_test = num_samples_validation // batch_size_val
    seen_samples_val = batch_size_val * num_batches_validation

    print("Testing")
    eddl.set_mode(net, 0)

    miou_evaluator = utils.Evaluator()
    pearson_evaluator = utils.Evaluator()
    dice_evaluator = utils.Evaluator()
    test_loss, test_acc = 0.0 , 0.0
    start_time = time.time()

    for b in range(num_batches_test):
        print("Batch {:d}/{:d} ".format(
            b + 1, num_batches_test), end="", flush=True)
        x,y = LoadBatch(dataset_test, b , batch_size = batch_size_val)

        eddl.forward(net, [x])
        output = eddl.getOutput(net.lout[0])

        # ! .value return a sum of scores
        test_acc += metric.value(output, y)
        test_loss += error.value(output, y)

        for k in range(output.shape[0]):#args.batch_size):
            pred = output.select([str(k)])
            gt = y.select([str(k)])
            pred_np = np.array(pred, copy=False)
            gt_np = np.array(gt, copy=False)
            miou_evaluator.BinaryIoU(pred_np, gt_np, thresh=thresh)
            pearson_evaluator.PearsonCorrelation(pred_np, gt_np)
            dice_evaluator.DiceCoefficient(pred_np, gt_np)
            if args.save_images and args.runs_dir:
                pred_np *= 255
                pred_ecvl = ecvl.TensorToView(pred)
                pred_ecvl.colortype_ = ecvl.ColorType.GRAY
                pred_ecvl.channels_ = "xyc"

                filename = os.path.join('test-',dataset_test.img_list[b*args.batch_size + k].replace('/','-').replace('.npy','.jpg'))
                filepath = os.path.join(args.runs_dir, 'images', filename)

                # "xyc" 
                img_hmap = cv2.applyColorMap(np.uint8(np.array(pred_ecvl)), cv2.COLORMAP_JET)
                img_hmap = ecvl.Image.fromarray(img_hmap,"xyc",ecvl.ColorType.RGB)
                ecvl.ImWrite(filepath, img_hmap)

        print()

    iou = miou_evaluator.MeanMetric()
    pearson = pearson_evaluator.MeanMetric()
    dice = dice_evaluator.MeanMetric()

    test_loss = test_loss / seen_samples_val
    test_acc  = test_acc / seen_samples_val
    print("Loss:\t", test_loss)
    print("MSE:\t", test_acc)
    print("IntersectionOverUnion Score:\t", iou)
    print("Pearson Correlation Score:\t", pearson)
    print("Dice Score:\t", dice)
    print("---Time to Inference: %s seconds ---" % (time.time() - start_time))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_ds", metavar="INPUT_DATASET")
    parser.add_argument("--ckpts", metavar='CHECKPOINTS_PATH',
                        default='checkpoints/dh-uc3_epoch_200_miou_1.bin')
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1)
    parser.add_argument("--shape", type=int, default=512)
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument("--runs-dir", default='outputs', help="if set, save images, checkpoints and logs in this directory")
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="full_mem")
    main(parser.parse_args())
