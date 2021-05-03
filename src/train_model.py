"""
Brain uc3 training with pyecvl/pyeddl.
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
import wandb
from functools import partial

MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")

def main(args):
   
    np.random.seed(42)

    size = [ args.shape, args.shape]  # size of images

    thresh = 0.5
    miou_best = -1

    if args.runs_dir:
        os.makedirs(os.path.join(args.runs_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.runs_dir, "checkpoints"), exist_ok=True)
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

    config = dict (
         learning_rate = args.lr,
         batch_size = args.batch_size,
         weight_decay = args.weight_decay,
         architecture = "UNet",
         dataset_id = "uc3-tensored",
         infra = 'HPC4AI',
    )

    run = wandb.init(
       name=args.name,
       project="uc3-eddl",
       notes="baseline",
       tags=["baseline","UNet"],
       config=config,
    )


    eddl.build(
        net,
        eddl.adam(args.lr, weight_decay = args.weight_decay),#eddl.sgd(args.lr,momentum=0.9,weight_decay = args.weight_decay),#
        ["mean_squared_error"],
        ["mean_absolute_error"],
        eddl.CS_GPU(args.gpu, mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )
    #eddl.summary(net)
    log_filepath = "uc_3"
    if args.runs_dir:
        log_filepath = os.path.join(args.runs_dir, "logs", "uc_3")
    eddl.setlogfile(net, log_filepath)

    if args.resume_ckpts and os.path.exists(args.resume_ckpts):
        print("Loading checkpoints '{}'".format(args.resume_ckpts))
        eddl.load(net, args.resume_ckpts, 'bin')

    print("Reading dataset")


    dataset_train = UC3_Dataset(os.path.join(args.in_ds,'input_tensored'),os.path.join(args.in_ds,args.target), num_channels,num_channels_gt, size, is_test=False, gpu=net.dev)
    dataset_test = UC3_Dataset(os.path.join(args.in_ds,'input_tensored'),os.path.join(args.in_ds,args.target), num_channels,num_channels_gt, size, is_test=True, gpu=net.dev)


    num_samples_train = len(dataset_train)
    print('Train Samples: ',num_samples_train)
    num_batches_train = num_samples_train // args.batch_size

    batch_size_val = 2
    num_samples_validation = len(dataset_test)
    num_batches_validation = num_samples_validation // batch_size_val
    seen_samples_val = batch_size_val * num_batches_validation


    miou_evaluator = utils.Evaluator()

    def mse_loss_cumulative_grads(b_size,inputs):
        print('diff',inputs[0])
        diff = eddl.Sub(inputs[0], inputs[1]);
        print('diff',diff)
        return eddl.Div(eddl.Mult(diff, diff),b_size);
    mse_loss_cg = partial(mse_loss_cumulative_grads, args.batch_size)
    
    #out out
    #target = eddl.Input([num_channels_gt, size[0], size[1]])
    #target.info()
    #exit()
    #mse_loss = eddl.newloss(mse_loss_cg, [out, target], "mse_loss");

    print("Starting training")
    for e in range(args.epochs):
        print("Epoch {:d}/{:d} - Training".format(e + 1, args.epochs),
              flush=True)

        train_loss, train_acc = 0.0 , 0.0
        eddl.set_mode(net, 1)
        shuffling_idx = np.random.permutation(len(dataset_train))

        start_e_time = time.time()
        for i, b in enumerate(range(num_batches_train)):#
            start_time_e = time.time()

            '''
            tr_loss = 0
            eddl.zeroGrads(net)
            for sample in range(args.batch_size):
                x,y = LoadBatch(dataset_train, b , batch_size = 1 , permutation = shuffling_idx)

                tx, ty = [x], [y]
                eddl.forward(net, tx)

                output = eddl.getOutput(net.lout[0])

                #print(dir(output))
                #exit()
                print(net.layers[1].name)
                #print(dir(net.layers[1]))
                print(net.layers[1].gradients)


                # Log first image pair
                if b==0 and sample == 0:
                    examples = [wandb.Image(output.getdata(), caption="Train-Output"),wandb.Image(y.getdata(), caption="Train-"+args.target)]

                print(dir(y))
                print(dir(net))
                target = eddl.Input(y,'target',net.dev,net.mem_level)

                mse_loss = eddl.newloss(mse_loss_cg, [out, target], "mse_loss")
                #Tensor.copy(y,target.input)                                
                tr_loss += eddl.compute_loss(mse_loss)

                print(tr_loss)
                #eddl.backward(net,[y])
                
                print(np.sum(np.abs(net.layers[1].gradients[0].getdata())))
            '''    

            eddl.zeroGrads(net)
            eddl.reset_loss(net)
            x,y = LoadBatch(dataset_train, b , batch_size = args.batch_size , permutation = shuffling_idx)
            tx, ty = [x], [y]
            #eddl.train_batch(net, tx, ty, range(num_samples_train))
            eddl.forward(net, tx)
            if b==0 :
                output = eddl.getOutput(out)
                examples = [wandb.Image(output.getdata()[0], caption="Train-Output"),wandb.Image(y.getdata()[0], caption="Train-"+args.target)]
                examples += [wandb.Image(output.getdata()[1], caption="2Train-Output"),wandb.Image(y.getdata()[1], caption="2Train-"+args.target)]
            eddl.backward(net,ty)
            eddl.update(net)
            

            if i % args.log_interval == 0:
                print("Epoch {:d}/{:d} (batch {:d}/{:d}) - ".format(
                    e + 1, args.epochs, b + 1, num_batches_train
                ), end="", flush=True)
                eddl.print_loss(net, b)
                print()
            #train_loss += tr_loss 
            train_loss += eddl.get_losses(net)[0]
            train_acc  += eddl.get_metrics(net)[0]


        tttse = (time.time() - start_e_time)
        train_loss = train_loss / num_batches_train
        train_acc  = train_acc / num_batches_train
        print("---Time to Train - Single Epoch: %s seconds ---" % tttse)
        print("---Train Loss:\t\t%s ---" % train_loss)
        print("---Train Accuracy:\t%s ---" % train_acc)

        ## Sheduler logic
        ## ...
        ## eddl.setlr(net, 0.0001)


        print("Epoch %d/%d - Evaluation" % (e + 1, args.epochs), flush=True)
        start_time_e = time.time()
        miou_evaluator.ResetEval()
        metric = eddl.getMetric("mean_absolute_error")
        error = eddl.getLoss("mean_squared_error")
        val_loss, val_acc = 0.0 , 0.0
        eddl.set_mode(net, 0)
        for b in range(num_batches_validation):#
            start_time_e = time.time()
            print("Epoch {:d}/{:d} (batch {:d}/{:d}) ".format(
                e + 1, args.epochs, b + 1, num_batches_validation
            ), end="", flush=True)
            x,y = LoadBatch(dataset_test, b , batch_size = batch_size_val)

            eddl.forward(net, [x])

            output = eddl.getOutput(net.lout[0])

            # Log first image pair
            if b==0:
                examples.append(wandb.Image(output.getdata()[0], caption="Validation-Output"))
                examples.append(wandb.Image(y.getdata()[0], caption="Validation-"+args.target))
                examples.append(wandb.Image(output.getdata()[1], caption="2Validation-Output"))
                examples.append(wandb.Image(y.getdata()[1], caption="2Validation-"+args.target))


            # ! .value return a sum of scores
            val_acc += metric.value(output, y)
            val_loss += error.value(output, y)
            for k in range(output.shape[0]):#args.batch_size):
                pred = output.select([str(k)])
                gt = y.select([str(k)])
                pred_np = np.array(pred, copy=False)
                gt_np = np.array(gt, copy=False)
                miou_evaluator.BinaryIoU(pred_np, gt_np, thresh=thresh)
                if args.save_images and args.runs_dir:
    
                    pred_np *= 255
                    pred_ecvl = ecvl.TensorToView(pred)
                    pred_ecvl.colortype_ = ecvl.ColorType.GRAY
                    pred_ecvl.channels_ = "xyc"
 
                    filename = dataset_test.img_list[b*args.batch_size + k].replace('/','-').replace('.npy','.jpg')
                    filepath = os.path.join(args.runs_dir, 'images', filename)

                    # "xyc" 
                    img_hmap = cv2.applyColorMap(np.uint8(np.array(pred_ecvl)), cv2.COLORMAP_JET)
                    img_hmap = ecvl.Image.fromarray(img_hmap,"xyc",ecvl.ColorType.RGB)
                    ecvl.ImWrite(filepath, img_hmap)
                    examples.append(wandb.Image(pred_np, caption="Validation-HeatMap"))

            print()
        print("---Evaluation Epoch takes %s seconds ---" % (time.time() - start_time_e))
        val_loss = val_loss / seen_samples_val
        val_acc  = val_acc / seen_samples_val
        miou = miou_evaluator.MeanMetric()
        print("---MIoU Score: %.6g" % miou)
        print("---Validation Loss:\t\t%s ---" % val_loss)
        print("---Validation Accuracy:\t%s ---" % val_acc)
        if miou > miou_best:
            print("Saving weights")
            checkpoint_path = os.path.join(args.runs_dir, "checkpoints", "dh-uc3_epoch_{}_miou_{}.bin".format(e+1, miou))
            eddl.save(net, checkpoint_path, "bin")
            miou_best = miou
        run.log({"train_time_epoch": tttse, "train_loss": train_loss, "train_mae": train_acc, "val_loss": val_loss, "val_mae": val_acc , "examples": examples})
    print("---Time to Train: %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_ds", metavar="INPUT_DATASET")
    parser.add_argument("--target", help='TTP or CBF or CBV', metavar="INPUT_TARGET", default='TTP')
    parser.add_argument("--epochs", type=int, metavar="INT", default=250)#250
    parser.add_argument("--batch-size", type=int, metavar="INT", default=8)#8
    parser.add_argument("--shape", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, metavar="INT", default=1)
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--runs-dir", default='outputs', metavar="DIR", help="if set, save images, checkpoints and logs in this directory")
    parser.add_argument("--resume_ckpts", type=str)
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="full_mem")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--name", type=str,  default='uc3_train')
    main(parser.parse_args())
