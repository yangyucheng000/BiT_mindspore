"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin
import time
import os

# import bit_pytorch.fewshot as fs
# import bit_pytorch.lbtoolbox as lb
# import bit_pytorch.models as models
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
import mindspore.dataset.vision.c_transforms as c_trans
import mindspore.dataset.transforms.c_transforms as c2_trans
import mindspore.common.initializer as weight_init
from mindspore.communication import get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.common import set_seed
import mindspore.nn.probability.distribution as msd
# `set_seed` function -- share parameters among multi-devices
# or use mindspore.context.set_auto_parallel_context(parameter_broadcast=True)
# according to the warning given by mindspore 

import lbtoolbox as lb
import models_ms_graph as models
# import fewshot as fs

import bit_common
import bit_hyperrule
from tqdm import tqdm

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="Ascend")
ms.context.set_context(device_id=0)

def topk(output, target, ks=(1,)):
  """Returns one boolean vector for each k, whether the target is within the output's top-k."""
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i

def mktrainval(args, logger):
    """Returns train and validation datasets."""
    precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
    # ========transforms========
    train_transform = [
        c_trans.Resize((precrop, precrop)),
        c_trans.RandomCrop((crop, crop)),
        c_trans.RandomHorizontalFlip(),
        c_trans.Normalize((0.5*255, 0.5*255, 0.5*255), (0.5*255, 0.5*255, 0.5*255)),
        c_trans.HWC2CHW()
    ]
    val_transform = [
        c_trans.Resize((crop, crop)),
        c_trans.Normalize((0.5*255, 0.5*255, 0.5*255), (0.5*255, 0.5*255, 0.5*255)),
        c_trans.HWC2CHW()
    ]
    type_cast_op = c2_trans.TypeCast(ms.int32) # cast label to int32 datatype

    # ========load data========
    if args.dataset == "cifar10":
        # "/home/ma-user/work/workspace_lc/dist_test/dataset/cifar-10-batches-bin"
        train_set = ds.Cifar10Dataset(args.datadir, "train", shuffle=True)
        valid_set = ds.Cifar10Dataset(args.datadir, "test", shuffle=True, num_samples=1000)
    else:
        # other datasets to be completed
        pass

    # ========about free shot========
    if args.examples_per_class is not None:
        # to be completed
        pass

    num_train, num_val = train_set.get_dataset_size(), valid_set.get_dataset_size()
    logger.info(f"Using a training set with {num_train} images.")
    logger.info(f"Using a validation set with {num_val} images.")

    micro_batch_size = args.batch // args.batch_split

    train_set = train_set.map(operations=train_transform, input_columns=["image"])
    train_set = train_set.map(operations=type_cast_op, input_columns=["label"])
    train_set = train_set.batch(batch_size=micro_batch_size)
    valid_set = valid_set.map(operations=val_transform, input_columns=["image"])
    valid_set = valid_set.map(operations=type_cast_op, input_columns=["label"])
    valid_set = valid_set.batch(batch_size=micro_batch_size)

    return train_set, valid_set, num_train, num_val

#============eval============
def run_eval(model, data_loader, logger, step):
    # to be completed
    
    pass

#============mixup=============
# to be completed
def mixup_data(x, y, l):
    """Returns mixed inputs, pairs of targets, and lambda"""
    ops_perm = ops.Randperm(max_length=x.shape[0])
    temp = ms.Tensor([x.shape[0]],ms.int32)
    indices = ops_perm(temp)
    # indices = np.random.permutation(x.shape[0]).tolist()
    mixed_x = l * x + (1 - l) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b

def mixup_criterion(criterion, pred, y_a, y_b, l):
    return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

# backbone with loss func
class NetWithLossMixup(nn.Cell):
    def __init__(self, backbone, loss_func):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_func
        # self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction="mean")
    
    # def construct(self, x, y, mixup_l):
    #     mixed_x, y_a, y_b = mixup_data(x, y, mixup_l)
    #     y_a = ops.stop_gradient(y_a)
    #     y_b = ops.stop_gradient(y_b)
    #     mixed_x = ops.stop_gradient(mixed_x)
    #     logits = self.backbone(mixed_x)
    #     return mixup_criterion(self.loss_fn, logits, y_a, y_b, mixup_l)

    def construct(self, mixed_x, y_a, y_b, mixup_l):
        logits = self.backbone(mixed_x)
        return mixup_criterion(self.loss_fn, logits, y_a, y_b, mixup_l)
        

class train_step(nn.TrainOneStepCell):
    """define training process"""
    def __init__(self, network, optimizer, mixup, args, logger):
       super().__init__(network, optimizer)
       self.grad = ops.GradOperation(get_by_list=True)
       self.mixup = mixup
       # self.create_mixupl = msd.Beta([self.mixup], [self.mixup], dtype=ms.float32)
       self.args = args
       self.logger = logger
    
    # def construct(self, data, label):
    #     weights = self.weights
    #     batch_split = self.args.batch_split

    #     if self.mixup > 0:
    #         x, y_a, y_b = mixup_data(data, label, self.create_mixupl.sample())
    #         c = self.net_loss_mixup(x, y_a, y_b, self.mixup_l)
    #         grads = self.grad(self.net_loss_mixup, weights)(x, y_a, y_b, self.mixup_l)
    #         grads = self.grad_reducer(grads)
    #     else:
    #         c = self.net_loss(data, label)
    #         grads = self.grad(self.net_loss, weights)(data, label)
    #         grads = self.grad_reducer(grads)
    #     c_num = float(c.asnumpy())
    #     self.optimizer(grads)
    #     return c

    def construct(self, x, y_a, y_b, mixup_l):
        weights = self.weights
        batch_split = self.args.batch_split

        if self.mixup > 0:
            #x, y_a, y_b = mixup_data(data, label, self.create_mixupl.sample())
            c = self.network(x, y_a, y_b, mixup_l)
            grads = self.grad(self.network, weights)(x, y_a, y_b, mixup_l)
            grads = self.grad_reducer(grads)
        else:
            c = self.network(x, y_a, y_b, mixup_l)
            grads = self.grad(self.network, weights)(x, y_a, y_b, mixup_l)
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return c

def init_weight(model):
    for _, cell in model.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                            cell.weight.shape,
                                                            cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                            cell.weight.shape,
                                                            cell.weight.dtype))

class net_eval(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.network = network
    
    def construct(self, data, label):
        out = self.network(data)
        return out, label

class eval_metric(nn.Metric):
    def __init__(self):
        super().__init__()
        self.clear()
    
    def clear(self):
        self.num_samples = 0
        self.num_correct = 0

    def update(self, *inputs):
        logits = inputs[0].asnumpy()
        y = inputs[1].asnumpy()
        y_pred = np.argmax(logits, 1)
        self.num_correct += (y_pred == y).sum()
        self.num_samples += y_pred.shape[0]
    
    def eval(self):
        return self.num_correct / self.num_samples, self.num_correct, self.num_samples



def main(args):
    logger = bit_common.setup_logger(args)

    train_set, valid_set, num_train, num_val = mktrainval(args, logger)
    logger.info(f"dataset batch size: {train_set.get_batch_size()}")

    #======dataparallel=======
    logger.info("Moving model onto all NPUs")
    ### model = torch.nn.DataParallel(model)

    #=======load weights for the model=======
    logger.info(f"Loading model from {args.model}.ckpt")
    model = models.KNOWN_MODELS[args.model](head_size=args.num_classes, zero_head=True)
    param_dict = ms.load_checkpoint("BiT-M-R50x1.ckpt")
    ms.load_param_into_net(model, param_dict)

    #=======optim==========
    total_step = bit_hyperrule.get_schedule(num_train*args.num_devices)[-1]
    lr_list = [bit_hyperrule.get_lr(step, num_train*args.num_devices, args.base_lr) for step in range(total_step)]
    #lr_list = [[bit_hyperrule.get_lr(step, num_train, args.base_lr)]*args.batch_split for step in range(total_step)]
    #lr_list = sum(lr_list, []) #setup for batch split

    # optim = nn.Momentum(model.trainable_params(), args.base_lr, momentum=0.9)
    optim = nn.Momentum(model.trainable_params(), lr_list, momentum=0.9)

    #========loss and other things=======
    # model.set_train()
    mixup = bit_hyperrule.get_mixup(num_train*args.num_devices)
    cri = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction="mean")

    logger.info("Starting training!")



    #============start training process==========
    net_loss_mixup = NetWithLossMixup(model, cri)

    train_net = train_step(net_loss_mixup, optim, mixup, args, logger)
    train_net.set_train()

    eval_net = net_eval(model)
    eval_net.set_train(False)
    metric = eval_metric()

    step_cnt = 0
    for x, y in recycle(train_set):
        if  step_cnt  >= total_step*args.batch_split:
            break
        create_mixupl = msd.Beta([mixup], [mixup], dtype=ms.float32)
        mixup_l = create_mixupl.sample() if mixup > 0 else 1
        mixed_x, y_a, y_b = mixup_data(x, y, mixup_l)
        loss = train_net(mixed_x, y_a, y_b,mixup_l)
        logger.info(f"[step {step_cnt}]: loss={float(loss.asnumpy()):.5f}")
        step_cnt += 1

        if args.eval_every and step_cnt % (args.eval_every*args.batch_split) == 0:
            logger.info("Starting evaluating!")
            with tqdm(total=valid_set.get_dataset_size()) as prog:
                for (_x, _y) in tqdm(valid_set):
                    logits, y_truth = eval_net(_x, _y)
                    metric.update(logits, y_truth)
                    prog.update(1)
            accuracy, temp1, temp2 = metric.eval()
            metric.clear()
            logger.info(f"Validation@{step_cnt} accuracy: {accuracy:.2%}({temp1}/{temp2})")
    
    # when training is done, do evaluation on the test set
    logger.info("Starting evaluating!")
    with tqdm(total=valid_set.get_dataset_size()) as prog:
        for (_x, _y) in tqdm(valid_set):
            logits, y_truth = eval_net(_x, _y)
            metric.update(logits, y_truth)
            prog.update(1)
    accuracy, temp1, temp2 = metric.eval()
    metric.clear()
    logger.info(f"Validation@end accuracy: {accuracy:.2%}({temp1}/{temp2})")

    # normal training way
    # cb = ms.train.callback.LossMonitor()
    # train_model = ms.Model(model, loss_fn=cri, optimizer=optim)
    # train_model.train(epoch=100, train_dataset=train_set, callbacks=cb)


def npz2ckpt(zero_head, model):
    weights = np.load("BiT-M-R50x1.npz")
    # print(type(weights))
    save_list = []
    save_list.append({"name": "root.conv.weight",
                      "data": ms.Parameter(ms.Tensor(weights["resnet/root_block/standardized_conv2d/kernel"]))})
    save_list.append({"name": "head.gn.gamma",
                      "data": ms.Parameter(ms.Tensor(weights["resnet/group_norm/gamma"]))})
    save_list.append({"name": "head.gn.beta",
                      "data": ms.Parameter(ms.Tensor(weights["resnet/group_norm/beta"]))})
    
    if zero_head == True:
        save_list.append({"name": "head.conv.weight",
                        "data": ms.Parameter(ms.Tensor(weights["resnet/head/conv2d/kernel"]))})
        save_list.append({"name": "head.conv.bias",
                        "data": ms.Parameter(ms.Tensor(weights["resnet/head/conv2d/bias"]))})
    
    convname = 'standardized_conv2d'
    for block_id in range(1,5):
        bname = f"block{block_id}"
        num_unit = [3, 4, 6, 3]
        for unit_id in range(1, num_unit[block_id-1]+1):
            prefix_sc = f"resnet/{bname}/unit{unit_id:02d}/"
            prefix_tg = f"body.{bname}.unit{unit_id:02d}."
            save_list.append({"name": prefix_tg + "conv1.weight",
                            "data": ms.Parameter(ms.Tensor(weights[prefix_sc+f"a/{convname}/kernel"]))})
            save_list.append({"name": prefix_tg + "conv2.weight",
                            "data": ms.Parameter(ms.Tensor(weights[prefix_sc+f"b/{convname}/kernel"]))})
            save_list.append({"name": prefix_tg + "conv3.weight",
                            "data": ms.Parameter(ms.Tensor(weights[prefix_sc+f"c/{convname}/kernel"]))})
            save_list.append({"name": prefix_tg + "gn1.gamma",
                            "data": ms.Parameter(ms.Tensor(weights[prefix_sc+f"a/group_norm/gamma"]))})
            save_list.append({"name": prefix_tg + "gn2.gamma",
                            "data": ms.Parameter(ms.Tensor(weights[prefix_sc+f"b/group_norm/gamma"]))})
            save_list.append({"name": prefix_tg + "gn3.gamma",
                            "data": ms.Parameter(ms.Tensor(weights[prefix_sc+f"c/group_norm/gamma"]))})
            save_list.append({"name": prefix_tg + "gn1.beta",
                            "data": ms.Parameter(ms.Tensor(weights[prefix_sc+f"a/group_norm/beta"]))})
            save_list.append({"name": prefix_tg + "gn2.beta",
                            "data": ms.Parameter(ms.Tensor(weights[prefix_sc+f"b/group_norm/beta"]))})
            save_list.append({"name": prefix_tg + "gn3.beta",
                            "data": ms.Parameter(ms.Tensor(weights[prefix_sc+f"c/group_norm/beta"]))})
            
            if(unit_id == 1):
                save_list.append({"name": prefix_tg + "downsample.weight",
                                "data": ms.Parameter(ms.Tensor(weights[prefix_sc+f"a/proj/{convname}/kernel"]))})
        
        ms.save_checkpoint(save_list, "BiT-M-R50x1.ckpt")


if __name__ =="__main__":
    parser = bit_common.argparser(models.KNOWN_MODELS.keys())
    parser.add_argument("--datadir", required=True,
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--num_devices", type=int, default=8,
                        help="Number of devices(GPU/NPU etc) to use.")
    # set_seed(0)
    main(parser.parse_args())
