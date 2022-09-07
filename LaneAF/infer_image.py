import os
import json
from datetime import datetime
from statistics import mean
import argparse
import pdb

import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import datasets.transforms as tf

from datasets.tusimple import TuSimple, get_lanes_tusimple
from models.dla.pose_dla_dcn import get_pose_net
from models.erfnet.erfnet import ERFNet
from models.enet.ENet import ENet
from utils.affinity_fields import decodeAFs
from utils.metrics import match_multi_class, LaneEval
from utils.visualize import tensor2image, create_viz


parser = argparse.ArgumentParser('Options for inference with LaneAF models in PyTorch...')
parser.add_argument('--input_image', type=str, default='/home/hsuan/Thesis/LaneAF/test/20.jpg', help='path to input image')
parser.add_argument('--dataset-dir', type=str, default='/media/hsuan/data/TuSimple', help='path to dataset')
parser.add_argument('--output-dir', type=str, default='/home/hsuan/Thesis/LaneAF/infer_output', help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default='/home/hsuan/Thesis/LaneAF/laneaf-weights/tusimple-weights/dla34/net_0025.pth', help='path to pre-trained model snapshot')
parser.add_argument('--split', type=str, default='test', help='dataset split to evaluate on (train/val/test)')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--save-viz', action='store_true', default=True, help='save visualization depicting intermediate and final results')

args = parser.parse_args()
# check args
if args.input_image is None:
    assert False, 'Path to image not provided!'
if args.snapshot is None:
    assert False, 'Model snapshot not provided!'
if args.split is ['train', 'val', 'test']:
    assert False, 'Incorrect dataset split provided!'

# set batch size to 1 for visualization purposes
args.batch_size = 1

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M-infer")
    args.output_dir = os.path.join('.', 'experiments', 'tusimple', args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    pass
    #assert False, 'Output directory already exists!'

# load args used from training snapshot (if available)
if os.path.exists(os.path.join(os.path.dirname(args.snapshot), 'config.json')):
    with open(os.path.join(os.path.dirname(args.snapshot), 'config.json')) as f:
        json_args = json.load(f)
    # augment infer args with training args for model consistency
    if 'backbone' in json_args.keys():
        args.backbone = json_args['backbone']
    else:
        args.backbone = 'dla34'

# store config in output directory
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

# set random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 1}
# test_loader = DataLoader(TuSimple(args.dataset_dir, args.split, False), **kwargs)
# print(test_loader)
#pdb.set_trace()

# preprocessing image
def preprocessing(image,net):
    net.eval()
    # img preprocessing
    img=cv2.imread(image).astype(np.float32)/255
    img=cv2.resize(img[16:,:,:], (1280, 704), interpolation=cv2.INTER_LINEAR) # 圖片長寬要可以被 32 整除
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_transforms = transforms.Compose([
        tf.GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
        tf.GroupNormalize(mean=([0.485, 0.456, 0.406], (0, )), std=([0.229, 0.224, 0.225], (1, ))),
    ])
    input_img,_ = img_transforms((img,img))
    input_img = torch.from_numpy(input_img).permute(2,0,1).contiguous().float()
    input_img = np.expand_dims(input_img, axis=0).astype(np.float32)

    if args.cuda:
        input_img = torch.tensor(input_img).cuda()

    # do the forward pass
    outputs = net(input_img)[-1]

    # convert to arrays
    img = tensor2image(input_img.detach(), np.array([0.485, 0.456, 0.406]), np.array([0.229,0.224,0.225]))
    mask_out = tensor2image(torch.sigmoid(outputs['hm']).repeat(1, 3, 1, 1).detach(), 
        np.array([0.0 for _ in range(3)], dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
    vaf_out = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    haf_out = np.transpose(outputs['haf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))

    # print(vaf_out)
    # cv2.imshow("vaf_out",vaf_out)
    # cv2.waitKey(0)
    # decode AFs to get lane instances
    seg_out = decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, fg_thresh=128, err_thresh=5)

    # create video visualization
    if args.save_viz:
        img_out = create_viz(img, seg_out.astype(np.uint8), mask_out, vaf_out, haf_out)
        cv2.imshow("img_out",img_out)

        ##create_viz
        # scale = 8
        # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("img_out1",img)

        # img = np.ascontiguousarray(img, dtype=np.uint8)
        # seg = seg_out.astype(np.uint8)
        # print((seg))
        # seg_color = cv2.applyColorMap(40*seg, cv2.COLORMAP_JET)
        # cv2.imshow("img_out2",seg_color)
        # rows, cols = np.nonzero(seg)

        # for r, c in zip(rows, cols):
        #     img = cv2.arrowedLine(img, (c*scale, r*scale),(int(c*scale+vaf_out[r, c, 0]*scale*0.75), 
        #         int(r*scale+vaf_out[r, c, 1]*scale*0.5)), seg_color[r, c, :].tolist(), 1, tipLength=0.4)
        ##create_viz

        cv2.waitKey(0)
    return


if __name__ == "__main__":
    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    if args.backbone == 'dla34':
        model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)
    elif args.backbone == 'erfnet':
        model = ERFNet(heads=heads)
    elif args.backbone == 'enet':
        model = ENet(heads=heads)

    model.load_state_dict(torch.load(args.snapshot), strict=True)
    if args.cuda:
        model.cuda()
    print(model)

    preprocessing(args.input_image,model)

