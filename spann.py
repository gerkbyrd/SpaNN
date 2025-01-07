import sys
import time
import os
import math
import copy
import warnings
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
from torch.utils import model_zoo
from torchvision import datasets, transforms
import PIL
from PIL import Image, ImageDraw

import nets.resnet
import nets.attack_detector
from darknet import *
from helper import *

import json
from tqdm import tqdm
import argparse
import joblib

from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
from skimage.restoration import inpaint

parser = argparse.ArgumentParser()
parser.add_argument("--save",action='store_true',help="save results to txt")
parser.add_argument("--savedir",default='NN_based_imgclass/',type=str,help="save path")
#victim model (obj. det.)
parser.add_argument("--cfg",default="cfg/yolo.cfg",type=str,help="relative directory to cfg file")
parser.add_argument("--weightfile",default="weights/yolo.weights",type=str,help="path to checkpoints")

parser.add_argument("--performance_det",action='store_true',help="save detection performance (time) per frame")

parser.add_argument("--effective_files",default=None,type=str,help="file with list of effective adv examples")
parser.add_argument("--geteff",action='store_true',help="save array with effective attack names")
parser.add_argument("--uneffective",action='store_true',help="use only uneffective attacks")


parser.add_argument("--clean",action='store_true',help="use clean images")
parser.add_argument("--bypass_det",action='store_true',help="skip detection stage")
parser.add_argument("--bypass",action='store_true',help="skip recovery stage")

parser.add_argument("--lim",default=1000000,type=int,help="limit on number of images/frames to process")
parser.add_argument('--imgdir', default="inria/Train/pos", type=str,help="path to data")
parser.add_argument('--patch_imgdir', default="inria/Train/pos", type=str,help="path to adversarially patched data")
parser.add_argument('--ground_truth', default=None, type=str,help="path to patched data")

parser.add_argument('--dataset', default='inria', choices=('inria','voc','imagenet','cifar'),type=str,help="dataset")
parser.add_argument("--skip",default=1,type=int,help="number of example to skip")
parser.add_argument("--det_net",default='2dcnn',type=str,help="model for detection")
parser.add_argument("--det_net_path",default='checkpoints/2dcnn_raw_imagenet_atk_det.pth',type=str,help="path to trained detector model")
parser.add_argument("--nn_det_threshold",default=0.5,type=float,help="decision threshold for NN detector (beta star in paper)")
parser.add_argument("--iou_thresh",default=0.5,type=float,help="iou threshold for effective attack definition")

parser.add_argument("--save_scores",action='store_true',help="save detection scores")
parser.add_argument("--n_patches",default='1',type=str,help="number of patches (just an informative string to save results)")

parser.add_argument("--dbscan_eps", default=1.0, type=float, help="how close to cluster two neurons?")
parser.add_argument("--dbscan_min_pts", default=4, type=int, help="how many neurons is a cluster?")
parser.add_argument("--cluster_thresh",default=1,type=int,help="how many clsuters are too many for a clean image?")#1
parser.add_argument("--binarize",action='store_true',help="binarize feature maps before clustering")
parser.add_argument("--bin_mag", default=1.0, type=float, help="magnitude to scale binarized neurons (deprecated)")
parser.add_argument("--beta_sweep",action='store_true',help="do a sweep over beta (default: 0.2 to 0.95 in 0.05 steps)")
parser.add_argument("--scale_var",action='store_true',help="standardize variance before clustering")
parser.add_argument("--scale_mean",action='store_true',help="center mean before clustering")

parser.add_argument('--ensemble_step', default=5, type=int, help='100/threshold set size')
parser.add_argument('--inpainting_step', default=5, type=int, help='100/threshold set size')
parser.add_argument("--eval_class",action='store_true',help="match class for iou evaluation")

parser.add_argument('--remove', default='_', type=str, help='clusfeat to remove (ablation study)')
args = parser.parse_args()

warnings.filterwarnings("ignore")
print("Setup...")
if args.dataset in ['cifar', 'imagenet']:
    model = nets.resnet.resnet50(pretrained=True,clip_range=None,aggregation=None)
elif args.dataset in ['inria', 'voc']:
    cfgfile = args.cfg
    weightfile = args.weightfile
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    model = model.eval().cuda()
    img_size = model.height
    ious={'clean':[], 'random':[], 'adversarial':[]}

mn, std= args.scale_mean, args.scale_var
imgdir=args.imgdir
patchdir=args.patch_imgdir
device = 'cuda'
#build and initialize model
if args.dataset == 'imagenet':
    ds_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    denorm=transforms.Normalize([-0.485/.229, -0.456/.224, -0.406/.225], [1/0.229, 1/0.224, 1/0.225])
    ds_transforms_patch = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ds_transforms_inp = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()
elif args.dataset == 'cifar':
    ds_transforms = transforms.Compose([
        transforms.Resize(192),#
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    denorm=transforms.Normalize([-0.4914/.2023, -0.4822/.1994, -0.4465/.2010], [1/0.2023, 1/0.1994, 1/0.2010])
    ds_transforms_patch = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds_transforms_inp = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('./checkpoints/resnet50_192_cifar.pth')
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval()

#if torch.cuda.is_available() else 'cpu'

if args.det_net=='1dcnn':
    net = nets.attack_detector.attack_detector()#net = nets.resnet.resnet50(pretrained=True)
elif args.det_net=='2dcnn':
    net = nets.attack_detector.cnn()#net = nets.resnet.resnet50(pretrained=True)
elif args.det_net=='2dcnn_raw':
    net = nets.attack_detector.cnn_raw(in_feats=(not 'impneu' in args.remove) + (not 'nclus' in args.remove) +
                                                (not 'avg' in args.remove) + (not 'std' in args.remove))
elif args.det_net=='mlp':
    net = nets.attack_detector.mlp()#net = nets.resnet.resnet50(pretrained=True)
elif args.det_net=='mlp+':
    net = nets.attack_detector.mlp(in_size=4)#net = nets.resnet.resnet50(pretrained=True)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
print('loading detector net')
assert os.path.isdir('./checkpoints'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.det_net_path)
net.load_state_dict(checkpoint['net'])
net.eval()



all_atk=[]
clean_corr=0
detected=0
success_atk=0
kount=0
iou_thresh=args.iou_thresh
mask_ious=[]

if args.effective_files!=None:
    eff_files=list(np.load(os.path.join(patchdir, args.effective_files)))
    eff_files=[x.split('.')[0] for x in eff_files]
else:
    eff_files = None

if args.save_scores:
    score_array=[]
if args.performance_det:
    perf_array_clus=[]
    perf_array_det=[]

def beta_iteration(beta, fm, raw=True):

    binarized_fm=np.array(fm>=np.max(fm)*beta, dtype='float32')

    x,y=np.where(binarized_fm>0)
    thing=np.hstack((x.reshape(-1,1),y.reshape(-1,1)))#,binarized_fm.flatten().reshape(-1,1)))
    thing = StandardScaler(with_mean=mn, with_std=std).fit_transform(thing)
    cluster=DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_pts).fit(thing)

    #data1=thing
    clusters= np.unique(cluster.labels_)#, return_counts=True)

    #continue
    #centroids=[]
    avg_ic_d=[]
    biggie=[]
    for cluster_label in clusters:
        if cluster_label==-1:
            continue
        data_c=thing[np.where(cluster.labels_==cluster_label)]
        biggie.append(data_c)
        data_samp=data_c[np.random.choice([i for i in range(len(data_c))], size=min(1000, len(data_c)), replace=False)]
        dmx=distance_matrix(data_samp, data_samp)
        dmx=dmx[np.tril_indices(dmx.shape[0], k=-1)]
        if len(dmx):
            avg_ic_d.append(np.mean(dmx))
    if len(avg_ic_d):
        avg_intracluster_d=np.mean(avg_ic_d)
        avg_intracluster_std=np.std(avg_ic_d)
    else:
        avg_intracluster_d=0
        avg_intracluster_std=0

    if not raw:
        feat_stack=[]

        if not 'nclus' in args.remove:
            feat_stack.append(len([x!=-1 for x in clusters]))
        if not 'avg' in args.remove:
            feat_stack.append(avg_intracluster_d)
        if not 'std' in args.remove:
            feat_stack.append(avg_intracluster_std)
        if not 'impneu' in args.remove:
            feat_stack.append(binarized_fm.sum())
        #return biggie, [len([x!=-1 for x in clusters]), avg_intracluster_d, avg_intracluster_std, binarized_fm.sum()]
        return biggie, feat_stack
    return biggie

val_dataset=sorted(os.listdir(imgdir)[:min(args.lim, len(os.listdir(imgdir)))])

if args.dataset in ['inria', 'voc']:
    for imgfile in tqdm(val_dataset):
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            nameee=imgfile.split('.')[0]
            if (eff_files != None and nameee not in eff_files and not args.uneffective) or (eff_files != None and args.uneffective and nameee in eff_files):
                continue
            patchfile = os.path.abspath(os.path.join(patchdir, imgfile))
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            padded_img = Image.open(imgfile).convert('RGB')
            w,h=padded_img.size
            transform = transforms.ToTensor()
            padded_img = transform(padded_img).cuda()
            img_fake_batch = padded_img.unsqueeze(0)
            clean_boxes, feature_map = do_detect(model, img_fake_batch, 0.4, 0.4, True, direct_cuda_img=True)
            clean_boxes=clean_boxes
            #if nothing is detected in the clean version of the image...
            if not(len(clean_boxes)):
                continue
            kount=kount+1

            cbb=[]
            for cb in clean_boxes:
                cbb.append([T.detach() for T in cb])

            if not args.clean:
                data = Image.open(patchfile).convert('RGB')
                patched_img_cpu=data
                patched_img = transform(patched_img_cpu).cuda()
                p_img = patched_img.unsqueeze(0)
                adv_boxes, feature_map  = do_detect(model, p_img, 0.4, 0.4, True, direct_cuda_img=True)
                adb = []
                for ab in adv_boxes:
                    adb.append([T.detach() for T in ab])
            else:
                p_img=img_fake_batch
                candigatos=[]
                adv_boxes=clean_boxes
                adb=cbb

            """
            DETECTION
            """
            if args.bypass_det:
                condition=True
            else:# and not analysed:
                if args.performance_det:
                    start=time.process_time()
                fm_np=feature_map[0].detach().cpu().numpy()
                fm=np.sum(fm_np,axis=0)
                biginfo=[]
                vector_s=[]
                for beta in [0.0+x*0.01 for x in range(0,100,args.ensemble_step)]:#cifar until beta=0.45, imgnet until beta=0.25
                    biggie, clus_feats=beta_iteration(beta, fm, raw=False)
                    vector_s.append(clus_feats)
                    biginfo.append(biggie)
                if args.performance_det:
                    end=time.process_time()
                    perf_array_clus.append(end-start)
                vector_s=np.array(vector_s).reshape((1, len(vector_s), len(clus_feats)))
                detector_input = 2*nn.functional.normalize(torch.Tensor(clustering_data_preprocessing(vector_s, skip=True)), dim=2, p=float('inf')) - 1

                with torch.no_grad():
                    detector_output=net(detector_input.to(device))
                detection_score=detector_output.detach().cpu().numpy()
                condition=detection_score>=args.nn_det_threshold
                if args.performance_det:
                    end=time.process_time()
                    perf_array_det.append(end-start)
                if args.save_scores:
                    score_array.append(detection_score)
                if condition:
                    detected=detected+1
                """
                COMPUTE RESULTS
                """
                best_arr=[]
                for i in range(len(clean_boxes)):
                    ious['clean'].append(best_iou(cbb, [T.detach() for T in clean_boxes[i]]))
                    best=best_iou(adb, [T.detach() for T in clean_boxes[i]], args.eval_class)
                    best_arr.append(best)
                    ious['adversarial'].append(best)
                suc_atk=False
                for b in best_arr:
                    if b<iou_thresh:#clean_pred==labels[i]:
                        suc_atk=True
                        success_atk = success_atk + 1
                        if args.geteff:
                            all_atk.append(nameee)
                        break
                if not suc_atk:
                    clean_corr=clean_corr+1

elif args.dataset in ['cifar', 'imagenet']:
    for imgfile in tqdm(val_dataset):
        if imgfile.endswith('.jpg') or imgfile.endswith('.png') or imgfile.endswith('.JPEG'):
            nameee=imgfile.split('.')[0]
            if (eff_files != None and nameee not in eff_files and not args.uneffective) or (eff_files != None and args.uneffective and nameee in eff_files):
                continue
            kount=kount+1
            if args.ground_truth is not None:
                labelfile=os.path.abspath(os.path.join(args.ground_truth, imgfile.split('.')[0] + '.npy'))
                label=np.load(labelfile)
            else:
                #can do this as long as we deal only with images where the victim model predicts the right label when there is no attack (as the ones in the repository)
                imgfilelab = os.path.abspath(os.path.join(imgdir, imgfile))
                datalab = Image.open(imgfilelab).convert("RGB")
                datalab = ds_transforms(datalab).cuda()
                output_clean, feature_map = model(datalab.unsqueeze(0).detach())
                output_clean, feature_map = output_clean.detach().cpu().numpy()[0], feature_map.detach().cpu().numpy()[0]
                global_feature = np.mean(output_clean, axis=(0,1))
                pred_list = np.argsort(global_feature,kind='stable')
                label = pred_list[-1]
            if args.clean:
                imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
                data = Image.open(imgfile).convert("RGB")
                data = ds_transforms(data).cuda()
            else:
                imgfile = os.path.abspath(os.path.join(patchdir, imgfile.split('.')[0] + '.png'))
                data = Image.open(imgfile).convert("RGB")
                data = ds_transforms_patch(data).cuda()

            gt_mask=np.zeros((data.shape[1], data.shape[2]))
            data = data.unsqueeze(0)

            output_clean, feature_map = model(data)
            output_clean, feature_map = output_clean.detach().cpu().numpy()[0], feature_map.detach().cpu().numpy()[0]
            """
            DETECTION
            """
            if args.bypass_det:
                condition=True
            else:
                if args.performance_det:
                    start=time.process_time()
                fm=np.sum(feature_map,axis=0)
                biginfo=[]
                vector_s=[]
                for beta in [0.0+x*0.01 for x in range(0,100,args.ensemble_step)]:
                    biggie, clus_feats=beta_iteration(beta, fm, raw=False)
                    vector_s.append(clus_feats)
                    biginfo.append(biggie)

                #input(len(biginfo))
                if args.performance_det:
                    end=time.process_time()
                    perf_array_clus.append(end-start)
                vector_s=np.array(vector_s).reshape((1, len(vector_s), len(clus_feats)))
                detector_input = 2*nn.functional.normalize(torch.Tensor(clustering_data_preprocessing(vector_s, skip=True)), dim=2, p=float('inf')) - 1

                with torch.no_grad():
                    detector_output=net(detector_input.to(device))
                detection_score=detector_output.detach().cpu().numpy()
                condition=detection_score>=args.nn_det_threshold

                if args.performance_det:
                    end=time.process_time()
                    perf_array_det.append(end-start)

                if args.save_scores:
                    score_array.append(detection_score)
                if condition:
                    detected=detected+1
            """
            COMPUTE RESULTS
            """
            global_feature = np.mean(output_clean, axis=(0,1))
            pred_list = np.argsort(global_feature,kind='stable')
            clean_pred = pred_list[-1]
            if clean_pred==label:
                clean_corr = clean_corr + 1
            else:
                success_atk = success_atk + 1
                if args.geteff:
                    all_atk.append(nameee)

torch.cuda.empty_cache()
if args.save_scores:
    deer=os.path.join(args.savedir)
    if not os.path.exists(deer):
        os.makedirs(deer)
    fname=deer + '_' + args.dataset + '_' + args.det_net + '_npatches_' + str(args.n_patches) + '_ens_' + str(args.ensemble_step) + '_scores'
    if args.clean:
        fname = fname + '_clean'
    with open(fname + '.npy', 'wb') as f:
        np.save(f, np.array(score_array))

if args.performance_det:
    deer=os.path.join(args.savedir)
    if not os.path.exists(deer):
        os.makedirs(deer)
    fname=deer + '_' + args.dataset + '_' + args.det_net + '_npatches_' + str(args.n_patches) + '_ens_' + str(args.ensemble_step) + '_perfs'
    if args.clean:
        fname = fname + '_clean'
    with open(fname + '.npy', 'wb') as f:
        np.save(f, np.array(perf_array_det))

    fname=deer + '_' + args.dataset + '_' + args.det_net + '_npatches_' + str(args.n_patches) + '_ens_' + str(args.ensemble_step) + '_clusperfs'
    if args.clean:
        fname = fname + '_clean'
    with open(fname + '.npy', 'wb') as f:
        np.save(f, np.array(perf_array_clus))

if args.geteff:
    deer=os.path.join(args.patch_imgdir)
    if not os.path.exists(deer):
        os.makedirs(deer)
    fname=deer + 'effective'+ '_' + str(args.n_patches) + 'p'
    if args.clean:
        fname = fname + '_clean'
    with open(fname + '.npy', 'wb') as f:
        np.save(f, np.array(all_atk))
print(kount)
line1="Unsuccesful Attacks:"+str(clean_corr/max(1,kount))
line2="Detected Attacks:" + str(detected/max(1,kount))
line3="Successful Attacks:" + str(success_atk/max(1,kount))
print(line1)
print(line2)
print(line3)

print("------------------------------")
#print(lines)
if args.save:
    deer=os.path.join(args.savedir)
    if not os.path.exists(deer):
        os.makedirs(deer)
    txtpath = deer + args.det_net + '_npatches_' + str(args.n_patches) + '_det_thr_' + str(args.nn_det_threshold) + '_inp_' + str(args.inpainting_step)
    if args.clean:
        txtpath = txtpath + '_clean'
    with open(txtpath + '.txt', 'w+') as f:
        f.write('\n'.join([line1, line2, line3, "------------------------------"]))
