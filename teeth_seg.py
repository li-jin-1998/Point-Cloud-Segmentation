import argparse
import os
import shutil
import sys
from datetime import datetime

import convpoint.knn.cpp.nearest_neighbors as nearest_neighbors
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree
from tqdm import tqdm

import utils.metrics as metrics
from dataset import TeethDataset
from utils.process import load_seg, save_ply_property


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pc_normalize(pc):
    # l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def nearest_correspondance1(pts_src, pts_dest, data_src):
    # start = time.time()
    tree = BallTree(pts_src, leaf_size=2)
    _, indices = tree.query(pts_dest, k=1)
    indices = indices.ravel()
    data_dest = data_src[indices]
    # print(time.time() - start)
    return data_dest


def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    # start = time.time()
    indices = nearest_neighbors.knn(pts_src.copy(), pts_dest.copy(), K, omp=True)
    if K == 1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    # print(time.time() - start)
    return data_dest


def get_model(model_name, input_channels, output_channels):
    if model_name == "SegSmall":
        from networks.network_seg import SegSmall as Net
    if model_name == "SegBig":
        from networks.network_seg import SegBig as Net
    return Net(input_channels, output_channels)


THREADS = 4
USE_CUDA = True
N_CLASSES = 4
EPOCHS = 200


def train(args):
    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    data_train, labels, data_num_train = load_seg(os.path.join(args.dataset, 'train.h5'))
    print("Done", data_train.shape)

    print("Computing class weights (if needed, 1 otherwise)...")
    if args.weighted:
        frequences = []
        for i in range(4):
            frequences.append((labels == i).sum())
        frequences = np.array(frequences)
        frequences = frequences.mean() / frequences
    else:
        frequences = [1 for _ in range(4)]
    print(frequences)
    weights = torch.FloatTensor(frequences)
    if USE_CUDA:
        weights = weights.cuda()
    print("Done")

    print("Creating network...")
    net = get_model(args.model, input_channels=1, output_channels=N_CLASSES)
    net.cuda()
    print("parameters", count_parameters(net))

    ds = TeethDataset(data_train, data_num_train, labels, npoints=args.npoints)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=True,
                                               num_workers=THREADS
                                               )

    params_to_optimize = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adamax(params_to_optimize, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, last_epoch=-1, gamma=0.99, verbose=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, MILESTONES)

    # create the model folder
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    root_folder = os.path.join(args.savedir, 'log',
                               "{}_b{}_pts{}_weighted{}_{}".format(args.model, args.batchsize, args.npoints,
                                                                   args.weighted, time_string))
    os.makedirs(root_folder, exist_ok=True)

    train_loss = []
    # create the log file
    logs = open(os.path.join(root_folder, "log.txt"), "w")
    for epoch_num in range(1, EPOCHS + 1):
        cm = np.zeros((N_CLASSES, N_CLASSES))
        t = tqdm(train_loader, file=sys.stdout)
        for pts, features, seg, indices in t:

            if USE_CUDA:
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()

            optimizer.zero_grad()
            outputs = net(features, pts)

            loss = F.cross_entropy(outputs.view(-1, N_CLASSES), seg.view(-1))

            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            outputs_np = outputs.cpu().detach().numpy()

            output_np = np.argmax(outputs_np, axis=2).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
            cm += cm_

            oa = "{:.3f}".format(metrics.stats_overall_accuracy(cm))
            aa = "{:.3f}".format(metrics.stats_accuracy_per_class(cm)[0])

            t.set_postfix(OA=oa, AA=aa)
            t.desc = "[train epoch {}] loss: {:.4f}".format(epoch_num, np.mean(train_loss))
        scheduler.step()
        # save the model
        torch.save(net.state_dict(), os.path.join(root_folder, "state_dict.pth"))
        torch.save(net.state_dict(), os.path.join(args.savedir, "state_dict.pth"))
        # write the logs
        logs.write("{} {:.4f} {} {} \n".format(epoch_num, np.mean(train_loss), oa, aa))
        logs.flush()

    logs.close()


def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    return IoU


def val(args):
    args.data_folder = os.path.join(args.dataset, "test_data")

    # create the output folders
    result_path = os.path.join(args.savedir, 'predict')

    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    txt_file = os.path.join(args.dataset, 'test_label.txt')
    file_list = [os.path.join(result_path, line.strip()) for line in open(txt_file)][:2000]
    input_filelist = [os.path.join(os.path.dirname(txt_file), 'data', line.strip()) for line in open(txt_file)][:2000]

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    data, labels, data_num = load_seg(os.path.join(args.dataset, 'test.h5'))

    net = get_model(args.model, input_channels=1, output_channels=N_CLASSES)
    net.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")))
    net.cuda()
    net.eval()

    ds = TeethDataset(os.path.join(args.dataset, 'test.h5'), npoints=args.npoints, num_iter_per_shape=args.ntree)
    test_loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=False,
                                              num_workers=THREADS
                                              )

    t = tqdm(test_loader, ncols=120)

    predictions = [None for _ in range(data.shape[0])]
    predictions_max = [[] for _ in range(data.shape[0])]
    with torch.no_grad():

        for pts, features, seg, indices in t:

            if USE_CUDA:
                features = features.cuda()
                pts = pts.cuda()

            outputs = net(features, pts)

            indices = np.int32(indices.numpy())
            outputs = np.float32(outputs.cpu().numpy())

            # save results
            for i in range(pts.size(0)):

                # shape id
                shape_id = indices[i]

                # pts_src
                pts_src = pts[i].cpu().numpy()

                # pts_dest
                point_num = data_num[shape_id]
                pts_dest = data[shape_id]
                pts_dest = pts_dest[:point_num]

                # get the segmentation correspongin to part range
                seg_ = outputs[i]

                # interpolate to original points
                seg_ = nearest_correspondance(pts_src, pts_dest, seg_)

                if predictions[shape_id] is None:
                    predictions[shape_id] = seg_
                else:
                    predictions[shape_id] += seg_

                predictions_max[shape_id].append(seg_)

    for i in range(len(predictions)):
        a = np.stack(predictions_max[i], axis=1)
        a = np.argmax(a, axis=2)
        a = np.apply_along_axis(np.bincount, 1, a, minlength=6)
        predictions_max[i] = np.argmax(a, axis=1)

    # compute labels
    for i in range(len(predictions)):
        predictions[i] = np.argmax(predictions[i], axis=1)
    print(len(predictions), len(file_list))

    if args.save_ply:
        for i in tqdm(range(len(predictions))):
            # read the coordinates from the txt file for verification
            import pyvista as pv
            coordinates = pv.read(input_filelist[i]).points
            assert (data_num[i] == len(coordinates))
            save_ply_property(np.array(coordinates), np.array(predictions[i]), file_list[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=1, action="store_true")
    parser.add_argument("--save_ply", default=True, help="save ply files (test mode)")
    parser.add_argument("--savedir", default="./results", type=str)
    parser.add_argument("--dataset", type=str,
                        default='/mnt/algo_storage_server/PointCloudSeg/Dataset', required=False)
    parser.add_argument("--batchsize", "-b", default=16, type=int)
    parser.add_argument("--ntree", default=2, type=int)
    parser.add_argument("--npoints", default=5000, type=int)
    parser.add_argument("--weighted", default=0, action="store_true")
    parser.add_argument("--model", default="SegBig", help='SegSmall/SegBig', type=str)
    args = parser.parse_args()

    print(args)

    if args.test:
        print('**************************')
        print('test')
        print('**************************')
        val(args)
    else:
        print('**************************')
        print('train')
        print('**************************')
        train(args)


if __name__ == '__main__':
    main()
