import matlab.engine

import time
import argparse
from multiprocessing import Pool, cpu_count
import os
import pickle
import math

import mxnet as mx
import cv2

from mtcnn.core.symbol import P_Net, R_Net, O_Net
from mtcnn.core.detector import Detector
from mtcnn.core.fcn_detector import FcnDetector
from mtcnn.tools.load_model import load_param
from mtcnn.core.MtcnnDetector import MtcnnDetector

from utils import calc_total_score, calc_img_score, chunks


# Tiny Face Detector functions
def apply_tiny_fd_to_images(files, root_folder):
    print('Before start MATLAB engine. Files count: {}'.format(len(files)))
    t1 = time.time()
    eng = matlab.engine.start_matlab()
    me_started = time.time() - t1
    print('Started MATLAB engine in {} seconds'.format(me_started))
    res = ''
    for filename in files:
        print('[{}] Start processing image...'.format(filename))
        boxes_c = eng.tiny_fd(os.path.join(root_folder, filename))
        res += filename + '\n'
        faces_count = 0
        if boxes_c is not None:
            faces_count = len(boxes_c)
            for b in boxes_c:
                res += '{} {} {} {}\n'.format(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        print('[{}] Founded {} boxes. Finish.'.format(filename, faces_count))
    return res


def eval_tiny_fd(input_folder):
    pool = Pool()
    all_images = []
    for sub_folder in os.listdir(input_folder):
        all_images += [os.path.join(sub_folder, x) for x in
                       os.listdir(os.path.join(input_folder, sub_folder))]
    chunk_size = int(math.ceil(float(len(all_images)) / float(cpu_count())))
    chunks_images = chunks(all_images, chunk_size)
    to_process = [(filenames, input_folder) for filenames in chunks_images]
    res = pool.starmap(apply_tiny_fd_to_images, to_process)
    pool.close()
    pool.join()
    output_filename = '{}_tiny_fd_results.txt'.format(input_folder)
    with open(output_filename, 'w') as f:
        f.write(''.join(res))
        print('Finish')


# MTCNN functions
def init_mtcnn_detector(prefix=None, epoch=None, batch_size=None, ctx=None,
                        thresh=None, min_face_size=12,
                        stride=2, slide_window=False):
    # Set default prefix
    if prefix is None:
        prefix = ['mtcnn/model/pnet', 'mtcnn/model/rnet', 'mtcnn/model/onet']
    # Set default epoch
    if epoch is None:
        epoch = [16, 16, 16]
    # Set default thresh
    if thresh is None:
        thresh = [0.5, 0.5, 0.7]
    # Set default context
    if ctx is None:
        ctx = mx.cpu(0)
    # Set default batch size
    if batch_size is None:
        batch_size = [2048, 256, 16]

    detectors = [None, None, None]

    # load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
    if slide_window:
        PNet = Detector(P_Net("test"), 12, batch_size[0], ctx, args, auxs)
    else:
        PNet = FcnDetector(P_Net("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    args, auxs = load_param(prefix[1], epoch[0], convert=True, ctx=ctx)
    RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
    detectors[1] = RNet

    # load onet model
    args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
    ONet = Detector(O_Net("test"), 48, batch_size[2], ctx, args, auxs)
    detectors[2] = ONet
    return MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face_size,
                         stride=stride, threshold=thresh, slide_window=slide_window)


def apply_mtcnn_to_image(filename, root_folder, mtcnn_detector=None, save_processed_image=False):
    print('[{}] Apply MTCNN to image...'.format(filename))
    if mtcnn_detector is None:
        mtcnn_detector = init_mtcnn_detector()
    img = cv2.imread(os.path.join(root_folder, filename))

    t1 = time.time()

    boxes, boxes_c = mtcnn_detector.detect_pnet(img)
    boxes, boxes_c = mtcnn_detector.detect_rnet(img, boxes_c)
    boxes, boxes_c = mtcnn_detector.detect_onet(img, boxes_c)

    print('time: ', time.time() - t1)

    res = filename + '\n'
    faces_count = 0
    if boxes_c is not None:
        faces_count = len(boxes_c)
        if save_processed_image:
            draw = img.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
        for b in boxes_c:
            res += '{} {} {} {}\n'.format(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            if save_processed_image:
                cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 1)
                cv2.putText(draw, '%.3f' % b[4], (int(b[0]), int(b[1])), font, 0.4, (255, 255, 255), 1)

        if save_processed_image:
            with open('wider_face_val_bbx_gt._transformed.txt.pkl', 'rb') as f:
                truth = pickle.load(f)
                truth_boxes = truth[filename]
                for b in truth_boxes:
                    cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 1)
                score = calc_img_score(boxes_c, truth_boxes)
                cv2.putText(draw, '%.3f' % score, (10, 10), font, 0.4, (255, 255, 255), 1)
                cv2.imwrite('processed_{}'.format(os.path.basename(filename)), draw)

    print('[{}] Founded {} boxes. Finish.'.format(filename, faces_count))
    return res


def eval_mtcnn(input_folder):
    pool = Pool()
    all_images = []
    for sub_folder in os.listdir(input_folder):
        all_images += [(os.path.join(sub_folder, x), input_folder, None) for x in
                       os.listdir(os.path.join(input_folder, sub_folder))]
    res = pool.starmap(apply_mtcnn_to_image, all_images)
    output_filename = '{}_mtcnn_results.txt'.format(input_folder)
    with open(output_filename, 'w') as f:
        f.write(''.join(res))
        print('Finish')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation on folder')
    parser.add_argument('--folder', type=str, default='widerface')
    parser.add_argument('--report_filename', type=str, default=None)
    parser.add_argument('--model', type=str, default='mtcnn')
    parser.add_argument('--gt_filename', type=str, default='wider_face_val_bbx_gt._transformed.txt')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Model evaluation
    t1 = time.time()
    print('Start evaluation {} on folder {}'.format(args.model, args.folder))
    if args.model == 'mtcnn':
        eval_mtcnn(args.folder)
    elif args.model == 'tiny_fd':
        eval_tiny_fd(args.folder)
    execution_time = time.time() - t1
    print('Total time: ', execution_time)
    results_filename = '{}_{}_results.txt'.format(args.folder, args.model)
    score = calc_total_score(results_filename, args.gt_filename)
    print('Total score: {}'.format(score))
    if args.report_filename is None:
        report_filename = 'report_{}.txt'.format(args.model)
    else:
        report_filename = args.report_filename
    with open(report_filename, 'w') as rf:
        rf.write('==== {} results ====\nExecution time: {} seconds\nTotal score: {}\n'
                 .format(args.model, execution_time, score))
