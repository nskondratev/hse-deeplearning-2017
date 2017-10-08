from multiprocessing import Pool
import mxnet as mx
import cv2
from mtcnn.core.symbol import P_Net, R_Net, O_Net
from mtcnn.core.detector import Detector
from mtcnn.core.fcn_detector import FcnDetector
from mtcnn.tools.load_model import load_param
from mtcnn.core.MtcnnDetector import MtcnnDetector
import os
import time
import argparse


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


def apply_mtcnn_to_image(filename, root_folder, mtcnn_detector = None):
    print('[{}] Apply MTCNN to image...'.format(filename))
    if mtcnn_detector is None:
        mtcnn_detector = init_mtcnn_detector()
    print('[{}] MTCNN Detector is initialized.'.format(filename))
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
        for b in boxes_c:
            res += '{} {} {} {}\n'.format(int(b[0]), int(b[1]), int(b[2]), int(b[3]))

    print('[{}] Founded {} boxes. Finish.'.format(filename, faces_count))
    return res


def eval_mtcnn(input_folder):
    t1 = time.time()
    pool = Pool()
    # mtcnn_detector = init_mtcnn_detector()
    all_images = []
    for sub_folder in os.listdir(input_folder):
        all_images += [(os.path.join(sub_folder, x), input_folder, None) for x in
                       os.listdir(os.path.join(input_folder, sub_folder))]
    res = pool.starmap(apply_mtcnn_to_image, all_images)
    output_filename = '{}_results.txt'.format(input_folder)
    with open(output_filename, 'w') as f:
        f.write(''.join(res))
        print('Total time: ', time.time() - t1)
        print('Finish')


def parse_args():
    parser = argparse.ArgumentParser(description='MTCNN evaluation on folder')
    parser.add_argument('--folder', type=str, default='test_widerface')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # MTCNN model evaluation
    eval_mtcnn(args.folder)
