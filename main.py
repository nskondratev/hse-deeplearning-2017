from multiprocessing import Pool
import mxnet as mx
import cv2
from mtcnn.core.symbol import P_Net, R_Net, O_Net
from mtcnn.core.detector import Detector
from mtcnn.core.fcn_detector import FcnDetector
from mtcnn.tools.load_model import load_param
from mtcnn.core.MtcnnDetector import MtcnnDetector
import os


def apply_mtcnn_to_image(filename, prefix=None, epoch=None, batch_size=None, ctx=None,
                         thresh=None, min_face_size=24,
                         stride=2, slide_window=False):
    # Set default prefix
    if prefix is None:
        prefix = ['mtcnn/model/pnet', 'mtcnn/model/rnet', 'mtcnn/model/onet']
    # Set default epoch
    if epoch is None:
        epoch = [16, 16, 16]
        # Set default thresh
        thresh = [0.5, 0.5, 0.7]
    # Set default context
    if ctx is None:
        ctx = mx.cpu(0)

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

    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    img = cv2.imread(filename)
    t1 = time.time()

    boxes, boxes_c = mtcnn_detector.detect_pnet(img)
    boxes, boxes_c = mtcnn_detector.detect_rnet(img, boxes_c)
    boxes, boxes_c = mtcnn_detector.detect_onet(img, boxes_c)

    print('time: ', time.time() - t1)

    if boxes_c is not None:
        draw = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for b in boxes_c:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 1)
            cv2.putText(draw, '%.3f' % b[4], (int(b[0]), int(b[1])), font, 0.4, (255, 255, 255), 1)

        f = filename.split('.')
        extension = f.pop()
        f.append('_annotated')
        f.append(extension)
        f = '.'.join(f)

        cv2.imwrite(f, draw)
        cv2.waitKey(0)


def eval_mtcnn(input_folder):
    pool = Pool()
    all_images = []
    for subfolder in os.listdir(input_folder):
        all_images += [os.path.join(subfolder, x) for x in os.listdir(os.path.join(subfolder))]



if __name__ == '__main__':
# TODO Implement MTCNN model evaluation
