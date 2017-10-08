import numpy as np
import re
import pickle
import os
from multiprocessing import Pool

BOX_PATTERN = re.compile('^([0-9]+ ){9}[0-9]+( )?\n$')
TRANSFORMED_BOX_PATTERN = re.compile('^([0-9]+ ){3}[0-9]+\n$')

def jaccard_distance(box1, box2):
    """Compute IoU between detect box and gt boxes
    Parameters:
    ----------
    box1: x1, y1, x2, y2
    box2: x1, y1, x2, y2
    Returns:
    -------
    returns: Jaccard distance
    """
    S1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    S2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    xx1 = np.maximum(box1[0], box2[0])
    yy1 = np.maximum(box1[1], box2[1])
    xx2 = np.minimum(box1[2], box2[2])
    yy2 = np.minimum(box1[3], box2[3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    return inter / (S1 + S2 - inter)


def calc_img_score(box_found, box_true):
    distances = []
    N = len(box_true)
    for box1 in box_found:
        for box2 in box_true:
            distances.append(jaccard_distance(box1, box2))
    distances.sort(reverse=True)
    dist_sum = np.sum(distances[:N])
    score = (dist_sum) / N
    return score


def transform_annotations_file(path_to_file, output_file_path=None):
    # Generate output file path if it is not provided
    if output_file_path is None:
        f = path_to_file.split('.')
        ext = f.pop()
        f.append('_transformed')
        f.append(ext)
        output_file_path = '.'.join(f)

    print('Outfile: {}'.format(output_file_path))
    with open(output_file_path, 'w') as out_f, open(path_to_file, 'r') as inp_f:
        cur_line = inp_f.readline()
        while cur_line:
            # File path is provided
            if cur_line.endswith('.jpg\n'):
                out_f.write(cur_line)
            # Box description is provided
            elif BOX_PATTERN.match(cur_line):
                l = cur_line.strip().split(' ')
                box = []
                box.append(int(l[0]))
                box.append(int(l[1]))
                box.append(box[0] + int(l[2]))
                box.append(box[1] + int(l[3]))
                box = [str(x) for x in box]
                box = ' '.join(box)
                out_f.write(box + '\n')
            cur_line = inp_f.readline()


def parse_image_data(filename, use_pickled = False):
    print('[{}] Parse image data'.format(filename))
    pickled_filename = '{}.pkl'.format(filename)
    if use_pickled and os._exists(pickled_filename):
        with open(pickled_filename, 'rb') as f:
            res = pickle.load(f)
            return res
    else:
        with open(filename, 'r') as f:
            cur_line = f.readline()
            res = {}
            cur_filename = None
            while cur_line:
                # File path is provided
                if cur_line.endswith('.jpg\n'):
                    fn = cur_line.strip()
                    res[fn] = []
                    cur_filename = fn
                # Box description is provided
                elif TRANSFORMED_BOX_PATTERN.match(cur_line):
                    res[cur_filename].append([int(x) for x in cur_line.strip().split(' ')])
                cur_line = f.readline()
            print('[{}] Finish parsing image data'.format(filename))
            if use_pickled:
                with open(pickled_filename, 'wb') as pf:
                    pickle.dump(res, pf)
            return res


def calc_total_score(result_filename, truth_filename):
    print('Calc total score. Results filename: {}, truth filename: {}'.format(result_filename, truth_filename))
    # Parse results and truth files
    to_parse = [(result_filename, False), (truth_filename, True)]
    pool = Pool()
    (result_data, truth_data) = pool.starmap(parse_image_data, to_parse)
    pool.close()
    pool.join()
    # Calc score
    print('Start calculating score...')
    pool = Pool()
    to_calc = []
    for filename in result_data:
        to_calc.append((result_data[filename], truth_data[filename]))
    scores = pool.starmap(calc_img_score, to_calc)
    pool.close()
    pool.join()
    n = len(scores)
    return float(sum(scores)) / float(n)
