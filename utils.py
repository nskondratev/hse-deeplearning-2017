import numpy as np
import re

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
    distances = [];
    N = len(box_true)
    for box1 in box_found:
        for box2 in box_true:
            distances.append(jaccard_distance(box1, box2))
    distances.sort()
    score = (np.sum(distances[:N]))/N
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
    # Compile needed regexps
    box_pattern = re.compile('^([0-9]+ ){9}[0-9]+\n$')

    with open(output_file_path, 'w') as out_f:
        with open(path_to_file, 'r') as inp_f:
            cur_line = inp_f.readline()
            while cur_line:
                # File path is provided
                if cur_line.endswith('.jpg\n'):
                    out_f.write(cur_line)
                # Box description is provided
                elif box_pattern.match(cur_line):
                    l = cur_line.strip().split(' ')
                    box = []
                    box.append(int(l[0]))
                    box.append(int(l[1]))
                    box.append(box[0] + int(l[2]))
                    box.append(box[1] + int(l[3]))
                    box = [str(x) for x in box]
                    box = ' '.join(box)
                    out_f.write(box+'\n')
                cur_line = inp_f.readline()

