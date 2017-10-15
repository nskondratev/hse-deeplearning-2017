from multiprocessing import Pool
import os

import matlab.engine

eng = matlab.engine.start_matlab()

# Tiny Face Detector functions
def apply_tiny_fd_to_image(filename, root_folder):
    boxes_c = eng.tiny_fd(os.path.join(root_folder, filename))
    res = filename + '\n'
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
        all_images += [(os.path.join(sub_folder, x), input_folder, None) for x in
                       os.listdir(os.path.join(input_folder, sub_folder))]
    res = pool.starmap(apply_tiny_fd_to_image, all_images)
    output_filename = '{}_tiny_fd_results.txt'.format(input_folder)
    with open(output_filename, 'w') as f:
        f.write(''.join(res))
        print('Finish')